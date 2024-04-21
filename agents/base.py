import torch
import torch.nn as nn
import models
import warnings
import os
from utils import * 

class Base():
    def __init__(self, args, logger,):
        self.logger = logger
        self.args = args
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss()
        self.save_path = self.args.exp_dir + f'/original_model_{self.args.time_str}_{self.args.seed}.pth'
        if self.args.retrain:
            self.save_path = self.args.exp_dir + f'/retrain_model_{self.args.time_str}_{self.args.seed}.pth'

    def train(self):
        self.model = get_model(self.args)
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        train_loader, val_loader, test_loader = get_dataloader(self.args)
        iter_per_epoch = len(train_loader)
        if self.checkpoint_exists(self.save_path):
            self.load_checkpoint(self.model, self.save_path)
            self.logger.info(f'Loaded model from checkpoint {self.save_path}')
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), 
                                        lr=self.args.lr,
                                        momentum=self.args.momentum, 
                                        weight_decay=self.args.wd)
            
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones, gamma=self.args.gamma)
            warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * self.args.warm)
            self._train(train_loader, val_loader, optimizer, scheduler, warmup_scheduler, self.criterion)

        self.test_by_class(self.model, test_loader)

    # def get_dataloader(self):
    #     train_loader, test_loader = get_dataloader(self.args)

    #     indices = np.arange(len(train_loader.dataset))
    #     a = np.split(indices,[int(len(indices)*0.9), int(len(indices))])
    #     idx_train = a[0]
    #     idx_val = a[1]
    #     train_sampler = torch.utils.data.SubsetRandomSampler(idx_train)
    #     val_sampler = torch.utils.data.SubsetRandomSampler(idx_val)

    #     train_loader = torch.utils.data.DataLoader(train_loader.dataset,
    #                                                batch_size=self.args.batch_size,
    #                                                sampler=train_sampler)
        
    #     val_loader = torch.utils.data.DataLoader(train_loader.dataset,
    #                                              batch_size=self.args.batch_size,
    #                                              sampler=val_sampler)
        
    #     test_loader = torch.utils.data.DataLoader(test_loader.dataset,
    #                                             batch_size=self.args.batch_size,
    #                                             shuffle=False)
        
    #     if self.args.retrain:
    #         train_loader, _ = self.get_unlearn_dataloader(train_loader)
    #         val_loader, _ = self.get_unlearn_dataloader(val_loader)
            
    #     return train_loader, val_loader, test_loader
    

    def get_unlearn_dataloader(self, data_loader):
        dataset = data_loader.dataset
        _indices = data_loader.sampler.indices

        if self.args.dataset.lower() == 'svhn':
            train_targets = np.array(dataset.labels)[_indices]
        else:
            train_targets = np.array(dataset.targets)[_indices]
        unlearn_indices, remain_indices = [], []
        for i, target in enumerate(train_targets):
            if target in self.args.unlearn_class:
                unlearn_indices.append(i)
            else:
                remain_indices.append(i)

        # conver to the absolute index
        unlearn_indices = np.array(_indices)[unlearn_indices]
        remain_indices = np.array(_indices)[remain_indices]

        unlearn_sampler = torch.utils.data.SubsetRandomSampler(unlearn_indices)
        unlearn_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.args.batch_size,
                                                 sampler = unlearn_sampler,)

        remain_sampler = torch.utils.data.SubsetRandomSampler(remain_indices)
        remain_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=self.args.batch_size,
                                                    sampler = remain_sampler)
        return remain_loader, unlearn_loader


    @staticmethod
    def save_checkpoint(model, save_path):
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)
    
    @staticmethod
    def checkpoint_exists(save_path):
        return os.path.exists(save_path)
    
    @staticmethod
    def load_checkpoint(model, save_path):
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(torch.load(save_path))
        else:
            model.load_state_dict(torch.load(save_path))

    def _train(self, train_loader, val_loader, optimizer, scheduler, warmup_scheduler, criterion):
        best_acc = 0
        for ep in range(self.args.num_epochs):
            self.model.train()
            for batch, (x, y) in enumerate(train_loader):
                x = x.cuda()
                y = y.cuda()
                pred_y = self.model(x)
                loss = criterion(pred_y, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if ep < self.args.warm:
                    warmup_scheduler.step()
            
            if ep >= self.args.warm:
                scheduler.step()
            val_acc = test(self.model, val_loader)
            train_acc = test(self.model, train_loader)
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = deepcopy(self.model)
                p = self.args.patience
                self.save_checkpoint(self.model, self.save_path)
                self.logger.info('save best model to {} with acc {:.3f}'.format(self.save_path, val_acc))
            else:
                p -= 1
                if p == 0:
                    break
            self.logger.info('[train] epoch {}, train_acc {:.3f}, val_acc {:.3f}, lr {:.4f}'.format(ep, train_acc, val_acc, optimizer.param_groups[0]['lr']))
        self.model.load_state_dict(best_model.state_dict())
        self.logger.info('best_acc: {:.3f}'.format(best_acc))
    
    @staticmethod
    def test(model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(test_loader):
                x = x.cuda()
                y = y.cuda()
                pred_y = model(x)
                predicted = torch.argmax(pred_y.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct / total

    def test_by_class(self, model, test_loader, i=3):
        log_msg = ""
        model.eval()
        correct = 0
        total = 0
        y_list = []
        pred_y_list = []
        acc_lst = []
        with torch.no_grad():
            for batch, (x, y) in enumerate(test_loader):
                x = x.cuda()
                y = y.cuda()
                pred_y = model(x)
                predicted = torch.argmax(pred_y.data, 1)
                y_list.append(y)
                pred_y_list.append(predicted)

            y_list = torch.cat(y_list)
            pred_y_list = torch.cat(pred_y_list)
            for c in torch.unique(y_list):
                class_indices = torch.where(y_list == c)[0]
                correct = torch.sum(pred_y_list[class_indices] == y_list[class_indices]).item()
                total = len(class_indices)
                acc = correct / total
                acc_lst.append(acc)
                log_msg += f'{acc:.3f}\t'
            
            acc_lst = acc_lst[:i] + acc_lst[i+1:]
            mean_acc = np.mean(acc_lst)
            self.logger.info(log_msg)
            self.logger.info(f'mean_acc: {mean_acc:.3f}')