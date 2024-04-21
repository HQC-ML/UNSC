from .base import Base
import torch
import torch.nn as nn
from utils import *

class Boundary_Unlearn(Base):
    def __init__(self, args, logger, agent_marker=None):
        super().__init__(args, logger, agent_marker)
        self.save_path = self.args.exp_dir + f'boundary_unlearn_model_{self.args.seed}.pth'

    def load_original_model(self):
        original_model = get_model(self.args)
        original_model_save_path = self.args.exp_dir + f'original_model_{self.args.seed}.pth'
        self.load_checkpoint(original_model, 
                             save_path=original_model_save_path )
        self.logger.info(f'Loaded model from checkpoint {original_model_save_path}')
        return original_model
    
    def train(self):
        self.logger.info('Training unlearned model')
        self.original_model = self.load_original_model()
        train_loader, val_loader, test_loader, class_loader_dict = self.get_unlearn_dataloader()
        
        self.logger.info('Training Acc of Original Model')
        self.test_by_class(self.original_model, train_loader)
        self.logger.info('Testing Acc of Original Model')
        self.test_by_class(self.original_model, test_loader)
        
        TODO: Unfinished
    
    
    def find_null_space_proj(self, model, class_loader_dict):
        for cls_id in range(self.args.num_classes): 
            if cls_id == self.args.unlearn_class:
                continue
            for batch, (x, y) in enumerate(class_loader_dict[cls_id]):
                x = x.cuda()
                y = y.cuda()
                mat_list = self.get_representation_matrix(model,  
                                                     x,
                                                     batch_list=self.batch_list)
                break
            threshold = 0.97 + 0.003*cls_id
            merged_feat_mat = update_GPM(mat_list, threshold, merged_feat_mat)
            proj_mat = [torch.Tensor(np.dot(layer_basis, layer_basis.transpose())) for layer_basis in merged_feat_mat]
        return proj_mat
    
    def get_unlearn_dataloader(self):
        train_loader, val_loader, test_loader = self.get_original_dataloader()
        class_loader_dict = self.get_class_loader_dict(train_loader)
        return train_loader, val_loader, test_loader, class_loader_dict
    
    def get_class_loader_dict(self, train_loader):
        class_loader_dict = {}
        if isinstance(train_loader.dataset, torch.utils.data.Subset):
            full_target = train_loader.dataset.dataset.targets
            target = full_target[train_loader.dataset.indices]
            ds = train_loader.dataset.dataset
        else:
            target = train_loader.dataset.targets
            ds = train_loader.dataset
        for cls_id in range(self.args.num_classes):
            cls_indices = np.where(np.isin(target, cls_id))[0]
            cls_sampler = torch.utils.data.SubsetRandomSampler(cls_indices)
            class_loader_dict[cls_id] = torch.utils.data.DataLoader(ds, 
                                                                    batch_size=self.args.batch_size, 
                                                                    sampler=cls_sampler)
        return class_loader_dict
    
    @staticmethod
    def get_2nd_score(model, x, y):
        indices = torch.topk(model(x), k=2, dim=1).indices
        top1_matches = indices[:, 0] == y
        selected_labels = torch.where(top1_matches, indices[:, 1], indices[:, 0])
        return selected_labels

    def train(self, unlearn_loader, train_loader, test_loader, proj_mat, pseudo_label=False):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.un_lr)
        
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        remain_class = list(set(list(range(self.args.num_classes))) -set([self.args.unlearn_class]))
        for ep in range(self.args.un_epochs):
            for batch, (x, y) in enumerate(unlearn_loader):
                x = x.cuda()
                if pseudo_label:
                    y = self.get_2nd_score(self.original_model, x, y)
                else:
                    y = torch.from_numpy(np.random.choice(remain_class, size=x.shape[0])).cuda()
                pred_y = self.model(x)
                loss = self.criterion(pred_y, y)
                optimizer.zero_grad()
                loss.backward()
                kk = 0 
                for k, (m,params) in enumerate(self.model.named_parameters()):
                    if len(params.size())!=1:
                        sz =  params.grad.data.size(0)
                        params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                proj_mat[kk].cuda()).view(params.size())
                        kk +=1
                    elif 'feature' in m and len(params.size())!=1:
                        params.grad.data.fill_(0)
                optimizer.step()
            self.logger.info('[train] epoch {}, batch {}, loss {}'.format(ep, batch, loss))
            self.logger.info('>> unlearned model testing acc by class SGD <<')
            self.test_by_class(self.model, test_loader)
            self.test_by_class(self.model, train_loader)
    
    @staticmethod
    def get_representation_matrix(net, x, batch_list=[24, 100, 100,125, 125, 125]): 
        net.eval()
        with torch.no_grad():
            x = x.cuda()
            _ = net(x)
        mat_list=[]
        for i, (layer_name, in_act_map) in enumerate(net.in_act.items()):
            bsz=batch_list[i]
            k=0
            act = in_act_map.detach().cpu().numpy()
            if 'feature' in layer_name:
                ksz= net.ksize[layer_name]
                s=compute_conv_output_size(net.in_map_size[layer_name],ksz)
                mat = np.zeros((ksz*ksz*net.in_channel[layer_name],s*s*bsz))
                for kk in range(bsz):
                    for ii in range(s):
                        for jj in range(s):
                            mat[:,k]=act[kk,:,ii:ksz+ii,jj:ksz+jj].reshape(-1) 
                            k +=1
                mat_list.append(mat)
            else:
                activation = act[0:bsz].transpose()
                mat_list.append(activation)
        return mat_list  

    def update_subspaces(self, mat_list, threshold, feature_list=[]):
        self.logger.info ('Threshold: ', threshold) 
        if not feature_list:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U,S,Vh = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio)<threshold) #+1  
                feature_list.append(U[:,0:r])
        else:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1**2).sum()
                act_hat = activation - np.dot(np.dot(feature_list[i],feature_list[i].transpose()),activation)
                U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                sval_hat = (S**2).sum()
                sval_ratio = (S**2)/sval_total               
                accumulated_sval = (sval_total-sval_hat)/sval_total
                r = 0
                for ii in range (sval_ratio.shape[0]):
                    if accumulated_sval < threshold:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    self.logger.info('Skip Updating GPM for layer: {}'.format(i+1)) 
                    continue
                Ui=np.hstack((feature_list[i],U[:,0:r]))  
                if Ui.shape[1] > Ui.shape[0] :
                    feature_list[i]=Ui[:,0:Ui.shape[0]]
                else:
                    feature_list[i]=Ui
        
        self.logger.info('-'*40)
        for i in range(len(feature_list)):
            self.logger.info ('Layer {} : {}/{}'.format(i+1,feature_list[i].shape[1], feature_list[i].shape[0]))
        self.logger.info('-'*40)
        return feature_list  