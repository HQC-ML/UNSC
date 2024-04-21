import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm
import torchvision
from torchvision import transforms
from copy import deepcopy
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import torchvision
from sklearn.svm import SVC

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / self.count

cifar100_mapping = {
    "aquatic mammal": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flower": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food container": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit and vegetable": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
    "household electrical device": ["clock", "keyboard", "lamp", "telephone", "television"],
    "household furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insect": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large carnivore": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large man-made outdoor thing": ["bridge", "castle", "house", "road", "skyscraper"],
    "large natural outdoor scene": ["cloud", "forest", "mountain", "plain", "sea"],
    "large omnivore and herbivore": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium-sized mammal": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non-insect invertebrate": ["crab", "lobster", "snail", "spider", "worm"],
    "person": ["baby", "boy", "girl", "man", "woman"],
    "reptile": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small mammal": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "tree": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    "vehicle 1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicle 2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]
}


def compute_maximum_length(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))


def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

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

def update_GPM(mat_list, threshold, feature_list=[]):
    print ('Threshold: ', threshold) 
    if not feature_list:
        # After First Task 
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
                print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                continue
            # update GPM
            Ui=np.hstack((feature_list[i],U[:,0:r]))  
            if Ui.shape[1] > Ui.shape[0] :
                feature_list[i]=Ui[:,0:Ui.shape[0]]
            else:
                feature_list[i]=Ui
    
    print('-'*40)
    print('Gradient Constraints Summary')
    print('-'*40)
    for i in range(len(feature_list)):
        print ('Layer {} : {}/{}'.format(i+1,feature_list[i].shape[1], feature_list[i].shape[0]))
    print('-'*40)
    return feature_list  

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

def test_by_class(model, test_loader, i=3):
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
        # calculate accuracy for each class
        for c in torch.unique(y_list):
            class_indices = torch.where(y_list == c)[0]
            correct = torch.sum(pred_y_list[class_indices] == y_list[class_indices]).item()
            total = len(class_indices)
            acc = correct / total
            acc_lst.append(acc)
            try: 
                if c in i:
                    print(f'[{acc:.4f}], ', end='')
                else:
                    print(f'{acc:.4f}, ', end='')
            except:
                print(f'{acc:.4f}, ', end='')
        

        acc_lst = np.array(acc_lst)
        mask = np.ones(len(acc_lst), dtype=bool)
        mask[i] = False
        unlearn_class = acc_lst[i]
        remain_class = acc_lst[mask]
        print(f'Acc_f: {unlearn_class.mean():.4f}, Acc_r: {remain_class.mean():.4f}')

    return unlearn_class.mean(), remain_class.mean()



def plot_umap(umap_f, model, dataloader, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42, title='t-sne', alpha=0.5, skip_class=6):
    model.eval()
    data = []
    labels = []
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x = x.cuda()
            y = y.cuda()
            data.append(model(x).cpu().numpy())
            labels.append(y.cpu().numpy())
        data = np.vstack(data)
        labels = np.hstack(labels)

    if umap_f is None:
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state)
        data = reducer.fit_transform(data)
    else:
        data = umap_f.transform(data)

    cmp = plt.get_cmap("tab10")
    if skip_class is None:
        plt.figure(figsize=(6, 6), dpi=300)
        for i in np.unique(labels):
            plt.scatter(data[labels == i, 0], data[labels == i, 1], label=str(i), alpha=alpha, color = cmp.colors[i])
        plt.legend(loc='upper right')
        plt.show()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        for i in np.unique(labels):
            if i == skip_class:
                ax1.scatter(data[labels == i, 0], data[labels == i, 1], label=str(i), alpha=alpha, color = cmp.colors[i], zorder=1)
            else:
                ax1.scatter(data[labels == i, 0], data[labels == i, 1], label=str(i), alpha=alpha, color = cmp.colors[i], zorder=0)
                ax2.scatter(data[labels == i, 0], data[labels == i, 1], label=str(i), alpha=alpha, color = cmp.colors[i])
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')  
        plt.show()

    return reducer if umap_f is None else None

def merge_subspace(per_class_basis_dict, unlearn_class):
    new_basis = []
    proj_mat = []
    for layer in range(len(per_class_basis_dict[0])):
        concat_layer_basis = []
        for cls_id in per_class_basis_dict.keys():
            basis_list = per_class_basis_dict[cls_id]
            if cls_id in unlearn_class:
                continue
            else:
                concat_layer_basis.append(basis_list[layer])
        U,S,Vh=np.linalg.svd(np.concatenate(concat_layer_basis, axis=1), full_matrices=False)
        sval_total = (S**2).sum()
        sval_ratio = (S**2)/sval_total
        r = np.sum(np.cumsum(sval_ratio)<0.99) #+1  
        new_layer_basis = U[:, :r]
        new_basis.append(new_layer_basis)
        proj_mat.append(torch.Tensor(np.dot(new_layer_basis, new_layer_basis.transpose())).cuda())
    
    return new_basis, proj_mat


def get_model(args):
    if args.model_name.lower() == 'alexnet':
        from models.alexnet import AlexNet
        model = AlexNet(n_channels=args.n_channels, num_classes=args.num_classes).cuda()

    elif args.model_name.lower() == 'resnet18':
        from models.resnet import resnet18
        model = resnet18(num_classes=args.num_classes).cuda()

    elif args.model_name.lower() == 'resnet20':
        from models.resnet import resnet20
        model = resnet20( num_classes=args.num_classes).cuda()

    elif args.model_name.lower() == 'resnet50':
        from models.resnet import resnet50
        model = resnet50(num_classes=args.num_classes).cuda()

    elif args.model_name.lower() == 'vgg11':
        from models.vgg import vgg11_bn
        model = vgg11_bn(num_classes=args.num_classes).cuda()
    
    elif args.model_name.lower() == 'allcnn':
        from models.allcnn import AllCNN
        model = AllCNN(n_channels=args.n_channels, num_classes=args.num_classes, filters_percentage=0.5).cuda()
    else:
        raise NotImplementedError
    return model


def train(args, model, train_loader, val_loader, optimizer, scheduler, criterion):
    best_acc = 0
    for ep in range(args.num_epochs):
        model.train()
        for batch, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            pred_y = model(x)
            loss = criterion(pred_y, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_acc = test(model, val_loader)
        train_acc = test(model, train_loader)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = deepcopy(model)
            p = args.patience
        else:
            p -= 1
            if p == 0:
                break
        print('[train] epoch {}, train_acc {:.3f}, val_acc {:.3f}'.format(ep, train_acc, val_acc), flush=True)
    model.load_state_dict(best_model.state_dict())


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
    

def get_unlearn_dataloader(args, data_loader):
    dataset = data_loader.dataset
    _indices = data_loader.sampler.indices

    if args.dataset.lower() == 'svhn':
        train_targets = np.array(dataset.labels)[_indices]
    else:
        train_targets = np.array(dataset.targets)[_indices]
    unlearn_indices, remain_indices = [], []
    for i, target in enumerate(train_targets):
        if target in args.unlearn_class:
            unlearn_indices.append(i)
        else:
            remain_indices.append(i)

    unlearn_indices = np.array(_indices)[unlearn_indices]
    remain_indices = np.array(_indices)[remain_indices]

    unlearn_sampler = torch.utils.data.SubsetRandomSampler(unlearn_indices)
    unlearn_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=args.batch_size,
                                                sampler = unlearn_sampler,)

    remain_sampler = torch.utils.data.SubsetRandomSampler(remain_indices)
    remain_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=args.batch_size,
                                                sampler = remain_sampler)
    return remain_loader, unlearn_loader


def _get_dataloader(args):
    if args.dataset.lower() == 'fmnist':
        tsfm_train = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.286], std=[0.353]),
        ])

        tsfm_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.286], std=[0.353]),
        ])

        train_set = torchvision.datasets.FashionMNIST(root='../data/fashionmnist', 
                                                    train=True, 
                                                    download=True, 
                                                    transform=tsfm_train)
        test_set = torchvision.datasets.FashionMNIST(root='../data/fashionmnist', 
                                                    train=False, 
                                                    download=True, 
                                                    transform=tsfm_test)

    elif args.dataset.lower() == 'cifar10':
        tsfm_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        tsfm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_set = torchvision.datasets.CIFAR10(root='../data/cifar10',
                                         train=True, 
                                         download=True, 
                                         transform=tsfm_train)

        test_set = torchvision.datasets.CIFAR10(root='../data/cifar10', 
                                                train=False, 
                                                download=True, 
                                                transform=tsfm_test)
    elif args.dataset.lower() == 'cifar100':
        tsfm_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
        ])

        tsfm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
        ])
        train_set = torchvision.datasets.CIFAR100(root='../data/cifar100',
                                         train=True, 
                                         download=True, 
                                         transform=tsfm_train)

        test_set = torchvision.datasets.CIFAR100(root='../data/cifar100', 
                                                train=False, 
                                                download=True, 
                                                transform=tsfm_test)

    elif args.dataset.lower() == 'svhn':
        tsfm_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.437, 0.443, 0.472], std=[0.198, 0.201, 0.197]),
        ])

        tsfm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.437, 0.443, 0.472], std=[0.198, 0.201, 0.197]),
        ])
        train_set = torchvision.datasets.SVHN(root='../data/svhn',
                                         split='train', 
                                         download=True, 
                                         transform=tsfm_train)

        test_set = torchvision.datasets.SVHN(root='../data/svhn', 
                                                split='test', 
                                                download=True, 
                                                transform=tsfm_test)
    else:
        raise NotImplementedError
    
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size=args.batch_size, 
                                               num_workers=4,
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(test_set, 
                                              batch_size=args.batch_size, 
                                              num_workers=4,
                                              shuffle=False)
    return train_loader, test_loader

def get_dataloader(args):
    train_loader, test_loader = _get_dataloader(args)

    indices = np.arange(len(train_loader.dataset))
    a = np.split(indices,[int(len(indices)*0.9), int(len(indices))])
    idx_train = a[0]
    idx_val = a[1]
    train_sampler = torch.utils.data.SubsetRandomSampler(idx_train)
    val_sampler = torch.utils.data.SubsetRandomSampler(idx_val)

    train_loader = torch.utils.data.DataLoader(train_loader.dataset,
                                                batch_size=args.batch_size,
                                                sampler=train_sampler)
    
    val_loader = torch.utils.data.DataLoader(train_loader.dataset,
                                                batch_size=args.batch_size,
                                                sampler=val_sampler)
    
    test_loader = torch.utils.data.DataLoader(test_loader.dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False)
    
    if args.retrain:
        train_loader, _ = get_unlearn_dataloader(args, train_loader)
        val_loader, _ = get_unlearn_dataloader(args, val_loader)
        
    return train_loader, val_loader, test_loader

@torch.no_grad()
def get_mia_dataset(model, dataloader, criterion=torch.nn.CrossEntropyLoss(reduction='none'), device='cuda'):
    Loss = []
    Prob = []
    model.eval()
    model = model.to(device)
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=-1)
        loss = criterion(logits, y)
        Prob.append(prob.cpu())
        Loss.extend(loss.cpu().tolist())  # Append the loss values to the Loss list

    Prob = torch.cat(Prob, dim=0)
    Loss = torch.tensor(Loss)
    return Prob, Loss


def split_2_remain_unlearn(args, dataloader):
    """
    dataloader: the dataloader of the whole dataset, is produced by specifying sampler
    """
    if isinstance(dataloader.sampler, torch.utils.data.SubsetRandomSampler):
        original_indices = dataloader.sampler.indices
    elif isinstance(dataloader.sampler, torch.utils.data.SequentialSampler):
        original_indices = np.arange(len(dataloader.dataset))
    try:
        target_list = np.array(dataloader.dataset.targets)[original_indices]
    except:
        target_list = np.array(dataloader.dataset.labels)[original_indices]

    unlearn_indices = np.where(np.isin(target_list, args.unlearn_class))[0]
    remain_indices = np.where(~np.isin(target_list, args.unlearn_class))[0]

    # conver to the original indices
    unlearn_indices = original_indices[unlearn_indices]
    remain_indices = original_indices[remain_indices]

    unlearn_sampler = torch.utils.data.SubsetRandomSampler(unlearn_indices)
    unlearn_loader = torch.utils.data.DataLoader(dataloader.dataset,
                                                batch_size=args.batch_size,
                                                sampler = unlearn_sampler,)
    remain_sampler = torch.utils.data.SubsetRandomSampler(remain_indices)
    remain_loader = torch.utils.data.DataLoader(dataloader.dataset,
                                                batch_size=args.batch_size,
                                                sampler = remain_sampler)
    return remain_loader, unlearn_loader
 

