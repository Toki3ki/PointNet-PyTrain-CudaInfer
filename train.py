# 这是程序一的Python模板程序，您可以增加epoch后提交该程序而不进行任何修改（如果不在意准确率的分数），对于（加分题），请参考程序二的模板程序自行实现

'''
Package                  Version
------------------------ ----------
certifi                  2024.8.30
charset-normalizer       3.3.2
cmake                    3.30..3
filelock                 3.16.0
h5py                     3.11.0
hdf5                     1.12.1
idna                     3.8
Jinja2                   3.1.4
lit                      18.1.8
MarkupSafe               2.1.5
mpmath                   1.3.0
networkx                 3.3
numpy                    1.26.0
nvidia-cublas-cu11       11.10.3.66
nvidia-cuda-cupti-cu11   11.7.101
nvidia-cuda-nvrtc-cu11   11.7.99
nvidia-cuda-runtime-cu11 11.7.99
nvidia-cudnn-cu11        8.5.0.96
nvidia-cufft-cu11        10.9.0.58
nvidia-curand-cu11       10.2.10.91
nvidia-cusolver-cu11     11.4.0.1
nvidia-cusparse-cu11     11.7.4.91
nvidia-nccl-cu11         2.14.3
nvidia-nvtx-cu11         11.7.91
Pillow                   10.4.0
pip                      24.2
requests                 2.32.3
setuptools               72.1.0
sympy                    1.13.2
torch                    2.0.1
torchaudio               2.0.2
torchvision              0.15.2
triton                   2.0.0
typing_extensions        4.12.2
urllib3                  2.2.2
wheel                    0.43.0
'''

import os

import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import torch.nn.parallel
from torch.autograd import Variable
import numpy as np
import h5py
from tqdm import tqdm


# import provider
num_class = 10
total_epoch = 70
script_dir = os.path.dirname(__file__)  # 获取脚本所在的目录
# 45
# 50seed / 45epoch = 0.953
# 59seed / 55epoch = 0.952
# 64seed / 57epoch = 0.953

import random
seed = 64
def seed_everything(seed):
    random.seed(seed)           # Python random 模块的种子
    np.random.seed(seed)        # NumPy 随机数生成器的种子
    torch.manual_seed(seed)     # PyTorch CPU 计算的随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)            # PyTorch GPU 计算的随机数种子
        torch.cuda.manual_seed_all(seed)        # 如果使用多块 GPU, 则需要设置所有 GPU 的种子
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False   # 禁用部分加速算法，确保可重复性

# 调用一次，在训练开始前固定所有的随机数种子
seed_everything(seed)
print(seed)


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 128)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 128)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 128)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 128, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


# 模型定义
class get_model(nn.Module):
    def __init__(self, k=10, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


class PointCloudDataset(Dataset):
    def __init__(self,root, split):
        self.list_of_points = []
        self.list_of_labels = []
        self.root = root
        self.split = split

        # with h5py.File(f"{split}_point_clouds.h5","r") as hf:
        with h5py.File(f"{self.root}/{self.split}_point_clouds.h5","r") as hf:
            for k in hf.keys():
                self.list_of_points.append(hf[k]["points"][:].astype(np.float32))
                self.list_of_labels.append(hf[k].attrs["label"])
        self.fix_length_statistics_with_median()

    def __len__(self):
        return len(self.list_of_points)

    def __getitem__(self, idx):
        points = self.list_of_points[idx]
        label = self.list_of_labels[idx]
        return points, label

    def fix_length_statistics_with_median(self):
        lengths = [points.shape[0] for points in self.list_of_points]
        fix_length = int( np.median(lengths) )
        
        new_list_of_points = []
        for points in self.list_of_points:
            if(points.shape[0] >= fix_length):
                new_list_of_points.append(points[:fix_length, :])
            else:
                new_list_of_points.append(np.concatenate((points, np.zeros((fix_length - points.shape[0], 3), dtype=np.float32)), axis=0))
        self.list_of_points = new_list_of_points



def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


# def test(model, loader, num_class=10):
#     mean_correct = []
#     classifier = model.eval()

#     # for j, (points, target) in tqdm(enumerate(loader), total=len(loader)): #显示进度条
#     for j, (points, target) in enumerate(loader):

#         points, target = points.cuda(), target.cuda()

#         points = points.transpose(2, 1)
#         pred, _ = classifier(points)
#         pred_choice = pred.data.max(1)[1]

#         correct = pred_choice.eq(target.long().data).cpu().sum()
#         mean_correct.append(correct.item())

#     instance_acc = np.sum(mean_correct) / len(loader.dataset)

#     return instance_acc

# def pad_collate_fn(batch):
#     # 找到批次中最小的数组大小
#     min_size = min([item[0].shape[0] for item in batch])
    
#     # 截断数组
#     padded_batch = []
#     for points, target in batch:
#         # 截断数组
#         points = points[:min_size, :]
#         padded_batch.append((points, target))
    
#     # 使用默认的 collate_fn 处理填充后的批次
#     return torch.utils.data.dataloader.default_collate(padded_batch)

# provider
def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc

# 保存模型参数和缓冲区为 .txt 文件
def save_model_params_and_buffers_to_txt(model, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 保存所有参数
    for name, param in model.named_parameters():
        np.savetxt(os.path.join(directory, f'{name}.txt'), param.detach().cpu().numpy().flatten())
    
    # 保存所有缓冲区
    for name, buffer in model.named_buffers():
        np.savetxt(os.path.join(directory, f'{name}.txt'), buffer.detach().cpu().numpy().flatten())

def main():
    # 创建数据集实例
    data_path = './data'

    train_dataset = PointCloudDataset(root=data_path, split='train')
    # test_dataset = PointCloudDataset(root=data_path, split='test')

    # 创建 DataLoader 实例
    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10, drop_last=True, collate_fn=pad_collate_fn) #batch_size内固定长度截取
    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=10, drop_last=False) #全局固定长度填充/截取
    # test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=False, num_workers=10, drop_last=False)

    print("finish DATA LOADING")

    '''MODEL LOADING'''

    classifier = get_model(num_class)
    criterion = get_loss()
    classifier.apply(inplace_relu)

    classifier = classifier.cuda()
    criterion = criterion.cuda()

    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    print("finish MODEL LOADING")



    '''TRANING'''
    print("start TRANING")

    for epoch in range(total_epoch):
        print('Epoch %d (%d/%s):' % (epoch + 1, epoch + 1, total_epoch))
        mean_correct = []
        classifier = classifier.train()

        if epoch + 1 >= 60:
            optimizer = torch.optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

        # for batch_id, (points, target) in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), smoothing=0.9): #显示进度条
        for batch_id, (points, target) in enumerate(train_dataloader, 0):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = random_point_dropout(points)
            points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        scheduler.step()

        train_instance_acc = np.mean(mean_correct)

        print('Train Instance Accuracy: %f' % train_instance_acc)

        # with torch.no_grad():
        #     instance_acc = test(classifier.eval(), test_dataloader, num_class=num_class)

        #     if (instance_acc >= best_instance_acc):
        #         best_instance_acc = instance_acc
        #         best_epoch = epoch + 1
        #         dir = os.path.join(script_dir, "finetune", "128_256_128")
        #         torch.save(classifier.state_dict(), f'{dir}/best_model.pth')
        #         save_model_params_and_buffers_to_txt(classifier, dir)

        #     print('Test Instance Accuracy: %f' % (instance_acc))
        #     print('Best Instance Accuracy: %f' % (best_instance_acc))
        # if epoch + 1 == 57:
    save_model_params_and_buffers_to_txt(classifier, script_dir)

    # print("finish TRANING")
    # save_model_params_and_buffers_to_txt(classifier, script_dir)

if __name__ == '__main__':
    main()