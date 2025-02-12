import numpy as np
import time
import cv2
import sys
import os
import pickle
import torch
import torch.nn as nn
import SimpleITK as sitk
from torch import optim
from torch.utils.data import DataLoader, Subset, Dataset
from PIL import Image
import random
from albumentations import (
    GridDistortion,
    ElasticTransform,
    Compose,
    HorizontalFlip,
    OneOf, 
    ShiftScaleRotate
)
import matplotlib.pyplot as plt


class PARegressDataset(Dataset):
    """ 
    Dataset class for loading training and testing data.  
    """
    def __init__(self, data_path, random_seed=0, mode='train', test_group=14, is_aug=True):
        self.data_path = data_path
        self.random_seed = random_seed
        self.mode = mode
        self.test_group = test_group
        self.is_aug = is_aug
        self.parse_data()
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.data_point_train)
        elif self.mode == 'eval':
            return len(self.data_point_train)
        elif self.mode == 'test':
            return len(self.data_point_test)
        else: 
            return 0
    
    def __getitem__(self, idx):
        ''' get item based on index
        data_point = [PA_j, sO2_gt]'''
        if self.mode == 'train':
            # Equivalent chance of feeding in pos/neg sample
            is_pos = random.randint(0, 1)
            if is_pos:
                temp_idx = int(np.floor(float(idx) / float(len(self.data_point_train)) * float(len(self.train_positives))))
                img = self.data_point_all[self.train_positives[temp_idx]][0].astype('float32')
                sO2_gt = self.data_point_all[self.train_positives[temp_idx]][1].astype('float32')
            else:
                temp_idx = int(np.floor(float(idx) / float(len(self.data_point_train)) * float(len(self.train_negatives))))
                img = self.data_point_all[self.train_negatives[temp_idx]][0].astype('float32')
                sO2_gt = self.data_point_all[self.train_negatives[temp_idx]][1].astype('float32')
        elif self.mode == 'eval':
            img = self.data_point_train[idx][0].astype('float32')
            sO2_gt = self.data_point_train[idx][1].astype('float32')
        elif self.mode == 'test':
            img = self.data_point_test[idx][0].astype('float32')
            sO2_gt = self.data_point_test[idx][1].astype('float32')
        else:
            raise Exception('dataset mode should be either \'train\' or \'test\'')
        
        # Preprocessing
        img = img/np.max(img)
        sO2_gt = np.expand_dims(sO2_gt, 0)
        # npad = [(0, 0), (0, 0), (0, 1)]
        # img = np.pad(img, pad_width=npad, mode='constant', constant_values=0)
        
        # Augmentation
        if self.is_aug:
            self.aug = Compose([
                        HorizontalFlip(p=0.5),
                        ShiftScaleRotate(p=1, scale_limit=[-0.1,0.3], shift_limit_x=0.2, shift_limit_y=[0, 0.2], rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, value=0.02)
                    ], p=1)
            augmented = self.aug(image=np.moveaxis(img, 0, -1))
            img = np.moveaxis(augmented['image'], -1, 0)

        # Create sample  
        img = torch.from_numpy(img).float()        
        sO2_gt = torch.from_numpy(sO2_gt).float()
        sample = {'img': img, 'sO2_gt': sO2_gt}
        return sample
    
    def parse_data(self):
        # default: "../dataset/data_point_ius.pk"
        with open(self.data_path, "rb") as f:
            self.data_point_all = pickle.load(f)
        piglet_split = [[0, 10], [10, 20], [20, 32], [32, 35], [35, 40], [40, 46], \
            [46, 55], [55, 66], [66, 77], [77, 86], \
                [86, 94], [94, 102], [102, 113], [113, 119], [119, 129]]
        
        # Initialize random seed
        random.seed(self.random_seed)

        # Choose test group
        self.train_index = []
        self.test_index = []
        for i in range(len(piglet_split)):
            if i == self.test_group:
                self.test_index.append(piglet_split[i])
            else:
                self.train_index.append(piglet_split[i])
        
        self.positives = []
        self.negatives = []
        self.data_point_train = []
        self.data_point_test = []
        self.train_positives = []
        self.train_negatives = []

        # Find out training data, positive vs negative
        for i in range(len(self.train_index)):
            for j in range(self.train_index[i][0], self.train_index[i][1]):
                if self.data_point_all[j][1] >= 30:
                    self.negatives.append(j)
                elif self.data_point_all[j][1] < 30:
                    self.positives.append(j)
        self.train_positives = self.positives
        self.train_negatives = self.negatives
        
        # Create train/test data
        for i in range(len(self.train_index)):
            for j in range(self.train_index[i][0], self.train_index[i][1]):
                self.data_point_train.append(self.data_point_all[j])
        for i in range(len(self.test_index)):
            for j in range(self.test_index[i][0], self.test_index[i][1]):
                self.data_point_test.append(self.data_point_all[j])


# Define deep neural network
class PARegressNet(nn.Module):
    def __init__(self):
        super(PARegressNet, self).__init__()
        self.nn1 = nn.Sequential(
            nn.Conv2d(21, 4, 3),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(4, 8, 3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(8, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.nn2 = nn.Sequential(
            nn.Linear(1536, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        temp = self.nn1(x)
        temp = temp.view(temp.size(0), -1)
        # output = self.sigm(self.nn2(temp))
        output = self.nn2(temp)
        return output

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

if __name__ == '__main__':
    for test_group_idx in range(0, 8):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if not os.path.exists("./result_"+timestr):
            os.makedirs("./result_"+timestr)

        # training setup
        epochs = 10000
        use_cuda = True
        loss_fn = nn.MSELoss()
        net = PARegressNet()
        if use_cuda:
            net = net.cuda()
        net.apply(init_weights)
        lr = 1e-6
        optimizer = optim.Adam(net.parameters(), lr=lr)
        batch_size = 8

        # analysis rec
        counter = []
        loss_history = []
        loss_history_test = []
        is_eval = True
        rec_step = 5

        # load data
        train_set = PARegressDataset(data_path="../dataset/data_point_ius_pa_regress.pk", random_seed=0, mode='train', test_group=test_group_idx)
        test_set = PARegressDataset(data_path="../dataset/data_point_ius_pa_regress.pk", random_seed=0, mode='test', test_group=test_group_idx, is_aug=False)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

        iteration=0

        for epoch in range(epochs):
            for batch_idx, batch_sample in enumerate(train_loader):
                img = batch_sample['img']
                y = batch_sample['sO2_gt']
                if use_cuda:
                    img = img.to(device=torch.device("cuda"), dtype=torch.float)
                    y = y.to(device=torch.device("cuda"), dtype=torch.float)
                
                net.train()
                y_pred = net(img)
                loss = loss_fn(y_pred, y) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_idx % int(len(train_set)/batch_size/10) == 0:
                    print("Epoch %d, Batch %d Loss %f" % (epoch, batch_idx, loss.item()))
                    iteration += 10
                    counter.append(iteration)
                    loss_history.append(loss.item())
                    if is_eval:
                        with torch.no_grad():
                            loss_current_total = 0
                            net.eval()
                            for batch_idx_test, batch_sample_test in enumerate(test_loader):
                                test_img = batch_sample_test['img'].type(torch.FloatTensor)
                                test_y = batch_sample_test['sO2_gt'].type(torch.FloatTensor)
                                if use_cuda:
                                    test_img = test_img.cuda()
                                    test_y = test_y.cuda()
                                test_y_pred = net(test_img)
                                test_loss = loss_fn(test_y_pred, test_y) 
                                loss_current_total = loss_current_total + test_loss.item()
                                # eval(test_y_pred, test_y)
                            loss_current_total = loss_current_total / len(test_set)
                            loss_history_test.append(loss_current_total)
                            print("Epoch %d, Batch %d Loss on test: %f" % (epoch, batch_idx, loss_current_total))

                    total_hist = [counter, loss_history, loss_history_test]
            if epoch % rec_step == 0:
                with open("./result_"+timestr+"/training_hist_"+timestr+".txt", "wb") as fp:
                    pickle.dump(total_hist, fp)
                torch.save(net.state_dict(), "./result_"+timestr+"/net_weights_"+timestr+".pt")