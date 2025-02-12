import numpy as np
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
    ShiftScaleRotate,
    Cutout,
    MultiplicativeNoise,
    ChannelDropout,
    CoarseDropout
)
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter



class RegressorDatasetPatch(Dataset):
    """ 
    Dataset class for loading training and testing data.  
    """
    def __init__(self, data_path, random_seed=0, mode='train', test_group=10, is_aug=True, inputSelect='FullPA'):
        self.data_path = data_path
        self.random_seed = random_seed
        self.mode = mode
        self.test_group = test_group
        self.is_aug = is_aug
        self.all_waves = np.arange(21)
        self.inputSelect = inputSelect
        self.parse_data()
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.data_point_train)
        elif self.mode == 'test':
            return len(self.data_point_test)
        else: 
            return 0
    
    def __getitem__(self, idx):
        ''' get item based on index
        data_point = [PA_j, sss_x, sss_y, ischemia_label, Hypoxia_FiO2_label, Hypoxia_SSS_30P_label, condition, sample_num]'''
        if self.inputSelect == 'FullPA':
            if self.mode == 'train':
                is_pos = random.randint(0, 1)
                if is_pos:
                    temp_idx = int(np.floor(float(idx) / float(len(self.data_point_train)) * float(len(self.train_positives))))
                    img = self.data_point_all[self.train_positives[temp_idx]][0].astype('float32')
                    sss_x = self.data_point_all[self.train_positives[temp_idx]][1]
                    sss_y = self.data_point_all[self.train_positives[temp_idx]][2]
                    sO2_map = self.data_point_all[self.train_positives[temp_idx]][4].astype('float32')
                    sO2_gt = self.data_point_all[self.train_positives[temp_idx]][5].astype('float32')
                else:
                    temp_idx = int(np.floor(float(idx) / float(len(self.data_point_train)) * float(len(self.train_negatives))))
                    img = self.data_point_all[self.train_negatives[temp_idx]][0].astype('float32')
                    sss_x = self.data_point_all[self.train_positives[temp_idx]][1]
                    sss_y = self.data_point_all[self.train_positives[temp_idx]][2]
                    sO2_map = self.data_point_all[self.train_negatives[temp_idx]][4].astype('float32')
                    sO2_gt = self.data_point_all[self.train_negatives[temp_idx]][5].astype('float32')
            elif self.mode == 'test':
                img = self.data_point_test[idx][0].astype('float32')
                sss_x = self.data_point_test[idx][1]
                sss_y = self.data_point_test[idx][2]
                sO2_map = self.data_point_test[idx][4].astype('float32')
                sO2_gt = self.data_point_test[idx][5].astype('float32')
            else:
                raise Exception('dataset mode should be either \'train\' or \'test\'')
        
        # Preprocessing
        patch_pa = img[:,sss_x-10:sss_x+10, sss_y-10:sss_y+10]
        patch_so2 = sO2_map[sss_x-10:sss_x+10, sss_y-10:sss_y+10]
        patch_pa = patch_pa/np.max(patch_pa)
        patch_so2 = np.expand_dims(patch_so2,0)
        imgout = np.concatenate((patch_pa, patch_so2),axis=0)

        # Data Augmentation
        if self.is_aug:
            # # Contrast 
            # mean_vals = (1 + random.random())*np.mean(img)
            # high = mean_vals
            # img[img>high]=1.
            # img = img / high
            # img[img>1.]=1.

            # Geometric augmentation
            self.aug = Compose([
                HorizontalFlip(p=0.5),
                GridDistortion(distort_limit=0.1, p=0.5),
                ShiftScaleRotate(p=1, scale_limit=[-0.2,0.3], shift_limit_x=0.3, shift_limit_y=[0, 0.3], rotate_limit=20, border_mode=cv2.BORDER_WRAP)
            ], p=0.9)
            augmented = self.aug(image=np.moveaxis(imgout, 0, -1))
            imgout = np.moveaxis(augmented['image'], -1, 0)
        else:
            pass
            # # Enhance contrast with fixed value
            # mean_vals = 1.5*np.mean(img)
            # high = mean_vals
            # img[img>high]=1.
            # img = img / high
            # img[img>1.]=1.

        # Create sample  
        imgout = torch.from_numpy(imgout).float()        
        # sss_gt = torch.from_numpy(sss_gt).float()
        sO2_gt = np.expand_dims(sO2_gt, 0)
        sO2_gt = torch.from_numpy(sO2_gt).float()

        sample = {'img': imgout, 'sO2_gt': sO2_gt}
        return sample
    
    def parse_data(self):
        # default: "../dataset/data_point_ius.pk"
        with open(self.data_path, "rb") as f:
            self.data_point_all = pickle.load(f)
        piglet_split = [[0, 10], [10, 15], [15, 21], \
            [21, 30], [30, 41], [41, 50], \
                [50, 58], [58, 69], [69, 75], [75, 84]]
        
        # Initialize random seed
        if self.random_seed is not None:
            random.seed(self.random_seed)

        # Choose test group
        self.train_index = []
        self.test_index = []
        for i in range(len(piglet_split)):
            if i == self.test_group:
                self.test_index.append(piglet_split[i])
            else:
                self.train_index.append(piglet_split[i])
        self.data_point_train = []
        self.data_point_test = []

        self.positives = []
        self.negatives = []
        self.data_point_train = []
        self.data_point_test = []
        self.train_positives = []
        self.train_negatives = []

        # Find out training data, positive vs negative
        for i in range(len(self.train_index)):
            for j in range(self.train_index[i][0], self.train_index[i][1]):
                if self.data_point_all[j][5] >= 30:
                    self.negatives.append(j)
                elif self.data_point_all[j][5] < 30:
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
class PatchRegressNet(nn.Module):
    def __init__(self):
        super(PatchRegressNet, self).__init__()
        self.nn1 = nn.Sequential(
            nn.Conv2d(22, 8, 3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            #nn.Dropout(p=0.5),

            nn.Conv2d(8, 16, 3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            #nn.Dropout(p=0.5),

            nn.Conv2d(16, 32, 3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1),
            #nn.Dropout(p=0.5),
        )

        self.nn2 = nn.Sequential(
            nn.Linear(288, 32),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(16, 1),
        )
        # self.sigm = nn.Sigmoid()

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

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

if __name__ == '__main__':
    timestr = time.strftime("%Y%m%d-%H%M%S")
    results_root = "./results/exp_"+"patch_regressor_balanced"
    print("Training started, results folder: "+results_root)
    os.makedirs(results_root)
    for test_group_idx in range(0, 10):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(results_root+"/rec_weights_"+timestr):
            os.makedirs(results_root+"/rec_weights_"+timestr)

        # training setup
        epochs = 10000
        use_cuda = True
        loss_fn = nn.MSELoss()
        net = PatchRegressNet()
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
        rec_step = 50
        save_step = 500


        # load data
        train_set = RegressorDatasetPatch(data_path="./dataset/data_point_tensets_good.pk", random_seed=0, mode='train', test_group=test_group_idx, inputSelect='FullPA')
        test_set = RegressorDatasetPatch(data_path="./dataset/data_point_tensets_good.pk", random_seed=0, mode='test', test_group=test_group_idx, is_aug=False, inputSelect='FullPA')
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

            print("Currently at Group: %d, Epoch %d, Loss %f" % (test_group_idx, epoch, loss.item()))

            if epoch % rec_step == 0:
                iteration += rec_step
                counter.append(iteration)
                loss_history.append(loss.item())

                if is_eval:
                    with torch.no_grad():
                        loss_current_total = 0
                        net.eval()
                        # enable_dropout(net)
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
                        print("------ Group: %d, Epoch %d, Avg loss on test: %f" % (test_group_idx, epoch, loss_current_total))
                total_hist = [counter, loss_history, loss_history_test]

            if (epoch+1) % save_step == 0:
                current_dir = results_root+"/rec_weights_"+timestr
                for fname in os.listdir(current_dir):
                    if fname.endswith('.pt'):
                        filepath = current_dir + "/" + fname
                        os.remove(filepath)
                # only save the latest weight
                torch.save(net.state_dict(), results_root+"/rec_weights_"+timestr+"/net_weights_epoch_"+str(epoch)+".pt")
                with open(results_root+"/rec_weights_"+timestr+"/training_hist_"+timestr+".txt", "wb") as fp:
                    pickle.dump(total_hist, fp)