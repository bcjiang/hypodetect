#%%
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

#%%
class SSSLocDatasetTen(Dataset):
    """ 
    Dataset class for loading training and testing data.  
    """
    def __init__(self, data_path, random_seed=0, mode='train', test_group=10, is_aug=True, full_wave=False, waveselect='five'):
        self.data_path = data_path
        self.random_seed = random_seed
        self.mode = mode
        self.test_group = test_group
        self.full_wave = full_wave
        self.is_aug = is_aug
        self.all_waves = np.arange(21)
        self.waveselect = waveselect
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
        if self.full_wave == False:
            # [0,1,2,3],[4,5,6,7],[8,9,10,11,12],[13,14,15,16],[17,18,19,20]
            if self.waveselect=='five':
                wave_selected = np.array([random.randrange(4),
                                        random.randrange(4,8),
                                        random.randrange(8,13),
                                        random.randrange(13,17),
                                        random.randrange(17,21)])
                wave_removed = np.delete(self.all_waves, wave_selected)
            else:
                wave_removed = np.sort(np.random.choice(21,10,replace=False))
        
            if self.mode == 'train':
                img = self.data_point_train[idx][0].astype('float32') #[wave_selected,:,:]
                if random.random() < 0.5:
                    img[wave_removed,:,:] = 0 # Turn off all other wavelengths
                # img = np.expand_dims(img, 0) # Still 21-channel input
                sss_x = self.data_point_train[idx][1]
                sss_y = self.data_point_train[idx][2]
                sss_gt = self.data_point_train[idx][3].astype('float32')
                sO2_map = self.data_point_train[idx][4].astype('float32')
                sO2_gt = self.data_point_train[idx][5].astype('float32')
                label = self.data_point_train[idx][6].astype('float32')
            elif self.mode == 'test':
                img = self.data_point_test[idx][0].astype('float32') #[wave_selected,:,:]
                img[wave_removed,:,:] = 0 # Turn off all other wavelengths
                # img = np.expand_dims(img, 0) # Still 21-channel input
                sss_x = self.data_point_test[idx][1]
                sss_y = self.data_point_test[idx][2]
                sss_gt = self.data_point_test[idx][3].astype('float32')
                sO2_map = self.data_point_test[idx][4].astype('float32')
                sO2_gt = self.data_point_test[idx][5].astype('float32')
                label = self.data_point_test[idx][6].astype('float32')
            else:
                raise Exception('dataset mode should be either \'train\' or \'test\'')
        else:
            if self.mode == 'train':
                img = self.data_point_train[idx][0].astype('float32')
                sss_x = self.data_point_train[idx][1]
                sss_y = self.data_point_train[idx][2]
                sss_gt = self.data_point_train[idx][3].astype('float32')
                sO2_map = self.data_point_train[idx][4].astype('float32')
                sO2_gt = self.data_point_train[idx][5].astype('float32')
                label = self.data_point_train[idx][6].astype('float32')
            elif self.mode == 'test':
                img = self.data_point_test[idx][0].astype('float32')
                sss_x = self.data_point_test[idx][1]
                sss_y = self.data_point_test[idx][2]
                sss_gt = self.data_point_test[idx][3].astype('float32')
                sO2_map = self.data_point_test[idx][4].astype('float32')
                sO2_gt = self.data_point_test[idx][5].astype('float32')
                label = self.data_point_test[idx][6].astype('float32')
            else:
                raise Exception('dataset mode should be either \'train\' or \'test\'')
        
        # Preprocessing
        sss_gt = np.expand_dims(sss_gt, 0)
        npad = [(0, 0), (0, 0), (0, 1)]
        img = np.pad(img, pad_width=npad, mode='edge')
        sss_gt = np.pad(sss_gt, pad_width=npad, mode='edge')

        img = img/np.max(img)
        # img = (img - np.mean(img))/np.std(img)

        # Data Augmentation
        if self.is_aug:
            # Contrast 
            mean_vals = (1 + random.random())*np.mean(img)
            high = mean_vals
            img[img>high]=1.
            img = img / high
            img[img>1.]=1.

            # Geometric augmentation
            self.aug = Compose([
                HorizontalFlip(p=0.5),
                GridDistortion(distort_limit=0.1, p=0.5),
                ShiftScaleRotate(p=0.5, scale_limit=[-0.2,0.05], shift_limit_x=0.1, shift_limit_y=[0, 0.1], rotate_limit=15, border_mode=cv2.BORDER_WRAP)
            ], p=0.5)
            augmented = self.aug(image=np.moveaxis(img, 0, -1), mask=np.moveaxis(sss_gt, 0, -1))
            img = np.moveaxis(augmented['image'], -1, 0)
            sss_gt = np.moveaxis(augmented['mask'], -1, 0)
        else:
            # Enhance contrast with fixed value
            mean_vals = 1.5*np.mean(img)
            high = mean_vals
            img[img>high]=1.
            img = img / high
            img[img>1.]=1.

        # Create sample  
        img = torch.from_numpy(img).float()        
        sss_gt = torch.from_numpy(sss_gt).float()

        sample = {'img': img, 'sss_x': sss_x, 'sss_y': sss_y, 'sss_gt': sss_gt, 'sO2_map': sO2_map, 'sO2_gt': sO2_gt, 'label': label, 'random_check': random.random()}
        return sample
    
    def parse_data(self):
        # default: "../dataset/data_point_ius.pk"
        with open(self.data_path, "rb") as f:
            self.data_point_all = pickle.load(f)
        piglet_split = [[0, 10], [10, 15], [15, 21], \
            [21, 30], [30, 41], [41, 50], \
                [50, 58], [58, 69], [69, 75], [75, 84]]
        
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
        self.data_point_train = []
        self.data_point_test = []
        
        # Create train/test data
        for i in range(len(self.train_index)):
            for j in range(self.train_index[i][0], self.train_index[i][1]):
                self.data_point_train.append(self.data_point_all[j])
        for i in range(len(self.test_index)):
            for j in range(self.test_index[i][0], self.test_index[i][1]):
                self.data_point_test.append(self.data_point_all[j])

# %%
# U-Net model (with Monte Carlo dropout)

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def get_monte_carlo_predictions(data_loader,
                                forward_passes,
                                model,
                                n_classes,
                                n_samples):
    """ Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes
    https://stackoverflow.com/questions/63285197/measuring-uncertainty-using-mc-dropout-on-pytorch

    Parameters
    ----------
    data_loader : object
        data loader object from the data loader module
    forward_passes : int
        number of monte-carlo samples/forward passes
    model : object
        keras model
    n_classes : int
        number of classes in the dataset
    n_samples : int
        number of samples in the test set
    """

    dropout_predictions = np.empty((0, n_samples, n_classes))
    softmax = nn.Softmax(dim=1)
    for i in range(forward_passes):
        predictions = np.empty((0, n_classes))
        model.eval()
        enable_dropout(model)
        for i, (image, label) in enumerate(data_loader):
            image = image.to(torch.device('cuda'))
            with torch.no_grad():
                output = model(image)
                output = softmax(output)  # shape (n_samples, n_classes)
            predictions = np.vstack((predictions, output.cpu().numpy()))

        dropout_predictions = np.vstack((dropout_predictions,
                                         predictions[np.newaxis, :, :]))
        # dropout predictions - shape (forward_passes, n_samples, n_classes)

    # Calculating mean across multiple MCD forward passes 
    mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    # Calculating variance across multiple MCD forward passes 
    variance = np.var(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes 
    entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)  # shape (n_samples,)

    # Calculating mutual information across multiple MCD forward passes 
    mutual_info = entropy - np.mean(np.sum(-dropout_predictions * np.log(dropout_predictions + epsilon),
                                           axis=-1), axis=0)  # shape (n_samples,)


class double_conv(nn.Module):
    '''Do convolution operation twice'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(p=0.5),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up_func = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, out_ch, 3, padding=1)
            )
        else:
            self.up_func = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        '''
        input is of size (N_batchsize,C_channel,Height,Width)
        x1 ~ low resolution feature map
        x2 ~ high resolution feature map from skipping path 
        '''
        x1 = self.up_func(x1)
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
        
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class LocNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(LocNet, self).__init__()
        self.inc = inconv(n_channels, 4)
        self.down1 = down(4, 8)
        self.down2 = down(8, 16)
        self.down3 = down(16, 32)
        self.up1 = up(32, 16)
        self.up2 = up(16, 8)
        self.up3 = up(8, 4)
        self.outc = outconv(4, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)

class LocNetLarge(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(LocNetLarge, self).__init__()
        self.inc = inconv(n_channels, 8)
        self.down1 = down(8, 16)
        self.down2 = down(16, 32)
        self.down3 = down(32, 64)
        self.up1 = up(64, 32)
        self.up2 = up(32, 16)
        self.up3 = up(16, 8)
        self.outc = outconv(8, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)


class BCELossW(nn.Module):
    ''' Weighted Binary Cross Entropy loss '''
    def __init__(self):
        super(BCELossW, self).__init__()
    
    def forward(self, predict, target, weights=[1,10]):
        assert len(weights) == 2
        landmark_weight = weights[1]
        background_weight = weights[0]
        loss = torch.neg(landmark_weight * torch.mean(target * torch.log(predict)) + \
            background_weight * torch.mean((1-target) * torch.log(1-predict)))
        return loss

class LocLoss(nn.Module):
    ''' Weighted Binary Cross Entropy loss '''
    def __init__(self):
        super(LocLoss, self).__init__()
    
    def forward(self, predict, target, weights=[1,10]):
        assert len(weights) == 2
        landmark_weight = weights[1]
        background_weight = weights[0]
        loss = torch.neg(landmark_weight * torch.mean(target * torch.log(predict)) + \
            background_weight * torch.mean((1-target) * torch.log(1-predict)))
        return loss

class DiceLoss(nn.Module):
    ''' DICE coefficient loss '''
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def forward(self, predict, target):
        assert predict.size() == target.size(), "Input sizes must be equal."
        assert predict.dim() == 4, "Input must be a 4D Tensor."
        smooth = 1.
        predict = torch.sigmoid(50*(predict-0.5))
        intersection = torch.sum(predict * target, dim = (2,3)).squeeze()
        predict_sum = torch.sum(predict, dim = (2,3)).squeeze()
        target_sum = torch.sum(target, dim = (2,3)).squeeze()
        loss = torch.mean((2 * intersection + smooth) / (predict_sum + target_sum +smooth))
        return 1 - loss

#%%
if __name__ == '__main__':
    timestr = time.strftime("%Y%m%d-%H%M%S")
    results_root = "./results/exp_"+"tensets_fullwave"
    print("Training started, results folder: "+results_root)
    os.makedirs(results_root)
    for test_group_idx in range(0, 10):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(results_root+"/rec_weights_"+timestr):
            os.makedirs(results_root+"/rec_weights_"+timestr)

        # training setup
        batch_size_train = 4
        batch_size_test = 1
        epochs = 2000
        use_cuda = True
        # loss_fn = BCELossW()
        loss_fn = nn.MSELoss()
        # loss_fn = nn.L1Loss()
        net = LocNetLarge(n_channels=21, n_classes=1)
        if use_cuda:
            net = net.cuda()
        lr = 1e-4
        optimizer = optim.Adam(net.parameters(), lr=lr)

        # analysis rec
        counter = []
        loss_history = []
        loss_history_test = []
        dist_avg_all = []
        is_eval = True
        rec_step = 5
        save_step = 50

        # load data
        train_set = SSSLocDatasetTen(data_path="./dataset/data_point_tensets_good.pk", random_seed=0, mode='train', test_group=test_group_idx, full_wave=False)
        test_set = SSSLocDatasetTen(data_path="./dataset/data_point_tensets_good.pk", random_seed=0, mode='test', test_group=test_group_idx, is_aug=False, full_wave=False)
        train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size_test, shuffle=True)
        
        iteration=0

        for epoch in range(epochs):
            for batch_idx, batch_sample in enumerate(train_loader):
                img = batch_sample['img']
                y = batch_sample['sss_gt']
                if use_cuda:
                    img = img.to(device=torch.device("cuda"), dtype=torch.float)
                    y = y.to(device=torch.device("cuda"), dtype=torch.float)
                
                net.train()
                y_pred = net(img)
                # loss = loss_fn(y_pred.clamp(1e-8,1-1e-7), y, [1, 5]) 
                loss = loss_fn(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            result_np = y_pred.detach().cpu().numpy()
            sssloc_np = y.detach().cpu().numpy()
            loc_pred = np.unravel_index(result_np[0,0,:,:].argmax(), result_np[0,0,:,:].shape)
            loc_true = np.unravel_index(sssloc_np[0,0,:,:].argmax(), sssloc_np[0,0,:,:].shape)
            dist = np.sqrt((loc_pred[0]-loc_true[0])**2 + (loc_pred[1]-loc_true[1])**2)

            # if batch_idx % int(len(train_set)/batch_size/10) == 0:
            print("Currently at Group: %d, Epoch %d, Loss %f, Dist %f" % (test_group_idx, epoch, loss.item(), dist))

            if epoch % rec_step == 0:
                # print("Epoch %d, Loss %f" % (epoch, loss.item()))
                iteration += rec_step
                counter.append(iteration)
                loss_history.append(loss.item())
                if is_eval:
                    with torch.no_grad():
                        loss_current_total = 0
                        dist_sum = 0
                        for batch_idx_test, batch_sample_test in enumerate(test_loader):
                            net.eval()
                            enable_dropout(net)
                            test_img = batch_sample_test['img'].type(torch.FloatTensor)
                            test_y = batch_sample_test['sss_gt'].type(torch.FloatTensor)
                            if use_cuda:
                                test_img = test_img.cuda()
                                test_y = test_y.cuda()
                            test_y_pred = net(test_img)
                            
                            result_np = test_y_pred.detach().cpu().numpy()
                            sssloc_np = test_y.detach().cpu().numpy()
                            
                            loc_pred = np.unravel_index(result_np[0,0,:,:].argmax(), result_np[0,0,:,:].shape)
                            loc_true = np.unravel_index(sssloc_np[0,0,:,:].argmax(), sssloc_np[0,0,:,:].shape)

                            dist_sum += np.sqrt((loc_pred[0]-loc_true[0])**2 + (loc_pred[1]-loc_true[1])**2)
                            
                            # test_loss = loss_fn(test_y_pred.clamp(1e-8,1-1e-7), test_y, [1, 10]) 
                            test_loss = loss_fn(test_y_pred, test_y)
                            loss_current_total = loss_current_total + test_loss.item()
                            # eval(test_y_pred, test_y)
                        dist_avg = dist_sum / (batch_size_test * len(test_set))
                        loss_current_total = loss_current_total / (batch_size_test * len(test_set))
                        loss_history_test.append(loss_current_total)
                        dist_avg_all.append(dist_avg)
                        print("------ Group: %d, Epoch %d, Avg loss on test: %f" % (test_group_idx, epoch, loss_current_total))
                        print("------ Group: %d, Epoch %d, Avg test dist: %f" % (test_group_idx, epoch, dist_avg))
                        # print(loc_pred, loc_true,"     ", loc_pred[0]-loc_true[0], loc_pred[1]-loc_true[1])

                total_hist = [counter, loss_history, loss_history_test, dist_avg_all]

                with open(results_root+"/rec_weights_"+timestr+"/training_hist_"+timestr+".txt", "wb") as fp:
                    pickle.dump(total_hist, fp)

            if epoch % save_step == 0:
                current_dir = results_root+"/rec_weights_"+timestr
                for fname in os.listdir(current_dir):
                    if fname.endswith('.pt'):
                        filepath = current_dir + "/" + fname
                        os.remove(filepath)
                # only save the latest weight
                torch.save(net.state_dict(), results_root+"/rec_weights_"+timestr+"/net_weights_epoch_"+str(epoch)+".pt")