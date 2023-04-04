###############################################################
##### @Title:  ICDAR 2023 DTT in image1: Text Manipulation Classification
##### @Time:  2023/02/18
##### @Author: Frank
##### @Describe: 
        #  part0: data preprocess
        #  part1: build_transforme() & build_dataset() & build_dataloader()
        #  part2: build_model()
        #  part3: build_loss()
        #  part4: build_metric()
        #  part5: train_one_epoch() & valid_one_epoch() & test_one_epoch()
##### @To do: 
        #  submit code
##### @Reference:
        #  None
###############################################################
import os
import pdb
import cv2
import time
import glob
import random

from cv2 import transform
# import cupy as cp # https://cupy.dev/ => pip install cupy-cuda102
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import torch # PyTorch
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp # https://pytorch.org/docs/stable/notes/amp_examples.html

from sklearn.model_selection import StratifiedGroupKFold, KFold # Sklearn
import albumentations as A # Augmentations
import timm
import segmentation_models_pytorch as smp # smp

def set_seed(seed=42):
    ##### why 42? The Answer to the Ultimate Question of Life, the Universe, and Everything is 42.
    random.seed(seed) # python
    np.random.seed(seed) # numpy
    torch.manual_seed(seed) # pytorch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

###############################################################
##### part1: build_transforms & build_dataset & build_dataloader
###############################################################
# document: https://albumentations.ai/docs/
# example: https://github.com/albumentations-team/albumentations_examples
def build_transforms(CFG):
    data_transforms = {
        "train": A.Compose([
            # # dimension should be multiples of 32.
            # ref: https://github.com/facebookresearch/detr/blob/main/datasets/coco.py
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0)

            # A.HorizontalFlip(p=0.5),
            # # A.VerticalFlip(p=0.5),
            # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            # A.OneOf([
            #     A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            #     # A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
            #     A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            # ], p=0.25),
            # A.CoarseDropout(max_holes=8, max_height=CFG.img_size[0]//20, max_width=CFG.img_size[1]//20,
            #                 min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
            ], p=1.0),
        
        "valid_test": A.Compose([
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0)
            ], p=1.0)
        }
    return data_transforms

class build_dataset(Dataset):
    def __init__(self, df, train_val_flag=True, transforms=None):

        self.df = df
        self.train_val_flag = train_val_flag #
        self.img_paths = df['img_path'].tolist() 
        self.ids = df['img_name'].tolist()
        self.transforms = transforms

        if train_val_flag:
            self.label = df['img_label'].tolist()
        
    def __len__(self):
        return len(self.df)
        # return 8
    
    def __getitem__(self, index):
        #### id
        id       = self.ids[index]
        #### image
        img_path  = self.img_paths[index]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [h, w, c]
        
        if self.train_val_flag: # train
            ### augmentations
            data  = self.transforms(image=img)
            img = np.transpose(data['image'], (2, 0, 1)) # [c, h, w]
            gt = self.label[index]
            return torch.tensor(img), torch.tensor(int(gt))
        
        else:  # test
            ### augmentations
            data  = self.transforms(image=img)
            img = np.transpose(data['image'], (2, 0, 1)) # [c, h, w]
            return torch.tensor(img), id


def build_dataloader(df, fold, data_transforms):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)

    train_dataset = build_dataset(train_df, train_val_flag=True, transforms=data_transforms['train'])
    valid_dataset = build_dataset(valid_df, train_val_flag=True, transforms=data_transforms['valid_test'])

    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, num_workers=0, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, num_workers=0, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader

###############################################################
##### >>>>>>> part2: build_model <<<<<<
###############################################################
# document: https://timm.fast.ai/create_model
def build_model(CFG, pretrain_flag=False):
    if pretrain_flag:
        pretrain_weights = "imagenet"
    else:
        pretrain_weights = False
    model = timm.create_model(CFG.backbone, pretrained=pretrain_flag, num_classes=CFG.num_classes)
    model.to(CFG.device)
    return model

###############################################################
##### >>>>>>> part3: build_loss <<<<<<
###############################################################
def build_loss():
    CELoss = torch.nn.CrossEntropyLoss()
    return {"CELoss":CELoss}

###############################################################
##### >>>>>>> part4: build_metric <<<<<<
###############################################################
def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice

def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou
    
###############################################################
##### >>>>>>> part5: train & validation & test <<<<<<
###############################################################
def train_one_epoch(model, train_loader, optimizer, losses_dict, CFG):
    model.train()
    scaler = amp.GradScaler() 
    losses_all, ce_all = 0, 0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train ')
    for _, (images, gt) in pbar:
        optimizer.zero_grad()
        
        images = images.to(CFG.device, dtype=torch.float) # [b, c, w, h]
        gt  = gt.to(CFG.device)  

        with amp.autocast(enabled=True):
            y_preds = model(images) 
            ce_loss = losses_dict["CELoss"](y_preds, gt.long())
            losses = ce_loss
        
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses_all += losses.item() / images.shape[0]
        ce_all += ce_loss.item() / images.shape[0]

    current_lr = optimizer.param_groups[0]['lr']
    print("lr: {:.4f}".format(current_lr), flush=True)
    print("loss: {:.3f}, ce_all: {:.3f}".format(losses_all, ce_all), flush=True)
        
@torch.no_grad()
def valid_one_epoch(model, valid_loader, CFG):
    model.eval()
    correct = 0
    total = 0
    
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid ')
    for _, (images, gt) in pbar:
        images  = images.to(CFG.device, dtype=torch.float) # [b, c, w, h]
        gt  = gt.to(CFG.device) 
        
        y_preds = model(images) 

        _, y_preds = torch.max(y_preds.data, dim=1)
    
        correct += (y_preds==gt).sum()
        total += gt.shape[0]

    val_acc = correct/total
    print("val_acc: {:.2f}".format(val_acc), flush=True)
    
    return val_acc

# @torch.no_grad()
# def test_one_epoch(ckpt_paths, test_loader, CFG):
#     pred_strings = []
#     pred_ids = []
#     pred_classes = []
    
#     pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Test: ')
#     for _, (images, ids, h, w) in pbar:

#         images  = images.to(CFG.device, dtype=torch.float) # [b, c, w, h]
#         size = images.size()
#         masks = torch.zeros((size[0], 3, size[2], size[3]), device=CFG.device, dtype=torch.float32) # [b, c, w, h]
        
#         ############################################
#         ##### >>>>>>> cross validation infer <<<<<<
#         ############################################
#         for sub_ckpt_path in ckpt_paths:
#             model = build_model(CFG, test_flag=True)
#             model.load_state_dict(torch.load(sub_ckpt_path))
#             model.eval()
#             y_preds = model(images) # [b, c, w, h]
#             y_preds   = torch.nn.Sigmoid()(y_preds)
#             masks += y_preds/len(ckpt_paths)
        
#         masks = (masks.permute((0, 2, 3, 1))>CFG.thr).to(torch.uint8).cpu().detach().numpy() # [n, h, w, c]
#         result = masks2rles(masks, ids, h, w)
#         pred_strings.extend(result[0])
#         pred_ids.extend(result[1])
#         pred_classes.extend(result[2])
#     return pred_strings, pred_ids, pred_classes


if __name__ == '__main__':
    ###############################################################
    ##### >>>>>>> config <<<<<<
    ###############################################################
    class CFG:
        # step1: hyper-parameter
        seed = 42 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ckpt_fold = "ckpt_ddt1"
        ckpt_name = "efficientnetb0_img224224_bs4"  # for submit.
        tampered_img_paths = "../../data/train/tampered/imgs"
        untampered_img_paths = "../../data/train/untampered/"
        
        # step2: data
        n_fold = 4
        img_size = [224, 224]
        train_bs = 4
        valid_bs = train_bs * 2
        # step3: model
        backbone = 'efficientnet_b0'
        num_classes = 2
        # step4: optimizer
        epoch = 12
        lr = 1e-3
        wd = 1e-5
        lr_drop = 8
        # step5: infer
        thr = 0.5
    
    set_seed(CFG.seed)
    ckpt_path = f"./{CFG.ckpt_fold}/{CFG.ckpt_name}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    train_val_flag = True
    if train_val_flag:
        ###############################################################
        ##### part0: data preprocess
        ###############################################################
        col_name = ['img_name', 'img_path', 'img_label']
        imgs_info = []  # img_name, img_path, img_label
        for img_name in os.listdir(CFG.tampered_img_paths):
            if img_name.endswith('.jpg'): # pass other files
                imgs_info.append(["p_"+img_name, os.path.join(CFG.tampered_img_paths, img_name), 1])
            
        for img_name in os.listdir(CFG.untampered_img_paths):
            if img_name.endswith('.jpg'): # pass other files
                imgs_info.append(["n_"+img_name, os.path.join(CFG.untampered_img_paths, img_name), 0])
        
        imgs_info_array = np.array(imgs_info)    
        df = pd.DataFrame(imgs_info_array, columns=col_name)

        ###############################################################
        ##### >>>>>>> trick1: cross validation train <<<<<<
        ###############################################################
        # document: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
        # skf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        kf = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            df.loc[val_idx, 'fold'] = fold
        
        for fold in range(CFG.n_fold):
            print(f'#'*40, flush=True)
            print(f'###### Fold: {fold}', flush=True)
            print(f'#'*40, flush=True)

            ###############################################################
            ##### >>>>>>> step2: combination <<<<<<
            ##### build_transforme() & build_dataset() & build_dataloader()
            ##### build_model() & build_loss()
            ###############################################################
            data_transforms = build_transforms(CFG)  
            train_loader, valid_loader = build_dataloader(df, fold, data_transforms) # dataset & dtaloader
            model = build_model(CFG, pretrain_flag=True) # model
            optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, CFG.lr_drop) 
            losses_dict = build_loss() # loss

            best_val_acc = 0
            best_epoch = 0
            
            for epoch in range(1, CFG.epoch+1):
                start_time = time.time()
                ###############################################################
                ##### >>>>>>> step3: train & val <<<<<<
                ###############################################################
                train_one_epoch(model, train_loader, optimizer, losses_dict, CFG)
                lr_scheduler.step()
                val_acc = valid_one_epoch(model, valid_loader, CFG)
                
                ###############################################################
                ##### >>>>>>> step4: save best model <<<<<<
                ###############################################################
                is_best = (val_acc > best_val_acc)
                best_val_acc = max(best_val_acc, val_acc)
                if is_best:
                    save_path = f"{ckpt_path}/best_fold{fold}_epoch{epoch}.pth"
                    if os.path.isfile(save_path):
                        os.remove(save_path) 
                    torch.save(model.state_dict(), save_path)
                
                epoch_time = time.time() - start_time
                print("epoch:{}, time:{:.2f}s, best:{:.2f}\n".format(epoch, epoch_time, best_val_acc), flush=True)


    test_flag = False
    if test_flag:
        pass
