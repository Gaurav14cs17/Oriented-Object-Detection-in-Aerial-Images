import torch
import torch.nn as nn
import os
import numpy as np
import loss
import cv2
#import func_utils

def collater(data):
    pass


class TrainModule:
    def __init__(self , dataset , num_classes , model , decoder , down_ratio):
        torch.manual_seed(100)
        self.dataset = {

        }
        self.num_clasess = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.decoder = decoder
        self.down_ratio = down_ratio

    def save_model(self, path, epoch, model, optimizer):
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save({'epoch': epoch,'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss
        }, path)


