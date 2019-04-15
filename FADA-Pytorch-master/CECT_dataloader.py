import torch
import torchvision
import os
from torch.utils.data import Dataset
import numpy as np
class CECT_dataset(Dataset):#需要继承data.Dataset
    def __init__(self,path=None):
        # TODO
        # 1. Initialize file path or list of file names.
        self.path = path
        self.classes = os.listdir(self.path)
        self.class2id = {}
        self.imgs = []

        for each_class in self.classes:
            self.class2id[each_class] = len(self.class2id)
            for items in os.listdir(os.path.join(self.path,each_class)):
                self.imgs.append((os.path.join(self.path,each_class,items),each_class))

        
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        path, CECT_class = self.imgs[index]
        return torch.from_numpy(np.load(path)).unsqueeze(0),torch.from_numpy(np.array(self.class2id[CECT_class])).long()


    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.imgs)