import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset

class CIFAR10(Dataset):
    
    def __init__(self,root,train=True,transforms=None):
        self.root = root
        self.transforms = transforms
        self.split = train
        
        self.data = []
        self.targets = []
        self.train_data = [file for file in os.listdir(root) if "data_batch" in file]
        self.test_data = [file for file in os.listdir(root) if "test_batch" in file]
                
        data_split = self.train_data if self.split else self.test_data
        
        for files in data_split:
            entry = self.extract(os.path.join(root,files))
            self.data.append(entry["data"])
            self.targets.extend(entry["labels"])
                
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))
        self.load_meta()
        
    def extract(self,filename):
        with open(filename,"rb") as f:
            batch_data = pickle.load(f,encoding="latin1")
        return batch_data  
    
    def load_meta(self):
        path = os.path.join(self.root,"batches.meta")
        with open(path,"rb") as infile:
            data = pickle.load(infile,encoding="latin1")
            self.classes = data["label_names"]
            self.classes_to_idx = {_class:i for i,_class in enumerate(self.classes)}
            
    def plot(self,image,target=None):
        if target is not None:
            print(f"Target :{target} class :{self.classes[target]}")
        plt.figure(figsize=(2,2))
        plt.imshow(image.permute(1,2,0))
        plt.show()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        image,target = self.data[idx],self.targets[idx]
        image = Image.fromarray(image)
        
        if self.transforms:
            image = self.transforms(image)
            
        return image,target