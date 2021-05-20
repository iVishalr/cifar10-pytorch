import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


class TrainingConfig:
    
    lr=3e-4
    betas=(0.9,0.995)
    weight_decay=5e-4
    num_workers=0
    max_epochs=10
    batch_size=64
    ckpt_path=None #Specify a model path here. Ex: "./Model.pt"
    shuffle=True
    pin_memory=True
    verbose=True
    
    def __init__(self,**kwargs):
        for key,value in kwargs.items():
            setattr(self,key,value)


class Trainer:
    def __init__(self,model,train_dataset,test_dataset,config):
        self.model = model
        self.train_dataset=train_dataset
        self.test_dataset=test_dataset
        self.config = config
        
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = self.model.to(self.device)
    
    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model,"module") else self.model
        torch.save(raw_model.state_dict(),self.config.ckpt_path)
        print("Model Saved!")
        
    def train(self):
        model,config = self.model,self.config
        raw_model = self.model.module if hasattr(self.model,"module") else self.model
        optimizer = raw_model.configure_optimizers(config)
        
        def run_epoch(split):
            is_train = split=="train"
            if is_train:
                model.train()
            else:
                model.eval() #important don't miss this. Since we have used dropout, this is required.
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data,batch_size=config.batch_size,
                                shuffle=config.shuffle,
                                pin_memory=config.pin_memory,
                                num_workers=config.num_workers)
            
            losses = []
            accuracies = []
            correct = 0
            num_samples = 0
            
            pbar = tqdm(enumerate(loader),total=len(loader)) if is_train and config.verbose else enumerate(loader)
            for it,(images,targets) in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)
                num_samples += targets.size(0)
                
                with torch.set_grad_enabled(is_train):
                    #forward the model
                    logits,loss = model(images,targets)
                    loss = loss.mean()
                    losses.append(loss.item())
                    
                with torch.no_grad():
                    predictions = torch.argmax(logits,dim=1) #softmax gives prob distribution. Find the index of max prob
                    correct+= predictions.eq(targets).sum().item()
                    accuracies.append(correct/num_samples)
                    
                if is_train:
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if config.verbose:
                        pbar.set_description(f"Epoch:{epoch+1} iteration:{it+1} | loss:{np.mean(losses)} accuracy:{np.mean(accuracies)} lr:{config.lr}")
                    
                    self.train_losses.append(np.mean(losses))
                    self.train_accuracies.append(np.mean(accuracies))
            
            if not is_train:
                test_loss = np.mean(losses)
                if config.verbose:
                    print(f"\nEpoch:{epoch+1} | Test Loss:{test_loss} Test Accuracy:{correct/num_samples}\n")
                self.test_losses.append(test_loss)
                self.test_accuracies.append(correct/num_samples)
                return test_loss
                
        best_loss = float('inf')
        test_loss = float('inf')
        
        for epoch in range(config.max_epochs):
            run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch("test")
                
            good_model = self.test_dataset is not None and test_loss < best_loss
            if config.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()