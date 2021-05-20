import torch
import numpy as np
from trainer import TrainingConfig, Trainer
from dataset import CIFAR10
from model import ConvNet
from torchvision.transforms import Compose,ToTensor,RandomHorizontalFlip,RandomRotation,ColorJitter,Normalize

train_set = CIFAR10(root="./cifar-10-batches-py",train=True,
                    transforms=Compose([
                        ToTensor(),
                        RandomHorizontalFlip(),
                        RandomRotation(degrees=10),
                        ColorJitter(brightness=0.5),
                        Normalize(mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618),
                                  std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628))
                    ]))

test_set = CIFAR10(root="./cifar-10-batches-py",train=False,
                   transforms=Compose([
                        ToTensor(),
                        Normalize(mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618),
                                  std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628))
                    ]))

train_config = TrainingConfig(max_epochs=100,
                              lr=0.00023570926966106847,
                              weight_decay=0.00021257445443209662,
                              ckpt_path="./models/CIFAR10.pt",
                              batch_size=64,
                              num_workers=4)


if __name__ == "__main__":

    model = ConvNet()
    model.load_state_dict(torch.load("./Final_Model.pt")) #Uncomment this if you want to resume training process. Makesure to change the model name 
    trainer = Trainer(model,train_dataset=train_set,
                    test_dataset=test_set,config=train_config)

    trainer.train_losses = torch.load("./log/train_losses.pt")
    trainer.train_accuracies = torch.load("./log/train_accuracies.pt")
    trainer.test_losses = torch.load("./log/test_losses.pt")
    trainer.test_accuracies = torch.load("./log/test_accuracies.pt")

    trainer.train()

    torch.save(model.state_dict(),"./models/Model.pt")
    torch.save(trainer.train_losses,"./log/train_losses2.pt")
    torch.save(trainer.train_accuracies,"./log/train_accuracies2.pt")
    torch.save(trainer.test_losses,"./log/test_losses2.pt")
    torch.save(trainer.test_accuracies,"./log/test_accuracies2.pt")
    
    #Learning rates and weight_decay used
    # lr = 0.0009446932175584296 -> 0.00023570926966106847
    # reg = 0.00011257445443209662 -> 0.00021257445443209662