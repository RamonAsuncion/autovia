from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from torchvision.datasets import Cityscapes
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import segmentation_models_pytorch as smp
from pytorch_lightning import seed_everything, LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
import multiprocessing
import torchmetrics
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

DATA_DIR = "~/data/cityscapes/"
model_weights_dir = "./weights/"
model_checkpoints_dir = "./checkpoints/"
tensorboard_logger_dir = "./tb_logs/"

'''
Some notes: 

model input image needs to be of shape (N,C,H,W)
N - number of images
C - color channels (R,G,B) -- should be 3
H - height of image in pixels
W - width of image in pixels
'''

ignore_index=255
void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [ignore_index,7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', \
               'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', \
               'train', 'motorcycle', 'bicycle']
class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes=len(valid_classes)

# Transform applied to images for training.
myCustomTransform=A.Compose(
    [
        A.Resize(256, 512),
        A.HorizontalFlip(),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


colors = [   [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

label_colours = dict(zip(range(n_classes), colors))

# inv_normalize = transforms.Normalize(
#     mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
#     std=[1/0.229, 1/0.224, 1/0.255]
# )

def inv_normalize(*args):
    '''The inverse transform that we apply to images that was prepared for model input to now in a state ready for human view.'''
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.255]
    )
    return inv_normalize(*args)

def encode_segmap(mask):
    #remove unwanted classes and rectify the labels of wanted classes
    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_index
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask

def decode_segmap(temp):
    #convert gray scale to color
    temp=temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


class MyClass(Cityscapes):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            transformed=myCustomTransform(image=np.array(image), mask=np.array(target))            
        return transformed['image'],transformed['mask']
    #torch.unsqueeze(transformed['mask'],0)
    


class OurModel(LightningModule):
    def __init__(self):
        super(OurModel,self).__init__()
        #architecute
        self.layer = smp.Unet(
                    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    classes=n_classes,                      # model output channels (number of classes in your dataset)
                )

        #parameters
        self.lr=1e-3
        self.batch_size=4
        self.numworker=multiprocessing.cpu_count()//4

        self.criterion= smp.losses.DiceLoss(mode='multiclass')
        self.metrics = torchmetrics.JaccardIndex(task="multiclass", num_classes=n_classes)
        
        self.train_class = MyClass(DATA_DIR, split='train', mode='fine',
                        target_type='semantic',transforms=myCustomTransform)
        self.val_class = MyClass(DATA_DIR, split='val', mode='fine',
                        target_type='semantic',transforms=myCustomTransform)
    
    
    def process(self,image,segment):
        out=self(image)
        segment=encode_segmap(segment)
        loss=self.criterion(out,segment.long())
        jaccard=self.metrics(out,segment)
        return loss,jaccard
    
    def forward(self,x):
        return self.layer(x)


    def configure_optimizers(self):
        opt=torch.optim.AdamW(self.parameters(), lr=self.lr)
        return opt

    def train_dataloader(self):
        return DataLoader(self.train_class, batch_size=self.batch_size, 
                        shuffle=True,num_workers=self.numworker,pin_memory=True)

    def training_step(self,batch,batch_idx):
        image,segment=batch
        loss,jaccard=self.process(image,segment)
        self.log('train_loss', loss,on_step=False, on_epoch=True,prog_bar=True)
        self.log('train_jaccard', jaccard,on_step=False, on_epoch=True,prog_bar=False)
        return loss

    def val_dataloader(self):
        return DataLoader(self.val_class, batch_size=self.batch_size, 
                        shuffle=False,num_workers=self.numworker,pin_memory=True)
    
    def validation_step(self,batch,batch_idx):
        image,segment=batch
        loss,jaccard=self.process(image,segment)
        self.log('val_loss', loss,on_step=False, on_epoch=True,prog_bar=False)
        self.log('val_jaccard', jaccard,on_step=False, on_epoch=True,prog_bar=False)
        return loss