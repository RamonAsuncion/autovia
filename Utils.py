from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from torchvision.datasets import Cityscapes
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
import segmentation_models_pytorch as smp
from pytorch_lightning import seed_everything, LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
import multiprocessing
import torchmetrics
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt

"""
Some notes:

model input image needs to be of shape (N,C,H,W)
N - number of images
C - color channels (R,G,B) -- should be 3
H - height of image in pixels
W - width of image in pixels
"""

ignore_index = 255
void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [
    ignore_index,
    7,
    8,
    11,
    12,
    13,
    17,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    31,
    32,
    33,
]
class_names = [
    "unlabelled",
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic_light",
    "traffic_sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]
class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes = len(valid_classes)

# Transform applied to images for training.
myCustomTransform = A.Compose(
    [
        A.Resize(256, 512),
        A.HorizontalFlip(),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


colors = [
    [0, 0, 0],
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
    """The inverse transform that we apply to images that was prepared for model input to now in a state ready for human view."""
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
    )
    return inv_normalize(*args)


def encode_segmap(mask):
    # remove unwanted classes and rectify the labels of wanted classes
    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_index
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask


def decode_segmap(temp):
    # convert gray scale to color
    temp = temp.numpy()
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

def plotThreeImages(image, mask_truth, mask_predict):
    '''
    assume image and all masks aare on the cpu and NOT on the gpu.
    assume image is already inverted and all masks are decoded.
    '''
    fig,ax=plt.subplots(1,3,figsize=(10,5),facecolor='white')  
    if isinstance(image, type(torch.tensor)):
        ax[0].imshow(np.moveaxis(image.numpy(),0,2)) #(3,256, 512)
    else:
        # ax[0].imshow(np.moveaxis(image,0,2)) #(3,256, 512)
        ax[0].imshow(image)
    ax[1].imshow(mask_truth) #(256, 512, 3)
    ax[2].imshow(mask_predict) #(256, 512, 3)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[0].set_title('Input Image')
    ax[1].set_title('Ground mask')
    ax[2].set_title('Predicted mask')

class MyClass(Cityscapes):
    """
    Custom dataset class for cityscapes dataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.images:
            raise ValueError("The dataset is empty.")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert("RGB")

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            transformed = myCustomTransform(
                image=np.array(image), mask=np.array(target)
            )
        return transformed["image"], transformed["mask"]

    # torch.unsqueeze(transformed['mask'],0)


class OurModel(LightningModule):
    """
    Our model class for semantic segmentation.
    """

    def __init__(
        self,
        data_dir: str = "~/data/cityscapes/",
        lr: float = 1e-3,
        batch_size: int = 4,
    ):
        """
        inputs:
            data_dir (str) - location of the cityscapes data
        """
        super(OurModel, self).__init__()

        # architecute
        self.layer = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes,
        )

        # parameters
        self.lr = lr
        self.batch_size = batch_size
        self.numworker = multiprocessing.cpu_count() // 4

        self.criterion = smp.losses.DiceLoss(mode="multiclass")
        self.metrics = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=n_classes
        )

        if data_dir:
            self.train_class = MyClass(
                data_dir,
                split="train",
                mode="fine",
                target_type="semantic",
                transforms=myCustomTransform,
            )

        if data_dir:
            self.val_class = MyClass(
                data_dir,
                split="val",
                mode="fine",
                target_type="semantic",
                transforms=myCustomTransform,
            )

    def process(self, image, segment):
        """
        inputs:
            image (torch.Tensor) - the image to process
            segment (torch.Tensor) - the segmentation mask

        returns:
            loss (torch.Tensor) - the loss value
            jaccard (torch.Tensor) - the jaccard index
        """
        out = self(image)
        segment = encode_segmap(segment)
        loss = self.criterion(out, segment.long())
        jaccard = self.metrics(out, segment)
        return loss, jaccard

    def forward(self, x):
        """
        inputs:
            x (torch.Tensor) - the image

        returns:
            torch.Tensor - the output of the model
        """
        return self.layer(x)

    def configure_optimizers(self):
        """
        returns:
            torch.optim.Optimizer - the optimizer
        """
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return opt

    def train_dataloader(self):
        """
        returns:
            DataLoader - the training data loader
        """
        if self.train_class is None or not self.train_class:
            return None
        return DataLoader(
            self.train_class,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.numworker,
            pin_memory=True,
        )

    def training_step(self, batch, batch_idx):
        """
        inputs:
            batch (Tuple[torch.Tensor, torch.Tensor]) - the image and segmentation mask
            batch_idx (int) - the batch index

            returns:
                loss (torch.Tensor) - the loss value
            """
        image, segment = batch
        loss, jaccard = self.process(image, segment)
        self.log("train_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("train_jaccard", jaccard, on_step=False,
                 on_epoch=True, prog_bar=False)
        return loss

    def val_dataloader(self):
        """
        returns:
            DataLoader - the validation data loader
        """
        return DataLoader(
            self.val_class,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.numworker,
            pin_memory=True,
        )

    def validation_step(self, batch, batch_idx):
        """
        inputs:
            batch (Tuple[torch.Tensor, torch.Tensor]) - the image and segmentation mask
            batch_idx (int) - the batch index

            returns:
                loss (torch.Tensor) - the loss value"""
        image, segment = batch
        loss, jaccard = self.process(image, segment)
        self.log("val_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log("val_jaccard", jaccard, on_step=False,
                 on_epoch=True, prog_bar=False)
        return loss


class InferenceWorker:
    '''
    Runs the segmentation model on CPU.
    '''

    def __init__(self, model_weights_path: str):
        assert len(
            model_weights_path) > 0, 'Empty model_weights_path parameter detected.'

        model = OurModel(data_dir=None)
        model.load_state_dict(torch.load(model_weights_path))
        model.eval()
        self.model = model

    def remove_alpha_channel(self, img: np.ndarray) -> np.ndarray:
        """
        Remove the alpha channel from the image if present.
        """
        if img.shape[2] == 4:
            return img[:, :, :3]
        else:
            return img
        
    def resize_and_pad(self, img : np.ndarray) -> np.ndarray:
        """
        Resize the image preserving the aspect ratio and pad it to the required size.
        """
        # Resize the image preserving the aspect ratio
        target_height = 256
        target_width = 512

        height, width, _ = img.shape
        scale = min(target_height / height, target_width / width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        resized_img = cv2.resize(img, (new_width, new_height))

        # Pad the image to the required size
        pad_height = (target_height - new_height) // 2
        pad_width = (target_width - new_width) // 2
        padded_img = cv2.copyMakeBorder(resized_img, pad_height, target_height - new_height - pad_height,
                                        pad_width, target_width - new_width - pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return padded_img

    def prepare_image(self, img):
        """
        Prepare the image for the model.

        inputs:
            img (np.ndarray) - the image to prepare (assumed to be already resized and padded)

        returns:
            batch_img (np.ndarray) - the image prepared for the model
        """
        # move the color dimension
        padded_img = np.moveaxis(img, 2, 0)

        # # Normalize the image
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        padded_img = (padded_img - mean) / std

        # Add the batch dimension
        batch_img = np.expand_dims(padded_img, axis=0)

        return batch_img

    def raw_output_to_mask(self, modelOutput):
        '''
        converts the raw model output to the segmentation mask ready to view.
        assume modelOutput is on CPU
        assume modelOutput shape is of [N, 20, 256, 512] where N is batch size
        '''
        return decode_segmap(torch.argmax(modelOutput.detach(), 0))

    def segment(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Given an image in format (H, W, C) normalized to range 0 1, pass it through
        the pretrained model loaded in memory and return the segmented image.

        Returns
            mask_predict (np.ndarray) - the segmented image
            input_image (np.ndarray) - the input image (resized and padded -- but NOT custom normalized again)
        '''

        img = self.remove_alpha_channel(
            img)  # removes alpha channel if present
        # resizes, normalizes, and adds an extra batch dim

        resized_and_padded_img = self.resize_and_pad(img)
        tmp = self.prepare_image(resized_and_padded_img)

        tmp = tmp.astype(np.float32)  # ensure correct dtype
        inputImg = torch.tensor(tmp)  # make as torch tensor
        output = self.model(inputImg)  # pass thru model
        mask_predict = self.raw_output_to_mask(
            output[0])  # remove the batch dim

        return mask_predict, resized_and_padded_img
