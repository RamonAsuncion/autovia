'''train script'''

import torch
from Utils import OurModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
import torch

# for reproducibility
seed_everything(42, workers=True)

##########################################################

data_dir = "~/data/cityscapes"
model_weights_dir = "./weights/"

model_weights_dir = "./weights/"
model_name = "model-test.pth"

model_checkpoints_dir = "./checkpoints/"
tensorboard_logger_dir = "./tb_logs/"
logger_name = "semantic_segmentation_cityscapes_test"

resume_training = False

##########################################################

model = OurModel(data_dir=data_dir, lr=1e-3, batch_size=4)

if resume_training: 
    model.load_state_dict(torch.load(model_weights_dir+model_name))

checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                        dirpath=model_checkpoints_dir,
                                        filename='checkpoint_file',
                                        save_last=True)


logger = TensorBoardLogger(save_dir=tensorboard_logger_dir, name=logger_name)

trainer = Trainer(max_epochs = 5,
                    callbacks=[checkpoint_callback],
                    logger = logger,
                    deterministic=True)

print("Preparation complete. Begin model training:")

# TRAIN
trainer.fit(model)

# SAVE TRAINED MODEL
torch.save(model.state_dict(), model_weights_dir+model_name)

print("Train script complete.")