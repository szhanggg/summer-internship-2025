import sys
import torch
import lightning as L
import glob
import numpy as np

from abidatamodule import AbiDataModule
from models import LightningModel


BATCH_SIZE = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 14
LEARNING_RATE = 1e-4
TRAINING_SPLIT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
EPOCHS = 100
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else None
DATALOADER_WORKERS = 71

#MODEL_NAMES = ["satfull", "sathalf", "satquarter", "sateighth", "unetfull", "unethalf", "unetquarter", "uneteighth"]
MODEL_NAMES = ["unetfull"]

datapath = '/explore/nobackup/projects/pix4dcloud/szhang16/abiChips/GOES-16/'

class SaveTestResultsCallback(L.Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self.test_results = []
        self.model_name = None
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.test_results.append(outputs["iou"].detach().cpu().numpy())
    
    def on_test_end(self, trainer, pl_module):
        self.test_results = np.array(self.test_results)
        self.model_name = pl_module.hparams.model_name
        np.save(self.save_dir + f"{self.model_name}.npy", self.test_results)


if __name__ == '__main__':
    datamodule = AbiDataModule(chip_dir=datapath, batch_size=1, num_workers=DATALOADER_WORKERS)

    savecallback = SaveTestResultsCallback(save_dir="/explore/nobackup/projects/pix4dcloud/szhang16/test_results/")

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[savecallback],
        enable_checkpointing=False,
    )

    for model_name in MODEL_NAMES:
        best_ckpt = glob.glob("/explore/nobackup/people/szhang16/checkpoints/" + model_name + "/best-*.ckpt")
        if not best_ckpt:
            print(f"No checkpoint found for model {model_name}")
            continue
        
        if model_name.startswith("sat"):
            model = LightningModel.load_from_checkpoint(best_ckpt[0], model_name="SatVisionUNet", in_channels=IMG_CHANNELS, num_classes=1, lr=LEARNING_RATE, freeze_encoder=True, final_size=(91, 40))
        elif model_name.startswith("unet"):
            model = LightningModel.load_from_checkpoint(best_ckpt[0], model_name="UNet", in_channels=IMG_CHANNELS, num_classes=1, lr=LEARNING_RATE, freeze_encoder=False)

        # Loop through the test datasat pytorch datalaoder
        trainer.test(model=model, datamodule=datamodule)