import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchmetrics import JaccardIndex
from torchmetrics.functional import accuracy
import sys
import pytorch_lightning as pl

class UNET(nn.Module):
    def __init__(self, in_channels=16, classes=1):
        super().__init__()
        self.layers = [in_channels, 64, 128, 256, 512, 1024]
        
        self.double_conv_downs = nn.ModuleList(
            [self.__double_conv(layer, layer_n) for layer, layer_n in zip(self.layers[:-1], self.layers[1:])])
        
        self.up_trans = nn.ModuleList(
            [nn.ConvTranspose2d(layer, layer_n, kernel_size=2, stride=2)
             for layer, layer_n in zip(self.layers[::-1][:-2], self.layers[::-1][1:-1])])
            
        self.double_conv_ups = nn.ModuleList(
        [self.__double_conv(layer, layer//2) for layer in self.layers[::-1][:-2]])
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.final_conv = nn.Conv2d(64, classes, kernel_size=1)

        
    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return conv
        
    def forward(self, x):
        # down layers
        concat_layers = []
        
        for down in self.double_conv_downs:
            x = down(x)
            if down != self.double_conv_downs[-1]:
                concat_layers.append(x)
                x = self.max_pool_2x2(x)
        
        concat_layers = concat_layers[::-1]
        
        # up layers
        for up_trans, double_conv_up, concat_layer  in zip(self.up_trans, self.double_conv_ups, concat_layers):
            x = up_trans(x)
            if x.shape != concat_layer.shape:
                x = TF.resize(x, concat_layer.shape[2:])
            
            concatenated = torch.cat((concat_layer, x), dim=1)
            x = double_conv_up(concatenated)
            
        x = self.final_conv(x)
        x = F.interpolate(x, size=(91, 40), mode='bilinear', align_corners=False)

        x = x.squeeze(1)
        
        return x

    def predict(self, x):
        x = self.forward(x)
        x = torch.sigmoid(x)
        x = (x > 0.5).float()
        return x

class LightningUNET(pl.LightningModule):
    def __init__(self, in_channels=3, classes=1):
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.model = UNET(self.in_channels, self.classes)
        self.iou = JaccardIndex(task="binary", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.forward(inputs)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        self.log("Loss:", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.forward(inputs)
        preds = torch.sigmoid(logits) > 0.5
        loss = self.iou(preds, targets)

        self.log("test_iou", loss, prog_bar=True)
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
