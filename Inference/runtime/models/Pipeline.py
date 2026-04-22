import os
import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from models.Encoder import Checkboard_Encoder
from models.Decoder import StegaStampDecoder, FilterLayer, BlockDecoder, ImageDecoder, HeatMapDecoder, BCDecoder
from analysis_lgz import cal_contour
from einops import repeat
from attack.attack_layer import AttackLayer
from attack.JpegCompression import JpegFASL
from models.loss import DiceLoss, FocalLoss
import segmentation_models_pytorch as smp
import cv2
import torchvision
from PIL import Image
from data.utils import decode
from statistics import StatisticsError
from einops import rearrange


class Pipeline(pl.LightningModule):
    def __init__(self, secret_bits, image_size=256, extractor_weights=1., lr=2e-5):
        super(Pipeline, self).__init__()
        self.image_size = image_size
        self.extractor_weights = extractor_weights
        self.lr = lr

        self.encoder = Checkboard_Encoder(input_channel=secret_bits)
        self.filter = FilterLayer()
        self.decoder = StegaStampDecoder(secret_size=secret_bits)

        
        
        
        
        
        
        self.attack_layer = None

        self.bce_loss = nn.BCEWithLogitsLoss().eval()
        self.mse_loss = nn.MSELoss().eval()

    def get_rnd_pos(self, big_size=512, small_size=256):
        collaspe = big_size - small_size
        rnd_h = np.random.randint(0, max(0, collaspe))
        rnd_w = np.random.randint(0, max(0, collaspe))
        rnd_pos = (rnd_w, collaspe - rnd_w, rnd_h, collaspe - rnd_h)
        return rnd_pos

    def forward(self, secret_msg, input_image):
        watermark = self.encoder(secret_msg)
        rnd_pos = self.get_rnd_pos()
        pad_watermark = F.pad(watermark, rnd_pos, mode='constant', value=0.)

        watermarked_image = input_image + pad_watermark
        attacked_image = watermarked_image if self.attack_layer is None else self.attack_layer(watermarked_image)
        rnd_noise = torch.rand(attacked_image.shape) / 255.  
        rnd_noise = rnd_noise.to(attacked_image)
        attacked_watermark = rnd_noise + attacked_image
        filtered_hf = self.filter(attacked_watermark)
        extracted_msg = self.decoder(filtered_hf)
        return watermark, pad_watermark, watermarked_image, extracted_msg

    def cal_loss(self, gt_secret, extracted, watermark):
        zeros_watermark = torch.zeros(watermark.shape).to(watermark)
        mse_loss = self.mse_loss(watermark, zeros_watermark)
        bce_loss = self.bce_loss(extracted, gt_secret)
        loss = mse_loss + self.extractor_weights * bce_loss
        loss_dict = {
            'mse_loss': mse_loss.clone().detach().mean(),
            'bce_loss': bce_loss.clone().detach().mean(),
        }
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        gt_secret, input_image = batch
        secret_input = repeat(gt_secret, 'b c -> b c h w', h=self.image_size, w=self.image_size)
        watermark, pad_watermark, stego, extracted_msg = self(secret_input, input_image)
        loss, loss_dict = self.cal_loss(gt_secret, extracted_msg, watermark)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        lr = self.lr
        params_list = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.filter.parameters())
        optimizer = torch.optim.Adam(params_list, lr=lr, betas=(0.5, 0.9))
        return [optimizer], []

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, log_ema=False, **kwargs):
        log = dict()
        gt_secret, input_image = batch
        secret_input = repeat(gt_secret, 'b c -> b c h w', h=self.image_size, w=self.image_size)
        watermark, pad_watermark, stego, extracted_msg = self(secret_input, input_image)
        log['comparison'] = torch.cat([pad_watermark, stego], dim=-2)
        return log


class Extractor(pl.LightningModule):
    def __init__(self, image_size=192, lr=2e-5, tile_size=12, do_parser=False):
        super(Extractor, self).__init__()
        self.lr = lr
        self.tile_size = tile_size
        
        
        
        
        self.decoder = ImageDecoder(tile_size=tile_size, image_size=image_size, do_parser=do_parser)

        
        
        
        
        
        
        

        self.bce_loss = nn.BCEWithLogitsLoss().eval()
        self.sigmoid = nn.Sigmoid().eval()
        

    def forward(self, input_image):
        extracted_msg = self.decoder(input_image)
        return extracted_msg

    def cal_loss(self, gt_secret, extracted):
        bce_loss = self.bce_loss(extracted, gt_secret)
        loss = bce_loss
        loss_dict = {
            'bce_loss': bce_loss.clone().detach().mean(),
        }
        out = torch.round(self.sigmoid(extracted))
        acc = torch.eq(out, gt_secret).float().mean()
        loss_dict['ex_acc'] = acc.clone().detach()
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        gt_secret, input_image = batch
        extracted_msg = self(input_image)
        
        loss, loss_dict = self.cal_loss(gt_secret, extracted_msg)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        lr = self.lr
        params_list = list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params_list, lr=lr, betas=(0.5, 0.9))
        return [optimizer], []

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, log_ema=False, **kwargs):
        log = dict()
        gt_secret, input_image = batch
        
        
        log['input'] = input_image
        return log


class Locator(pl.LightningModule):
    def __init__(self, lr=2e-5):
        super(Locator, self).__init__()
        self.lr = lr
        
        
        
        
        
        
        
        
        

        
        

        ENCODER = 'se_resnext50_32x4d'
        ENCODER_WEIGHTS = 'imagenet'
        CLASSES = ['dan']
        ACTIVATION = None  
        
        self.decoder = smp.UnetPlusPlus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
        )

        attack_opts = {
            'JPEG': [80, 100],
            'CROP': [1.0, 1.0],
            'RESIZE': [1.0, 1.0],
            'GAUSSIAN': 0.00038
        }
        self.attack_layer = AttackLayer(attack_opts)

        self.dice_loss = DiceLoss().eval()

        self.sigmoid = nn.Sigmoid().eval()
        

    def forward(self, input_image):
        
        localized = self.decoder(input_image)
        pred_loc = self.sigmoid(localized)
        
        return pred_loc

    def cal_loss(self, gt, extracted):
        dice_loss = self.dice_loss(extracted, gt)
        loss = dice_loss.mean()
        loss_dict = {
            'dice_loss': dice_loss.clone().detach().mean(),
        }
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        input_image, gt_loc = batch
        pred_loc = self(input_image)
        
        loss, loss_dict = self.cal_loss(gt_loc, pred_loc)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        lr = self.lr
        params_list = list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params_list, lr=lr, betas=(0.5, 0.9))
        return [optimizer], []

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, log_ema=False, **kwargs):
        log = dict()
        input_image, gt_loc = batch
        pred_loc = self(input_image)
        
        input_image = torch.cat([pred_loc, gt_loc], dim=-2)
        log['input'] = input_image
        return log


class TwoStage(pl.LightningModule):
    def __init__(self, out_dir, do_parser=False, secret_hw=8, lr=2e-5, block_size=128, tile_size=12):
        super(TwoStage, self).__init__()
        self.lr = lr
        self.locator = Locator()
        self.do_parser = do_parser
        self.extractor = Extractor(tile_size=tile_size, image_size=block_size, do_parser=do_parser)
        self.block_size = block_size
        self.out_dir = out_dir
        self.secret_hw = secret_hw
        self.tile_size = tile_size

        attack_opts = {
            'JPEG': [80, 100],
            'CROP': [1.0, 1.0],
            'RESIZE': [1.0, 1.0],
            'GAUSSIAN': 0.00038
        }
        self.attack_layer = AttackLayer(attack_opts)
        
        
        
        bce_weights = torch.ones((1, 1, self.secret_hw * self.secret_hw))
        
        
        
        pad_loc = [0, 1, 10, 11, 12, 13, 22, 23, 120, 121, 130, 131, 132, 133, 142, 143]
        self.pad_loc = pad_loc
        
        
        
        
        for loc in pad_loc:
            bce_weights[:, :, loc] = 0.
        bce_weights = rearrange(bce_weights, 'b c (h w) -> b c h w', h=secret_hw, w=secret_hw)
        self.bce_weights = bce_weights
        
        self.sigmoid = nn.Sigmoid().eval()
        

    def bce_loss(self, pred, gt):
        return F.binary_cross_entropy_with_logits(pred, gt, weight=self.bce_weights.to(pred), reduction='mean')

    def forward(self, input_image):
        
        
        extracted = self.extractor(input_image)
        return extracted

    def extract(self, content):
        out = []
        for i in content:
            tmp = decode(i)
            
            out.append(tmp)
        return out

    def cal_loss(self, gt_secret, extracted):
        batch_size = gt_secret.shape[0]
        bce_loss = self.bce_loss(extracted, gt_secret)
        loss = bce_loss
        loss_dict = {
            'bce_loss': bce_loss.clone().detach().mean(),
        }
        out = torch.round(self.sigmoid(extracted))
        gt_secret = gt_secret.flatten(1).cpu().detach().numpy().astype(np.int8).tolist()
        out = out.flatten(1).cpu().detach().numpy().astype(np.int8).tolist()

        
        for loc in self.pad_loc[::-1]:
            for k in range(batch_size):
                gt_secret[k].pop(loc)
                out[k].pop(loc)
        
        
        
        

        
        

        acc = np.equal(out, gt_secret).mean()
        loss_dict['ex_acc'] = acc
        return loss, loss_dict

    def getContour(self, location_image, input_image):
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
        
        coords = cal_contour(location_image)
        start_point = coords[0]
        end_point = coords[2]
        

        aligned_image = input_image[:, start_point[1]:end_point[1] + 1, start_point[0]:end_point[0] + 1]

        return aligned_image

    def training_step(self, batch, batch_idx):
        self.locator.eval()
        input_image, gt_loc, secret_msg = batch
        with torch.no_grad():
            if self.attack_layer is not None:  
                
                input_image = self.attack_layer(input_image)
            pred_loc = self.locator(input_image)
            pred_loc = self.sigmoid(pred_loc)

            grids_gt = (gt_loc.permute((0, 2, 3, 1)) * 255).cpu().detach().numpy().astype(np.uint8)
            grids_pred = (pred_loc.permute((0, 2, 3, 1)) * 255).cpu().detach().numpy().astype(np.uint8)
            batch_images = []
            for image_index in range(len(grids_gt)):
                image_gt = grids_gt[image_index]
                image = grids_pred[image_index]
                tmp_input_image = input_image[image_index]

                _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)  
                try:
                    aligned_image = self.getContour(image, tmp_input_image)
                except StatisticsError:
                    
                    
                    
                    
                    aligned_image = None

                if aligned_image is None or aligned_image.shape[-1] == 0 or aligned_image.shape[-2] == 0:
                    _, image_gt = cv2.threshold(image_gt, 127, 255, cv2.THRESH_BINARY)  
                    aligned_image = self.getContour(image_gt, tmp_input_image)
                    print('动用gt的定位一次')
                aligned_image = F.interpolate(aligned_image.unsqueeze(0), (self.block_size, self.block_size))
                batch_images.append(aligned_image)
            new_input_image = torch.cat(batch_images, dim=0).to(input_image)

        extracted = self(new_input_image)
        
        loss, loss_dict = self.cal_loss(secret_msg, extracted)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        input_image, gt_loc, secret_msg = batch
        with torch.no_grad():
            if self.attack_layer is not None:  
                
                input_image = self.attack_layer(input_image)
            pred_loc = self.locator(input_image)
            pred_loc = self.sigmoid(pred_loc)

            grids_gt = (gt_loc.permute((0, 2, 3, 1)) * 255).cpu().detach().numpy().astype(np.uint8)
            grids_pred = (pred_loc.permute((0, 2, 3, 1)) * 255).cpu().detach().numpy().astype(np.uint8)
            batch_images = []
            for image_index in range(len(grids_gt)):
                image_gt = grids_gt[image_index]
                image = grids_pred[image_index]
                tmp_input_image = input_image[image_index]

                _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)  
                try:
                    aligned_image = self.getContour(image, tmp_input_image)
                except StatisticsError:
                    
                    
                    
                    
                    aligned_image = None

                if aligned_image is None or aligned_image.shape[-1] == 0 or aligned_image.shape[-2] == 0:
                    _, image_gt = cv2.threshold(image_gt, 127, 255, cv2.THRESH_BINARY)  
                    aligned_image = self.getContour(image_gt, tmp_input_image)
                    print('动用gt的定位一次')
                aligned_image = F.interpolate(aligned_image.unsqueeze(0), (self.block_size, self.block_size))
                batch_images.append(aligned_image)
            new_input_image = torch.cat(batch_images, dim=0).to(input_image)

        extracted = self(new_input_image)
        
        loss, loss_dict = self.cal_loss(secret_msg, extracted)
        ex_acc = loss_dict['ex_acc']
        loss_dict['val_loss'] = ex_acc
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return ex_acc

    def test_step(self, batch, batch_idx):
        self.locator.eval()
        input_image, gt_loc, secret_msg = batch
        loss = 0.
        with torch.no_grad():
            if self.attack_layer is not None:  
                input_image, _, _ = self.attack_layer(input_image, quality=90)
            pred_loc = self.locator(input_image)
            pred_loc = self.sigmoid(pred_loc)

            grids_pred = (pred_loc.permute((0, 2, 3, 1)) * 255).cpu().detach().numpy().astype(np.uint8)
            batch_images = []
            for image_index in range(len(grids_pred)):
                image = grids_pred[image_index]
                tmp_input_image = input_image[image_index]

                _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)  
                try:
                    aligned_image = self.getContour(image, tmp_input_image)
                except StatisticsError:
                    aligned_image = None

                if aligned_image is not None and aligned_image.shape[-1] > 0 and aligned_image.shape[-2] > 0:
                   aligned_image = F.interpolate(aligned_image.unsqueeze(0), (self.block_size, self.block_size))
                   batch_images.append(aligned_image)
            if len(batch_images) > 0:
                new_input_image = torch.cat(batch_images, dim=0).to(input_image)
                extracted = self(new_input_image)
                
                loss, loss_dict = self.cal_loss(secret_msg, extracted)
                acc = float(loss_dict['ex_acc'])
                str_suffix = '_{%.4f}' % acc
                
                
                
                
                
                
                
                
                
                
                

            else:
                str_suffix = '_定位失败'
                
                
                
                
                
                
                
                
                
                
            base_count = len(os.listdir(self.out_dir))
            filename = "{%05d}" % (base_count) + str_suffix + '.png'
            path = os.path.join(self.out_dir, filename)
            cv2.imwrite(path, image)
        return loss


    def configure_optimizers(self):
        lr = self.lr
        params_list = list(self.extractor.parameters())
        optimizer = torch.optim.Adam(params_list, lr=lr, betas=(0.5, 0.9))
        return [optimizer], []

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, log_ema=False, **kwargs):
        log = dict()
        self.locator.eval()
        input_image, gt_loc, secret_msg = batch
        with torch.no_grad():
            pred_loc = self.locator(input_image)
            pred_loc = self.sigmoid(pred_loc)
            loc_image = torch.cat([pred_loc, gt_loc], dim=-2)

            grids_pred = (pred_loc.permute((0, 2, 3, 1)) * 255).cpu().detach().numpy().astype(np.uint8)
            grids_input = (input_image.permute((0, 2, 3, 1)) * 255).cpu().detach().numpy().astype(np.uint8)
            batch_images = []
            for image_index in range(len(grids_pred)):
                image = grids_pred[image_index]
                tmp_input_image = grids_input[image_index].copy()
                _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)  
                try:
                    coords = cal_contour(image)
                    start_point = coords[0]
                    end_point = coords[2]
                    

                    color = (0, 0, 255)
                    cv2.rectangle(tmp_input_image, start_point, end_point, color, 2)
                    tmp_input_image = torch.from_numpy(tmp_input_image / 255).permute(2, 0, 1).unsqueeze(0).to(gt_loc)
                    batch_images.append(tmp_input_image)

                except StatisticsError:
                    
                    
                    
                    
                    batch_images.append(tmp_input_image)
            input_images = torch.cat(batch_images, dim=0)
        loc_image = repeat(loc_image, 'b 1 h w -> b c h w', c=3)
        log_images = torch.cat([input_images, loc_image], dim=-2)
        log['input'] = log_images
        return log
