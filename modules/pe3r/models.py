import os
import sys

from transformers import AutoTokenizer, AutoModel, AutoProcessor, SamModel
from modules.mast3r.model import AsymmetricMASt3R

# from modules.sam2.build_sam import build_sam2_video_predictor

from sam2.sam2_video_predictor import SAM2VideoPredictor
from ultralytics import YOLO

class Models:
    def __init__(self, device):
        # -- mast3r --
        # MAST3R_CKP = './checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
        MAST3R_CKP = 'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'
        self.mast3r = AsymmetricMASt3R.from_pretrained(MAST3R_CKP).to(device)

        # -- sam2 --
        # SAM2_CKP = "./checkpoints/sam2.1_hiera_large.pt"
        # SAM2_CKP = 'hujiecpp/sam2-1-hiera-large'
        # SAM2_CONFIG = "./configs/sam2.1/sam2.1_hiera_l.yaml"
        # self.sam2 = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CKP, device=device, apply_postprocessing=False)
        # self.sam2.eval()
        self.sam2 = SAM2VideoPredictor.from_pretrained('facebook/sam2.1-hiera-large', device=device)

        self.seg_model = YOLO("/content/PE3R/yoloe-11l-seg.pt")
        #self.seg_model.set_classes(["bed", "chair", "desk","table",'book','sofa'])
        # -- siglip --
      
