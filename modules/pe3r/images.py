import numpy as np
import torch
from PIL import Image
from modules.mobilesamv2.utils.transforms import ResizeLongestSide
from modules.dust3r.utils.image import _resize_pil_image

class Images:
    def __init__(self, filelist, device, size=512):
        
        self.pil_images = []
        self.pil_images_size = []
        self.np_images = []
        self.np_images_size = []
        # -- original images --
        tmp_images = []
        first_image_size = None
        all_images_same_size = True
        
        for img_path in filelist:
            # === [수정된 부분 시작] ===
            # Gradio TempFile 객체에서 파일 경로(String)만 추출
            if hasattr(img_path, 'name'):
                img_path = img_path.name
            # === [수정된 부분 끝] ===

            pil_image = Image.open(img_path).convert("RGB")
            tmp_images.append(pil_image)

            current_image_size = pil_image.size
            if first_image_size is None:
                first_image_size = current_image_size
            else:
                if current_image_size != first_image_size:
                    all_images_same_size = False
        
        for img in tmp_images:

            if not all_images_same_size:
                # resize long side to 512
                pil_image = _resize_pil_image(img, size)
                W, H = pil_image.size
                cx, cy = W//2, H//2
                halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
                if W == H:
                    halfh = 3*halfw/4
                pil_image = pil_image.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
            else:
                pil_image = img

            np_image = np.array(pil_image)

            height, width = pil_image.size
            np_shape = np_image.shape[:2]

            self.pil_images.append(pil_image)
            self.np_images.append(np_image)

            self.pil_images_size.append((height, width))
            self.np_images_size.append(np_shape)
            
            
        # -- sam2 images --
        img_mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
        img_std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
        self.sam2_images = []
        # TODO
        self.sam2_video_size = (self.pil_images_size[0][1], self.pil_images_size[0][0])
        self.sam2_input_size = 512
        for pil_image in self.pil_images:
            np_image = np.array(pil_image.resize((self.sam2_input_size, self.sam2_input_size)))
            np_image = np_image / 255.0
            sam2_image = torch.from_numpy(np_image).permute(2, 0, 1)
            self.sam2_images.append(sam2_image)
        self.sam2_images = torch.stack(self.sam2_images)
        self.sam2_images -= img_mean
        self.sam2_images /= img_std
        self.sam2_images.to(device)

        # -- sam1 images --
        self.sam1_images = []
        self.sam1_images_size = []
        self.sam1_input_size = 1024
        self.sam1_transform = ResizeLongestSide(self.sam1_input_size)
        for np_image in self.np_images:
            sam1_image = self.sam1_transform.apply_image(np_image)
            sam1_image_torch = torch.as_tensor(sam1_image, device=device)
            transformed_image = sam1_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
            
            self.sam1_images.append(transformed_image)
            self.sam1_images_size.append(tuple(transformed_image.shape[-2:]))