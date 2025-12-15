# YOLOE3R


### Quick Start
#### Install
```bash
conda create --name yoloe3r python=3.13
conda activate yoloe3r
```

#### Install torch
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
(torch 2.9.1, cuda 12.6 사용)
```

#### Clone
```bash
git clone https://github.com/cshyun1031/team6_yoloe3r.git
cd team6_yoloe3r
pip install -r requirements.txt
```

#### Setting 

```Python
# config.py
# Get your API key from Google AI Studio (https://aistudio.google.com/)
API_KEY = "YOUR_GOOGLE_GEMINI_API_KEY_HERE" # Paste your own Gemini API key here
INITIAL_IMAGE_PATHS = [
    "path/to/your/-30degree_img.jpg", # -30degree picture
    "path/to/your/original_img.jpg", # original picture
    "path/to/your/30degree_img.jpg" # 30 degree picture
]
```

#### Usage
```bash
python IFU_demo.py
```

#### Notes
- The necessary database files are already included in this repository, so you do not need to prepare or download a separate DB.
- In track_crop.py, GPU settings (for example, the CUDA device index) are hard coded.
- Please update these settings according to your own GPU setup before execution.


### Acknowledgements
- [DUSt3R](https://github.com/naver/dust3r) / [MASt3R](https://github.com/naver/mast3r) / [PE3R]
- [SAM](https://github.com/facebookresearch/segment-anything) / [SAM2](https://github.com/facebookresearch/sam2) / [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [SigLIP](https://github.com/google-research/big_vision)

### BibTeX
```BibTeX
@article{hu2025pe3r,
  title={PE3R: Perception-Efficient 3D Reconstruction},
  author={Hu, Jie and Wang, Shizun and Wang, Xinchao},
  journal={arXiv preprint arXiv:2503.07507},
  year={2025}
}
```
