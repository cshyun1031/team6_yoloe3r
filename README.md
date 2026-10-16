# YOLOE3R
Turns ordinary room photos into an interactive 3D model, so you can see your space from any angle and try out new furniture with an LLM (Gemini 2.5 Flash).

### Demo

<table>
  <tr>
    <th width="40%">Before: room photo</th>
    <th width="60%">After: 3D reconstruction</th>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/73eaaf45-09c1-41d6-947b-ca1626ef8942" width="100%"/></td>
    <td><img src="https://github.com/user-attachments/assets/1bb0d820-7093-4402-a572-9db8cc59b025" width="100%"/></td>
  </tr>
</table>

<p align="center">
  <img src="https://github.com/user-attachments/assets/fb7e659d-9806-4eb2-bf39-0aeeac247c4e" width="49%"/>
  <img src="https://github.com/user-attachments/assets/6d964795-d7c2-4258-a424-940fd5f86893" width="49%"/>
</p>
<p align="center"><sub>The same room viewed from different angles. Colored frames show the estimated camera positions.</sub></p>



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

#### Run
```bash
python IFU_demo.py
```

#### Notes
- The necessary database files are already included in this repository, so you do not need to prepare or download a separate DB.
- In track_crop.py(modules/IR/track_crop.py), GPU settings (such as the CUDA device index) are hard coded. 
  Please modify them to match your own GPU environment before running the code.


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
