import sys
import torchvision.transforms.functional as F
sys.modules['torchvision.transforms.functional_tensor'] = F

import cv2
import os
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def upscale_image(input_path, output_path):
    # Using the GAN architecture (same RRDBNet structure)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    # SWITCHED: Now pointing to the GAN weights for more detail
    model_weights = os.path.join('weights', 'RealESRGAN_x4plus.pth')

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_weights, 
        model=model,
        tile=1024,       
        tile_pad=10,
        pre_pad=0,
        half=True,       
        device='cuda'
    )

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return

    try:
        print("Upscaling with GAN model (adding detail)...")
        # CHANGED: outscale=2 will give you a 2x final image
        output, _ = upsampler.enhance(img, outscale=2)
        
        cv2.imwrite(output_path, output)
        print(f"Success! 2x sharp image saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    input_img = 'input.png'
    output_img = 'output_2x.png'

    if os.path.exists(input_img):
        upscale_image(input_img, output_img)
        
