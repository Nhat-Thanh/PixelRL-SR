import json
import yaml
from hat.archs.hat_arch import HAT
import torch
import torch.nn.functional as F
import torchvision.io as io
import matplotlib.pyplot as plt
import cv2
import numpy as np

def pad_image_to_factor_of_16(image_tensor):
    # Get the original dimensions of the image tensor
    original_height, original_width = image_tensor.size(-2), image_tensor.size(-1)

    # Calculate the padding amounts needed on x and y dimensions
    pad_x = (16 - (original_width % 16)) % 16
    pad_y = (16 - (original_height % 16)) % 16

    # Decide whether to pad from left or both sides based on whether the original width is odd or even
    pad_left = pad_x // 2 if original_width % 2 == 0 else pad_x
    pad_right = pad_x // 2 if original_width % 2 == 0 else pad_x - pad_left

    # Decide whether to pad from top or both sides based on whether the original height is odd or even
    pad_top = pad_y // 2 if original_height % 2 == 0 else pad_y
    pad_bottom = pad_y // 2 if original_height % 2 == 0 else pad_y - pad_top

    # Pad the image tensor
    padded_image_tensor = torch.nn.functional.pad(image_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    return padded_image_tensor


HAT_model = HAT(upscale=4,
            in_chans=3,
            img_size=64,
            window_size=16,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            img_range=1,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv').to('cuda')
model_path = "sr_weight/HAT_SRx4.pth"
HAT_model.load_state_dict(torch.load(model_path)['params_ema'])
HAT_model.eval()
# print(HAT_model.state_dict())

filepath = 'dataset/test/x4/labels/img_005_SRF_4_HR.png'
image = io.read_image(filepath, io.ImageReadMode.RGB)
print(image.shape)

image = pad_image_to_factor_of_16(image)

print(image.shape)


plt.imshow(  image.permute(1, 2, 0) )
plt.show()

with torch.no_grad():
    # predictions = HAT_model(image.to('cuda').float())
    predictions = HAT_model(image.to('cuda').float())

# print(predictions[0])
plt.imshow(predictions[0].int().detach().cpu().permute(1, 2, 0))
plt.show()