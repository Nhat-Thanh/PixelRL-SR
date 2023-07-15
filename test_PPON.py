import torch
from neuralnet import ESPCN_model, FSRCNN_model, SRCNN_model, VDSR_model
import torch.nn as nn
from PPON.PPON_model import PPONModel
from PPON import networks
import json
import os
from utils.common import *

SCALE = 4
LS_HR_PATHS = sorted_list(f"dataset/test/x{SCALE}/labels")
LS_LR_PATHS = sorted_list(f"dataset/test/x{SCALE}/data")
SIGMA = 0.3 if SCALE == 2 else 0.2

# declaringa a class
class obj:
     
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def main():
    
    opt = {
        'alpha': 1.0,
        'cuda': True,
        'isHR': True,
        'is_train': False,
        'models': 'sr_weight/PPON_G.pth',
        'pretrained_model_D': 'sr_weight/PPON_D.pth',
        'pretrained_model_G': 'sr_weight/PPON_G.pth',
        'only_y': True,
        'output_folder': 'result/Set5/',
        'save_path': 'save',
        'test_hr_folder': f'dataset/test/x{SCALE}/labels',
        'test_lr_folder': f'dataset/test/x{SCALE}/data',
        'upscale_factor': SCALE,
        'which_model': 'ppon'
    }
    opt = json.loads(json.dumps(opt), object_hook=obj)
    PPON = networks.define_G(opt)
    if isinstance(PPON, nn.DataParallel):
        PPON = PPON.module
    model_path = "sr_weight/PPON_G.pth"
    PPON.load_state_dict(torch.load(model_path), strict=True)
    psnr_sr = np.zeros(len(LS_HR_PATHS))
    ssim_sr = np.zeros(len(LS_HR_PATHS))
    for i in range(len(LS_HR_PATHS)):
        hr_image_path = LS_HR_PATHS[i]
        lr_image_path = LS_LR_PATHS[i]
        
        hr = read_image(hr_image_path)
        lr = read_image(lr_image_path)
        lr = gaussian_blur(lr, sigma=SIGMA)
        bicubic = upscale(lr, SCALE)

        bicubic = rgb2ycbcr(bicubic)
        lr = rgb2ycbcr(lr)
        hr= rgb2ycbcr(hr)


        bicubic = norm01(bicubic).unsqueeze(0)
        lr = norm01(lr).unsqueeze(0)
        hr = norm01(hr).unsqueeze(0)

        if opt.cuda:
            PPON = PPON.cuda()
            lr = lr.cuda()
        with torch.no_grad():
            out_c, out_s, out_p = PPON(lr)
            out_c, out_s, out_p = out_c.cpu(), out_s.cpu(), out_p.cpu()

            out_img_c = out_c.detach().numpy()#.squeeze()

            out_img_s = out_s.detach().numpy().squeeze()
            out_img_s = convert_shape(out_img_s)

            out_img_p = out_p.detach().numpy().squeeze()
            out_img_p = convert_shape(out_img_p)
            out_img_c = torch.from_numpy(out_img_c)
        
        sr_image = torch.clip(out_img_c, 0.0, 1.0)
        psnr_sr[i] = PSNR(hr, sr_image)

        sr_image_np = sr_image.detach().numpy()  # Convert tensor to numpy array
        print(f"HR shape: {hr.shape}")
        print(f"SR shape: {sr_image.shape}")
        ssim_sr[i] = compute_ssim(quantize(hr.detach().numpy()),quantize(sr_image_np))
        sr_image = denorm01(out_img_c.squeeze())
        sr_image = sr_image.type(torch.uint8)
        sr_image = ycbcr2rgb(sr_image)
        # write_image("sr.png", sr_image)

    print('Mean PSNR for SR: {}'.format(np.mean(psnr_sr)))
    print('Mean SSIM for SR: {}'.format(np.mean(ssim_sr)))

if __name__ == '__main__':
    main()
