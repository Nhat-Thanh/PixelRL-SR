import torch
from neuralnet import ESPCN_model, FSRCNN_model, SRCNN_model, VDSR_model
import torch.nn as nn
from PPON.PPON_model import PPONModel
from PPON import networks
from utils.common import exist_value, to_cpu, convert_shape, pad_image_to_factor_of_16
import json
from hat.archs.hat_arch import HAT

# declaringa a class
class obj:
     
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
class State:
    def __init__(self, scale, device):
        self.device = device
        self.lr_image = None
        self.sr_image = None
        self.tensor = None
        self.move_range = 3

        dev = torch.device(device)

        
        self.HAT_model = HAT(upscale=4,
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
            resi_connection='1conv').to(device)
        model_path = "sr_weight/HAT_SRx4.pth"
        self.HAT_model.load_state_dict(torch.load(model_path)['params_ema'])
        self.HAT_model.eval()
        

    def reset(self, lr, bicubic):
        self.lr_image = lr 
        self.sr_image = bicubic
        b, _, h, w = self.sr_image.shape
        previous_state = torch.zeros(size=(b, 64, h, w), dtype=self.lr_image.dtype)
        self.tensor = torch.concat([self.sr_image, previous_state], dim=1)

    def set(self, lr, bicubic):
        self.lr_image = lr
        self.sr_image = bicubic
        self.tensor[:,0:3,:,:] = self.sr_image

    def step(self, act, inner_state):
        act = to_cpu(act)
        inner_state = to_cpu(inner_state)
        hat = self.sr_image.clone()

        neutral = (self.move_range - 1) / 2
        move = act.type(torch.float32)
        move = (move - neutral) / 255
        moved_image = self.sr_image.clone()
        for i in range(0, self.sr_image.shape[1]):
            moved_image[:,i] += move[0]

        self.lr_image = self.lr_image.to(self.device)
        self.sr_image = self.sr_image.to(self.device)

        with torch.no_grad():
            # if exist_value(act, 3):
            #     # change ESPCN to PPON
            #     # espcn = to_cpu(self.ESPCN(self.lr_image))
            #     self.PPON.cuda()
            #     self.lr_image.cuda()
            #     # print(self.lr_image.shape)
            #     with torch.no_grad():
            #         out_c, out_s, out_p = self.PPON(self.lr_image)
            #         out_c, out_s, out_p = out_c.cpu(), out_s.cpu(), out_p.cpu()
            #         out_img_c = out_c.detach().numpy().squeeze()
            #         # out_img_c = convert_shape(out_img_c)

            #         out_img_s = out_s.detach().numpy()
            #         # out_img_s = convert_shape(out_img_s)

            #         out_img_p = out_p.detach().numpy()
            #        # out_img_p = convert_shape(out_img_p)
            #     # print(out_img_c.shape)
            #     ppon = torch.from_numpy(out_img_c)
            if exist_value(act, 3):
                # print(self.lr_image.shape)
                hat = self.HAT_model(self.lr_image.float())
                hat = to_cpu(hat.int())

        self.lr_image = to_cpu(self.lr_image)
        self.sr_image = moved_image
        act = act.unsqueeze(1)
        act = torch.concat([act, act, act], 1)
  
        self.sr_image = torch.where(act==3, hat,  self.sr_image)

        self.tensor[:,0:3,:,:] = self.sr_image
        self.tensor[:,-64:,:,:] = inner_state
