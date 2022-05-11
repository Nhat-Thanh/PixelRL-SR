import torch
from neuralnet import ESPCN_model, FSRCNN_model, SRCNN_model, VDSR_model
from utils.common import exist_value, to_cpu

class State:
    def __init__(self, scale, device):
        self.device = device
        self.lr_images = None
        self.sr_images = None
        self.tensor = None
        self.move_range = 3

        dev = torch.device(device)
        self.SRCNN = SRCNN_model().to(device)
        model_path = "sr_weight/SRCNN-955.pt"
        self.SRCNN.load_state_dict(torch.load(model_path, dev))
        self.SRCNN.eval()

        self.FSRCNN = FSRCNN_model(scale).to(device)
        model_path = f"sr_weight/x{scale}/FSRCNN-x{scale}.pt"
        self.FSRCNN.load_state_dict(torch.load(model_path, dev))
        self.FSRCNN.eval()

        self.ESPCN = ESPCN_model(scale).to(device)
        model_path = f"sr_weight/x{scale}/ESPCN-x{scale}.pt"
        self.ESPCN.load_state_dict(torch.load(model_path, dev))
        self.ESPCN.eval()

        self.VDSR = VDSR_model().to(device)
        model_path = f"sr_weight/VDSR.pt"
        self.VDSR.load_state_dict(torch.load(model_path, dev))
        self.VDSR.eval()

    def reset(self, lr, bicubic):
        self.lr_images = lr 
        self.sr_images = bicubic
        b, _, h, w = self.sr_images.shape
        previous_state = torch.zeros(size=(b, 64, h, w), dtype=self.lr_images.dtype)
        self.tensor = torch.concat([self.sr_images, previous_state], dim=1)

    def set(self, lr, bicubic):
        self.lr_images = lr
        self.sr_images = bicubic
        self.tensor[:,0:3,:,:] = self.sr_images

    def step(self, act, inner_state):
        act = to_cpu(act)
        inner_state = to_cpu(inner_state)
        srcnn = self.sr_images.clone()
        espcn = self.sr_images.clone()
        fsrcnn = self.sr_images.clone()
        vdsr = self.sr_images.clone()

        neutral = (self.move_range - 1) / 2
        move = act.type(torch.float32)
        move = (move - neutral) / 255
        moved_image = self.sr_images.clone()
        for i in range(0, self.sr_images.shape[1]):
            moved_image[:,i] += move[0]

        self.lr_images = self.lr_images.to(self.device)
        self.sr_images = self.sr_images.to(self.device)

        with torch.no_grad():
            if exist_value(act, 3):
                espcn = to_cpu(self.ESPCN(self.lr_images))
            if exist_value(act, 4):
                srcnn[:, :, 8:-8, 8:-8] = to_cpu(self.SRCNN(self.sr_images))
            if exist_value(act, 5):
                vdsr = to_cpu(self.VDSR(self.sr_images))
            if exist_value(act, 6):
                fsrcnn = to_cpu(self.FSRCNN(self.lr_images))

        self.lr_images = to_cpu(self.lr_images)
        self.sr_images = moved_image
        act = act.unsqueeze(1)
        act = torch.concat([act, act, act], 1)
        self.sr_images = torch.where(act==3, espcn,  self.sr_images)
        self.sr_images = torch.where(act==4, srcnn,  self.sr_images)
        self.sr_images = torch.where(act==5, vdsr,   self.sr_images)
        self.sr_images = torch.where(act==6, fsrcnn, self.sr_images)

        self.tensor[:,0:3,:,:] = self.sr_images
        self.tensor[:,-64:,:,:] = inner_state
