import argparse
from neuralnet import PixelRL_model 
from State import State
import torch
from utils.common import *

torch.manual_seed(1)

# =====================================================================================
# arguments parser
# =====================================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--scale", type=int, default=2, help='-')
FLAG, unparsed = parser.parse_known_args()


# =====================================================================================
# Global variables
# =====================================================================================

SCALE = FLAG.scale
MODEL_PATH = f"checkpoint/x{SCALE}/PixelRL_SR-x{SCALE}.pt"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_ACTIONS = 7
GAMMA = 0.95
T_MAX = 5

LS_HR_PATHS = sorted_list(f"dataset/test/x{SCALE}/labels")
LS_LR_PATHS = sorted_list(f"dataset/test/x{SCALE}/data")


# =====================================================================================
# Test each image
# =====================================================================================

def main():

    CURRENT_STATE = State(SCALE, DEVICE)

    MODEL = PixelRL_model(N_ACTIONS).to(DEVICE)
    if exists(MODEL_PATH):
        MODEL.load_state_dict(torch.load(MODEL_PATH, torch.device(DEVICE)))
    MODEL.train(False)

    rewards = []
    metrics = []
    for i in range(0, len(LS_HR_PATHS)):
        hr_image_path = LS_HR_PATHS[i]
        lr_image_path = LS_LR_PATHS[i]
        hr = read_image(hr_image_path)
        lr = read_image(lr_image_path)
        lr = gaussian_blur(lr, sigma=0.3)
        bicubic = upscale(lr, SCALE)

        bicubic = rgb2ycbcr(bicubic)
        lr = rgb2ycbcr(lr)
        hr= rgb2ycbcr(hr)


        bicubic = norm01(bicubic).unsqueeze(0)
        lr = norm01(lr).unsqueeze(0)
        hr = norm01(hr).unsqueeze(0)

        with torch.no_grad():
            CURRENT_STATE.reset(lr, bicubic)

            sum_reward = 0
            for t in range(0, T_MAX):
                prev_img = CURRENT_STATE.sr_images.clone()
                statevar = CURRENT_STATE.tensor.to(DEVICE)
                pi, _, inner_state = MODEL.pi_and_v(statevar)

                actions_prob = torch.softmax(pi, dim=1).cpu()
                actions = torch.argmax(actions_prob, dim=1)
                inner_state = inner_state.cpu()

                CURRENT_STATE.step(actions, inner_state)
                # Calculate reward on Y chanel only
                reward = (torch.square(hr[:,0:1] - prev_img[:,0:1]) - \
                          torch.square(hr[:,0:1] - CURRENT_STATE.sr_images[:,0:1])) * 255

                sum_reward += torch.mean(reward) * (GAMMA ** t)

            sr = torch.clip(CURRENT_STATE.sr_images, 0.0, 1.0)
            psnr = PSNR(hr, sr)
            sum_reward *= 255 
            metrics.append(psnr)
            rewards.append(sum_reward)

    print(f"Average reward: {torch.mean(torch.tensor(metrics)):.4f}",
          f"- PSNR: {torch.mean(torch.tensor(metrics)):.4f}")

if __name__ == '__main__':
    main()
