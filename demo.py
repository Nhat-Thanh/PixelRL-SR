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
parser.add_argument("--scale",      type=int, default=2,                                help='-')
parser.add_argument("--image-path", type=str, default="dataset/test2.png",              help='-')
parser.add_argument("--model-path", type=str, default="checkpoint/x2/PixelRL_SR-x2.pt", help='-')
FLAG, unparsed = parser.parse_known_args()


# =====================================================================================
# Global variables
# =====================================================================================

SCALE = FLAG.scale
MODEL_PATH = FLAG.model_path
IMAGE_PATH = FLAG.image_path

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_ACTIONS = 7
GAMMA = 0.95
T_MAX = 5


# =====================================================================================
# Test image
# =====================================================================================

def main():
    CURRENT_STATE = State(SCALE, 'cpu')

    MODEL = PixelRL_model(N_ACTIONS).to(DEVICE)
    if exists(MODEL_PATH):
        MODEL.load_state_dict(torch.load(MODEL_PATH, torch.device(DEVICE)))
    MODEL.eval()

    lr = read_image(IMAGE_PATH)
    lr = gaussian_blur(lr, sigma=0.2)
    bicubic = upscale(lr, SCALE)
    write_image("bicubic.png", bicubic)

    bicubic = rgb2ycbcr(bicubic)
    lr = rgb2ycbcr(lr)

    bicubic = norm01(bicubic).unsqueeze(0)
    lr = norm01(lr).unsqueeze(0)

    CURRENT_STATE.reset(lr, bicubic)
    with torch.no_grad():
        for _ in range(0, T_MAX):
            pi, _, inner_state = MODEL.pi_and_v(CURRENT_STATE.tensor.to(DEVICE))

            actions_prob = torch.softmax(pi, dim=1).cpu()
            actions = torch.argmax(actions_prob, dim=1)
            inner_state = inner_state.cpu()

            CURRENT_STATE.step(actions, inner_state)

    sr = torch.clip(CURRENT_STATE.sr_images[0], 0.0, 1.0)
    sr = denorm01(sr)
    sr = sr.type(torch.uint8)
    sr = ycbcr2rgb(sr)
    write_image("sr.png", sr)

if __name__ == '__main__':
    main()
