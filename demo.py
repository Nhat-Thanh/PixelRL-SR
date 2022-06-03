import sys
sys.dont_write_bytecode = True

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
parser.add_argument("--scale",           type=int, default=2,                   help='-')
parser.add_argument("--image-path",      type=str, default="dataset/test2.png", help='-')
parser.add_argument("--model-path",      type=str, default="",                  help='-')
parser.add_argument("--draw-action-map", type=int, default=0,                   help='-')
FLAG, unparsed = parser.parse_known_args()


# =====================================================================================
# Global variables
# =====================================================================================

SCALE = FLAG.scale
if SCALE not in [2, 3, 4]:
    raise ValueError("--scale must be 2, 3 or 4")

IMAGE_PATH = FLAG.image_path

MODEL_PATH = FLAG.model_path
if (MODEL_PATH == "") or (MODEL_PATH == "default"):
    MODEL_PATH = f"checkpoint/x{SCALE}/PixelRL_SR-x{SCALE}.pt"

DRAW_ACTION_MAP = (FLAG.draw_action_map == 1)
ACTION_MAP_DIR = "action_maps"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_ACTIONS = 7
GAMMA = 0.95
T_MAX = 5

COLOR_TABLE = {
    0: torch.tensor([0, 0, 0],       dtype=torch.uint8), # black
    1: torch.tensor([255, 255, 255], dtype=torch.uint8), # white
    2: torch.tensor([255, 0, 0],     dtype=torch.uint8), # red
    3: torch.tensor([0, 255, 0],     dtype=torch.uint8), # lime
    4: torch.tensor([0, 0, 255],     dtype=torch.uint8), # blue
    5: torch.tensor([255, 255, 0],   dtype=torch.uint8), # yellow
    6: torch.tensor([0, 255, 255],   dtype=torch.uint8), # cyan / aqua
    7: torch.tensor([255, 0, 255],   dtype=torch.uint8), # magenta / fuchsia
    8: torch.tensor([128, 128, 128], dtype=torch.uint8)  # gray
}


# =====================================================================================
# Test
# =====================================================================================

def main():

    if DRAW_ACTION_MAP:
        os.makedirs(ACTION_MAP_DIR, exist_ok=True)
    CURRENT_STATE = State(SCALE, 'cpu')

    MODEL = PixelRL_model(N_ACTIONS).to(DEVICE)
    if exists(MODEL_PATH):
        MODEL.load_state_dict(torch.load(MODEL_PATH, torch.device(DEVICE)))
    MODEL.eval()

    lr_image = read_image(IMAGE_PATH)
    lr_image = gaussian_blur(lr_image, sigma=0.2)
    bicubic = upscale(lr_image, SCALE)
    write_image("bicubic.png", bicubic)

    bicubic = rgb2ycbcr(bicubic)
    lr_image = rgb2ycbcr(lr_image)

    bicubic = norm01(bicubic).unsqueeze(0)
    lr_image = norm01(lr_image).unsqueeze(0)

    CURRENT_STATE.reset(lr_image, bicubic)
    with torch.no_grad():
        for i in range(0, T_MAX):
            state_var = CURRENT_STATE.tensor.to(DEVICE)
            actions, _, inner_state = MODEL.choose_best_actions(state_var)
            CURRENT_STATE.step(actions, inner_state)

            if DRAW_ACTION_MAP:
                actions = tensor2numpy(actions[0])
                action_map = draw_action_map(actions, COLOR_TABLE)
                write_image(os.path.join(ACTION_MAP_DIR, f"step_{i}.png"), action_map)

    sr_image = torch.clip(CURRENT_STATE.sr_image[0], 0.0, 1.0)
    sr_image = denorm01(sr_image)
    sr_image = sr_image.type(torch.uint8)
    sr_image = ycbcr2rgb(sr_image)
    write_image("sr.png", sr_image)

if __name__ == '__main__':
    main()
