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
    ValueError("--scale must be 2, 3 or 4")

IMAGE_PATH = FLAG.image_path

MODEL_PATH = FLAG.model_path
if MODEL_PATH == "":
    MODEL_PATH = f"checkpoint/x{SCALE}/PixelRL_SR-x{SCALE}.pt"

DRAW_ACTION_MAP = FLAG.draw_action_map
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
        for i in range(0, T_MAX):
            pi, _, inner_state = MODEL.pi_and_v(CURRENT_STATE.tensor.to(DEVICE))

            actions_prob = torch.softmax(pi, dim=1).cpu()
            actions = torch.argmax(actions_prob, dim=1)
            inner_state = inner_state.cpu()

            CURRENT_STATE.step(actions, inner_state)

            if DRAW_ACTION_MAP:
                actions = actions[0].cpu().numpy()
                action_map = draw_action_map(actions, COLOR_TABLE)
                write_image(os.path.join(ACTION_MAP_DIR, f"step_{i}.png"), action_map)

    sr = torch.clip(CURRENT_STATE.sr_images[0], 0.0, 1.0)
    sr = denorm01(sr)
    sr = sr.type(torch.uint8)
    sr = ycbcr2rgb(sr)
    write_image("sr.png", sr)

if __name__ == '__main__':
    main()
