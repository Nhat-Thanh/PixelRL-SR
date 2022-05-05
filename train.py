from neuralnet import PixelRL_model
from model import PixelWiseA3C_InnerState_ConvR 
from utils.dataset import dataset
from utils.common import PSNR, exists
import argparse
import torch
import os

torch.manual_seed(1)

# =====================================================================================
# arguments parser
# =====================================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--scale",      type=int, default=2,               help='-')
parser.add_argument("--episodes",   type=int, default=5000,            help='-')
parser.add_argument("--batch-size", type=int, default=64,              help='-')
parser.add_argument("--save-every", type=int, default=500,             help='-')
parser.add_argument("--ckpt-dir",   type=str, default="checkpoint/x2", help='-')
FLAG, unparsed = parser.parse_known_args()

# =====================================================================================
# Global variables
# =====================================================================================

# training settings
SCALE = FLAG.scale
BATCH_SIZE = FLAG.batch_size
CKPT_DIR = FLAG.ckpt_dir
CKPT_PATH = os.path.join(CKPT_DIR, f"ckpt-x{SCALE}.pt")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPISODES = FLAG.episodes
MODEL_PATH = os.path.join(CKPT_DIR, f"PixelRL_SR-x{SCALE}.pt")
# PRETRAINED_PATH = f"initial_weight/denoise_{SIGMA}_gray_ConvGRU_RMC.pt"
SAVE_EVERY = FLAG.save_every

# model settings
N_ACTIONS = 7
LEARNING_RATE = 1e-3

# A3C settings
GAMMA = 0.95
T_MAX = 5
BETA = 1e-2

# Dataset settings
HR_CROP_SIZE = 60
LR_CROP_SIZE = HR_CROP_SIZE // SCALE
DATASET_DIR = "dataset"


# =====================================================================================
# Train
# =====================================================================================

def main():
    train_set = dataset(DATASET_DIR, "train")
    train_set.generate(LR_CROP_SIZE, HR_CROP_SIZE)
    train_set.load_data(shuffle_arrays=True)

    valid_set = dataset(DATASET_DIR, "validation")
    valid_set.generate(LR_CROP_SIZE, HR_CROP_SIZE)
    valid_set.load_data(shuffle_arrays=False)

    MODEL = PixelRL_model(N_ACTIONS).to(DEVICE)
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), LEARNING_RATE)
    pixelRL = PixelWiseA3C_InnerState_ConvR(MODEL, T_MAX, GAMMA, BETA)
    pixelRL.setup(SCALE, OPTIMIZER, LEARNING_RATE, BATCH_SIZE, PSNR,  DEVICE, MODEL_PATH, CKPT_DIR)

    pixelRL.load_checkpoint(CKPT_PATH)
    # if not exists(CKPT_PATH):
    #     print(f"Load pre-trained model at {PRETRAINED_PATH}")
    #     pixelRL.load_weights(PRETRAINED_PATH)

    pixelRL.train(train_set, valid_set, BATCH_SIZE, EPISODES, SAVE_EVERY)

if __name__ == '__main__':
    main()
