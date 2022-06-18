import copy
import numpy as np
import os
from State import State
import torch
from torch.distributions import Categorical
from utils.common import exists, tensor2numpy

torch.manual_seed(1)

class logger:
    def __init__(self, path, values) -> None:
        self.path = path
        self.values = values

def MyEntropy(log_actions_prob, actions_prob):
    entropy = torch.stack([- torch.sum(log_actions_prob * actions_prob, dim=1)])
    return entropy.permute(([1, 0, 2, 3]))

def MyLogProb(log_actions_prob, actions):
    selected_pi = log_actions_prob.gather(1, actions.unsqueeze(1))
    return selected_pi

class PixelWiseA3C_InnerState_ConvR:
    def __init__(self, model, t_max, gamma, beta=1e-2,
                 pi_loss_coef=1.0, v_loss_coef=0.5):
        self.shared_model = model 
        self.model = copy.deepcopy(self.shared_model)
        self.ckpt_man = None
        self.t_max = t_max
        self.gamma = gamma
        self.beta = beta
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        self.past_log_prob= {}
        self.past_entropy= {}
        self.past_rewards = {}
        self.past_values = {}

    def setup(self, scale, optimizer, batch_size,
              metric, device, model_path, ckpt_dir):
        self.batch_size = batch_size
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.metric = metric
        self.model_path = model_path
        self.optimizer = optimizer
        self.scale = scale

    # https://github.com/FightingSrain/Pytorch-pixelRL/blob/main/pixelwise_a3c.py#L58
    def sync_parameters(self):
        for md_1, md_2 in zip(self.model.modules(), self.shared_model.modules()):
            md_1._buffers = md_2._buffers.copy()
        for target, src in zip(self.model.parameters(), self.shared_model.parameters()):
            target.detach().copy_(src.detach())

    # https://github.com/FightingSrain/Pytorch-pixelRL/blob/main/pixelwise_a3c.py#L66
    def copy_grad(self, src, target):
        target_params = dict(target.named_parameters())
        for name, param in src.named_parameters():
            if target_params[name].grad is None:
                if param.grad is None:
                    continue
                target_params[name].grad = param.grad
            else:
                if param.grad is None:
                    target_params[name].grad = None
                else:
                    target_params[name].grad[...] = param.grad

    def load_checkpoint(self, ckpt_path):
        if exists(ckpt_path):
            self.ckpt_man = torch.load(ckpt_path)
            self.optimizer.load_state_dict(self.ckpt_man['optimizer'])
            self.shared_model.load_state_dict(self.ckpt_man['shared_model'])
            self.model.load_state_dict(self.ckpt_man['model'])

    def load_weights(self, filepath):
        if exists(filepath):
            ckpt = torch.load(filepath, torch.device(self.device))
            self.model.load_state_dict(ckpt)

    def evaluate(self, dataset, batch_size):
        self.model.train(False)
        self.shared_model.train(False)
        current_state = State(self.scale, self.device)
        rewards, metrics = [], []
        with torch.no_grad():
            isEnd = False
            while isEnd == False:
                bicubic, lr, hr, isEnd = dataset.get_batch(batch_size, shuffle_each_epoch=False)
                current_state.reset(lr, bicubic)
                sum_reward = 0
                for t in range(0, self.t_max):
                    prev_image = current_state.sr_image.clone()
                    state_var = current_state.tensor.to(self.device)
                    actions, _, inner_state = self.model.choose_best_actions(state_var)
                    current_state.step(actions, inner_state)

                    # calculate reward on Y chanel only
                    reward = torch.square(hr[:,0:1] - prev_image[:,0:1]) - \
                             torch.square(hr[:,0:1] - current_state.sr_image[:,0:1])
                    sum_reward += torch.mean(reward * 255) * (self.gamma ** t)

                rewards.append(sum_reward)
                metric = self.metric(hr, current_state.sr_image)
                metrics.append(metric)

        total_reward = torch.mean(torch.tensor(rewards))
        total_metric = torch.mean(torch.tensor(metrics))
        return total_reward, total_metric

    def train(self, train_set, valid_set, batch_size, steps, save_every=1, save_log=False):
        cur_step = 0
        if self.ckpt_man is not None:
            cur_step = self.ckpt_man['step']
        max_step = cur_step + steps
        ckpt_path = os.path.join(self.ckpt_dir, f"ckpt-x{self.scale}.pt")

        dict_logger = { "train_loss":   logger(path=os.path.join(self.ckpt_dir, "train_losses.npy"),  values=[]),
                        "train_reward": logger(path=os.path.join(self.ckpt_dir, "train_rewards.npy"), values=[]),
                        "train_metric": logger(path=os.path.join(self.ckpt_dir, "train_metrics.npy"), values=[]),
                        "val_reward":   logger(path=os.path.join(self.ckpt_dir, "val_rewards.npy"),   values=[]),
                        "val_metric":   logger(path=os.path.join(self.ckpt_dir, "val_metrics.npy"),   values=[]) }

        for key in dict_logger.keys():
            if exists(dict_logger[key].path):
                dict_logger[key].values = np.load(dict_logger[key].path).tolist()

        self.current_state = State(self.scale, self.device)
        train_loss_buffer = []
        train_reward_buffer = []
        train_metric_buffer = []
        while cur_step < max_step:
            cur_step += 1
            bicubic, lr, hr, _ = train_set.get_batch(batch_size)
            train_reward, train_loss, train_metric = self.train_step(bicubic, lr, hr)
            train_reward_buffer.append(train_reward)
            train_loss_buffer.append(train_loss)
            train_metric_buffer.append(train_metric)

            if cur_step % save_every == 0:
                train_reward = torch.mean(torch.tensor(train_reward_buffer)) * 255
                train_loss = torch.mean(torch.tensor(train_loss_buffer))
                train_metric = torch.mean(torch.tensor(train_metric_buffer))
                val_reward, val_metric = self.evaluate(valid_set, batch_size)
                val_reward = val_reward * 255

                print(f"Step {cur_step}  / {max_step}",
                        f"- loss: {train_loss:.6f}",
                        f"- reward: {train_reward:.6f}",
                        f"- {self.metric.__name__}: {train_metric:.4f}",
                        f"- val_reward: {val_reward:.6f}",
                        f"- val_{self.metric.__name__}: {val_metric:.4f}")

                print(f"Save model weights to {self.model_path}")
                torch.save(self.model.state_dict(), self.model_path)

                print(f"Save checkpoint to {ckpt_path}\n")
                torch.save({ 'step': cur_step,
                             'model': self.model.state_dict(),
                             'shared_model': self.shared_model.state_dict(),
                             'optimizer': self.optimizer.state_dict() }, ckpt_path)
                
                train_loss_buffer = []
                train_reward_buffer = []
                train_metric_buffer = []
                
                if save_log == False:
                    continue
                dict_logger["train_loss"].values.append(train_loss)
                dict_logger["train_reward"].values.append(train_reward)
                dict_logger["train_metric"].values.append(train_metric)
                dict_logger["val_reward"].values.append(val_reward)
                dict_logger["val_metric"].values.append(val_metric)

        if save_log == False:
            return
        for key in dict_logger.keys():
            values = torch.tensor(dict_logger[key].values)
            values = tensor2numpy(values)
            np.save(dict_logger[key].path, values)

    def train_step(self, bicubic, lr, hr):
        self.model.train(True)
        self.shared_model.train(True)

        self.current_state.reset(lr, bicubic)
        total_reward = 0.0
        reward = 0.0
        for t in range(0, self.t_max):
            prev_images = self.current_state.sr_image.clone()
            state_var = self.current_state.tensor.to(self.device)
            pi, v, inner_state = self.model.pi_and_v(state_var)

            actions_prob = torch.softmax(pi, dim=1)
            log_actions_prob = torch.log_softmax(pi, dim=1)
            prob_trans = actions_prob.permute([0, 2, 3, 1])
            actions = Categorical(prob_trans).sample()
            self.current_state.step(actions, inner_state)
            
            # calculate reward on Y chanel only
            reward = (torch.square(hr[:,0:1] - prev_images[:,0:1]) - \
                      torch.square(hr[:,0:1] - self.current_state.sr_image[:,0:1])) * 255
            self.past_rewards[t] = reward.to(self.device)
            self.past_log_prob[t] = MyLogProb(log_actions_prob, actions)
            self.past_entropy[t] = MyEntropy(log_actions_prob, actions_prob)
            self.past_values[t] = v
            total_reward += torch.mean(reward) * (self.gamma ** t)

        total_metric = self.metric(hr, self.current_state.sr_image)

        pi_loss = 0.0
        v_loss = 0.0
        R = torch.zeros_like(v).to(self.device)
        for k in reversed(range(0, self.t_max)):
            R = self.model.conv_smooth(R * self.gamma) + self.past_rewards[k]
            Advantage = R - self.past_values[k]
            pi_loss -= self.past_log_prob[k] * Advantage
            pi_loss -= self.beta * self.past_entropy[k]
            v_loss += torch.square(Advantage) / 2

        if self.pi_loss_coef != 1.0:
            pi_loss *= self.pi_loss_coef

        if self.v_loss_coef != 1.0:
            v_loss *= self.v_loss_coef

        total_loss = torch.nanmean(pi_loss + v_loss)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.copy_grad(src=self.model, target=self.shared_model)
        self.sync_parameters()

        return total_reward, total_loss, total_metric
