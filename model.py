import copy
from State import State
import os
import torch
from torch.distributions import Categorical
from utils.common import exists

torch.manual_seed(1)

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

    def setup(self, scale, optimizer, init_lr, batch_size,
              metric, device, model_path, ckpt_dir):
        self.batch_size = batch_size
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.initial_lr = init_lr
        self.metric = metric
        self.model_path = model_path
        self.optimizer = optimizer
        self.scale = scale

    def sync_parameters(self):
        for md_1, md_2 in zip(self.model.modules(), self.shared_model.modules()):
            md_1._buffers = md_2._buffers.copy()

        for target, src in zip(self.model.parameters(), self.shared_model.parameters()):
            target.detach().copy_(src.detach())

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
        rewards = []
        metrics = []
        isEnd = False
        with torch.no_grad():
            while isEnd == False:
                bicubic, lr, hr, isEnd = dataset.get_batch(batch_size)
                current_state.reset(lr, bicubic)
                sum_reward = 0
                for t in range(0, self.t_max):
                    prev_image = current_state.sr_images.clone()
                    lr = current_state.tensor.to(self.device)
                    pi, _, inner_state = self.model.pi_and_v(lr)
    
                    actions_prob = torch.softmax(pi, dim=1)
                    actions = torch.argmax(actions_prob, dim=1)
                    current_state.step(actions.cpu(), inner_state.cpu())
    
                    # calculate reward on Y chanel only
                    reward = torch.square(hr[:,0:1] - prev_image[:,0:1]) - \
                             torch.square(hr[:,0:1] - current_state.sr_images[:,0:1])
                    sum_reward += torch.mean(reward * 255) * (self.gamma ** t)
    
                rewards.append(sum_reward)
                metric = self.metric(hr, current_state.sr_images)
                metrics.append(metric)
    
        total_reward = torch.mean(torch.tensor(rewards))
        total_metric = torch.mean(torch.tensor(metrics))
        return total_reward, total_metric

    def train(self, train_set, valid_set, batch_size, episodes, save_every):
        cur_episode = 0
        if self.ckpt_man is not None:
            cur_episode = self.ckpt_man['episode']
        max_episode = cur_episode + episodes
        ckpt_path = os.path.join(self.ckpt_dir, f"ckpt-x{self.scale}.pt")

        self.current_state = State(self.scale, self.device)
        while cur_episode < max_episode:
            cur_episode += 1
            bicubic, lr, hr, _ = train_set.get_batch(batch_size)
            total_reward, loss = self.train_step(bicubic, lr, hr)

            print(f"Epoch {cur_episode} / {max_episode} - loss: {loss:.6f} - sum reward: {total_reward * 255:.6f}")

            if cur_episode % save_every == 0:
                reward, metric = self.evaluate(valid_set, batch_size)
                print(f"\nEvaluate - reward: {reward * 255:.6f} - {self.metric.__name__}: {metric:.6f}")

                print(f"Save model weights to {self.model_path}")
                torch.save(self.model.state_dict(), self.model_path)

                print(f"Save checkpoint to {ckpt_path}\n")
                torch.save({ 'episode': cur_episode,
                             'model': self.model.state_dict(),
                             'shared_model': self.shared_model.state_dict(),
                             'optimizer': self.optimizer.state_dict() }, ckpt_path)

            # self.optimizer.param_groups[0]['lr'] = self.initial_lr - ((1 - cur_episode / max_episode) ** 0.9)

    def train_step(self, bicubic, lr, hr):
        self.model.train(True)
        self.shared_model.train(True)

        self.current_state.reset(lr, bicubic)
        total_reward = 0.0
        reward = 0.0
        for t in range(0, self.t_max):
            prev_images = self.current_state.sr_images.clone()
            lr = self.current_state.tensor.to(self.device)
            pi, v, inner_state = self.model.pi_and_v(lr)

            actions_prob = torch.softmax(pi, dim=1)
            log_actions_prob = torch.log_softmax(pi, dim=1)
            prob_trans = actions_prob.permute([0, 2, 3, 1])
            actions = Categorical(prob_trans).sample().detach()
            self.current_state.step(actions, inner_state)

            reward = (torch.square(hr[:,0:1] - prev_images[:,0:1]) - \
                      torch.square(hr[:,0:1] - self.current_state.sr_images[:,0:1])) * 255

            self.past_rewards[t] = reward.to(self.device)
            self.past_log_prob[t] = MyLogProb(log_actions_prob, actions)
            self.past_entropy[t] = MyEntropy(log_actions_prob, actions_prob)
            self.past_values[t] = v
            total_reward += torch.mean(reward) * (self.gamma ** t)

        pi_loss = 0.0
        v_loss = 0.0
        # R = 0 in author's source code
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

        # reset history
        self.past_log_prob = {}
        self.past_entropy = {}
        self.past_values = {}
        self.past_rewards = {}

        return total_reward, total_loss
