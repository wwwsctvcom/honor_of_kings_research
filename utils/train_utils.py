import torch
from typing import *
from pathlib import Path
from utils.tools import get_now_time
from torch import nn
from torch.utils.data import DataLoader
from utils.tools import get_lr
from utils.general import *
from model.image_embedding import ResNetEmbeddings
from tqdm import tqdm
from accelerate import Accelerator


# class StateModelTrainer:
#     """
#     model: state model
#     scheduler: learning rate scheduler
#     accelerator:
#     processor: resnet101 backend preprocess model, preprocess the input image
#     """
#
#     def __init__(self,
#                  args=None,
#                  model=None,
#                  scheduler=None,
#                  optimizer=None,
#                  accelerator=None,
#                  processor: ResNetEmbeddings = None,
#                  device: str = None):
#         self.device = device
#         self.args = args
#         self.model = model.to(self.device)
#         self.scheduler = scheduler
#         self.optimizer = optimizer
#         self.accelerator = accelerator if accelerator is not None else Accelerator()
#         self.processor = processor.to(self.device)
#         self.loss_fct = nn.CrossEntropyLoss()
#
#     def train(self, train_data_loader: DataLoader = None):
#         loss_min = float('inf')
#         for epoch in range(1, self.args.epochs + 1):
#             train_total_loss = 0
#             self.model.train()
#             with tqdm(enumerate(train_data_loader), total=len(train_data_loader),
#                       desc=f'Epoch: {epoch}/{self.args.epochs}', postfix=dict()) as train_pbar:
#                 for step, batch in train_pbar:
#                     images, labels = batch
#
#                     # do preprocess
#                     images = self.processor(images)  # [b, w, h, c]
#                     batch_size = images.shape[0]
#                     images = images.reshape(batch_size, -1).unsqueeze(1)  # [b, 1, w * h * c], support for old model
#
#                     # backward, calculate gradient
#                     logits, _ = self.model(images, labels, None)
#                     loss = self.loss_fct(logits.view(-1, self.args.num_states), labels.view(-1))
#
#                     loss.backward()
#                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
#
#                     self.optimizer.step()
#                     self.optimizer.zero_grad()
#
#                     # lr scheduler
#                     if self.scheduler is not None:
#                         self.scheduler.step()
#
#                     train_total_loss += loss.item()
#
#                     train_pbar.set_postfix(
#                         **{'train average loss': train_total_loss / (step + 1), 'train loss': loss.item(),
#                            'lr': get_lr(self.optimizer)})
#
#             if loss.item() < loss_min:
#                 self.save(Path(self.args.output_dir) / (get_now_time() + f"_state_weights_{epoch}_{loss.item()}.pth"))
#
#     def save(self, model_name: Union[Path, str]):
#         torch.save(self.model.state_dict(), 'model_state.pth')


class MainModelTrainer:
    """
    model: main model, including state model, unsupervised model
    scheduler: learning rate scheduler
    accelerator:
    processor: resnet101 backend preprocess model, preprocess the input image
    """

    def __init__(self,
                 args=None,
                 model=None,
                 scheduler=None,
                 optimizer=None,
                 accelerator=None,
                 # processor: ResNetEmbeddings = None,
                 normalize_reward: bool = None,
                 device: str = None):
        self.device = device
        self.args = args
        self.model = model.to(self.device)
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.accelerator = accelerator if accelerator is not None else Accelerator()
        # self.processor = processor.to(self.device)
        self.normalize_reward = normalize_reward
        self.loss_action = nn.CrossEntropyLoss()
        self.loss_critic = nn.MSELoss()
        self.reward_weights = torch.tensor([2, 5, -0.5, -2, 0.01]).to(self.device)

    def train(self, train_data_loader: DataLoader = None):
        loss_min = float('inf')
        for epoch in range(1, self.args.epochs + 1):
            train_total_loss = 0
            self.model.train()
            with tqdm(enumerate(train_data_loader), total=len(train_data_loader),
                      desc=f'Epoch: {epoch}/{self.args.epochs}', postfix=dict()) as train_pbar:
                for step, batch in train_pbar:
                    images, action_labels, reward_labels = batch
                    images = images.to(self.device)
                    action_labels = action_labels.to(self.device)
                    reward_labels = reward_labels.to(self.device)

                    # # do preprocess
                    # images = self.processor(images)  # [b, w, h, c]
                    # batch_size = images.shape[0]
                    # images = images.reshape(batch_size, -1).unsqueeze(1)  # [b, 1, w * h * c], for support old model

                    # backward, calculate gradient
                    logits_action, logits_reward = self.model(images)

                    # calculate the rewards by model logits
                    reward_index = torch.argmax(torch.softmax(logits_reward, dim=-1), dim=-1).to(self.device)
                    reward_weights = self.reward_weights.clone().detach()
                    rewards = reward_weights[reward_index]
                    normalized_rewards = (rewards - rewards.mean()) / rewards.std()

                    loss_action = self.loss_action(logits_action, action_labels)
                    if logits_reward.shape[-1] == 1:
                        loss_reward = self.loss_critic(normalized_rewards.squeeze(), reward_labels.squeeze())
                    else:
                        loss_reward = self.loss_critic(normalized_rewards, reward_labels)

                    if self.normalize_reward:
                        loss_reward = (loss_reward - loss_reward.mean()) / loss_reward.std()

                    loss = loss_action + loss_reward * 0

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # lr scheduler
                    if self.scheduler is not None:
                        self.scheduler.step()

                    train_total_loss += loss.item()

                    train_pbar.set_postfix(
                        **{'train average loss': train_total_loss / (step + 1), 'train loss': loss.item(),
                           'lr': get_lr(self.optimizer)})

            # evaluate acc
            self.test(train_data_loader)

            if loss.item() < loss_min:
                self.save(Path(self.args.output) / (get_now_time() + f"_main_weights_{epoch}_{loss.item()}.pth"))
                self.save(Path(self.args.output) / "main_best.pth")

    def save(self, model_name: Union[Path, str]):
        torch.save(self.model.state_dict(), model_name)

    def test(self, test_data_loader: DataLoader = None):
        from sklearn.metrics import accuracy_score

        self.model.eval()
        accuracy = 0
        count = 0
        with torch.no_grad():
            with tqdm(enumerate(test_data_loader), total=len(test_data_loader),
                      desc=f'Evaluate accuracy', postfix=dict()) as test_pbar:
                for step, batch in test_pbar:
                    images, action_labels, reward_labels = batch
                    images = images.to(self.device)
                    action_labels = action_labels.to(self.device)

                    logits_action, _ = self.model(images)
                    predict_labels = torch.argmax(torch.softmax(logits_action, dim=-1), dim=-1).squeeze().cpu()
                    action_labels = action_labels.squeeze().cpu()
                    accuracy += accuracy_score(y_pred=predict_labels, y_true=action_labels)
                    count += 1
                    test_pbar.set_postfix(**{'accuracy': float(accuracy / count)})
