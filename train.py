import os
import argparse
import torch
from pathlib import Path
from loguru import logger
from torch.utils.data import DataLoader
from utils.tools import seed_everything, get_available_device
from utils.datasets import AutoGameImageDataset, auto_game_collate_fn
from model.model_config import ResNetModelConfig
from model.model_resnet import AutoGameForImageClassification
from utils.train_utils import MainModelTrainer


def set_args():
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument("--scratch", type=bool, default=True,
                        help="whether train the model from scratch!")

    parser.add_argument("--model_name_or_path", default="./weights/main_best.pth", type=str)

    parser.add_argument("--epochs", type=int, default=50, help="")
    parser.add_argument("--output", type=str,
                        default="./weights",
                        help="output dir for save trained model!")
    parser.add_argument("--learning_rate", type=float, default=0.0003, help="learning rate for training!")
    parser.add_argument("--seed", type=int, default=42, help="")

    # data
    parser.add_argument("--train_data_path", type=str, default="./data", help="")
    parser.add_argument("--batch_size", type=int, default=32, help="")
    parser.add_argument("--num_workers", type=int, default=2, help="")
    return parser.parse_args()


def load_main_model(model_name_or_path: str = None, requires_grad: bool = False):
    resnet_config = ResNetModelConfig()
    model = AutoGameForImageClassification(config=resnet_config)

    if Path(model_name_or_path).exists():
        logger.info(f"Loading main model from: {model_name_or_path}")
        model.load_state_dict(torch.load(model_name_or_path, map_location=get_available_device()))

    # set weights trainable
    model.requires_grad_(requires_grad)
    return model


if __name__ == "__main__":
    args = set_args()
    seed_everything(args.seed)
    device = get_available_device()

    # dataset
    logger.info(f"Loading dataset from: {args.train_data_path}")
    for game_dataset_path in Path(args.train_data_path).iterdir():
        # init state model
        main_model = load_main_model(args.model_name_or_path, requires_grad=True)
        main_model.to(device)

        dataset = AutoGameImageDataset(game_dataset_path)
        data_loader = DataLoader(dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=auto_game_collate_fn)

        # train
        optimizer = torch.optim.Adam(main_model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               args.epochs * len(data_loader),
                                                               eta_min=0,
                                                               last_epoch=-1,
                                                               verbose=False)
        trainer = MainModelTrainer(args=args,
                                   model=main_model,
                                   scheduler=scheduler,
                                   optimizer=optimizer,
                                   normalize_reward=False,
                                   device=device)
        trainer.train(train_data_loader=data_loader)
