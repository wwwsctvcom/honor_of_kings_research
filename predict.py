import argparse
import time
import torch
from PIL import Image
from utils.tools import get_available_device, json_reader
from utils.control import GameControl, GAME
from utils.datasets import ImagePreprocess
from utils.general import *
from train_main_model import load_main_model
from loguru import logger


def summoner_skill_init(control: GameControl):
    logger.info("Summoner skill init, bought equipment and add skill!")
    control.send(GAME.ACTION_BOUGHT_EQUIPMENT_ONE.value)
    time.sleep(0.02)
    control.send(GAME.ACTION_BOUGHT_EQUIPMENT_TWO.value)
    time.sleep(0.02)
    control.send(GAME.ACTION_ADD_SKILL_ONE.value)
    time.sleep(0.02)
    control.send(GAME.ACTION_ADD_SKILL_TWO.value)
    time.sleep(0.02)
    control.send(GAME.ACTION_ADD_SKILL_THREE.value)
    time.sleep(0.02)
    control.send(GAME.ACTION_MOVE_STOP.value)
    time.sleep(0.02)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="./weights/main_best.pth", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = set_args()
    device = get_available_device()

    logger.info(f"Loading model from {args.model_name_or_path}")
    model = load_main_model(args.model_name_or_path, requires_grad=False)
    model.to(device)

    # image preprocessor
    transform = ImagePreprocess()

    # get latest screenshot
    game_control = GameControl()

    count = 0
    while True:
        try:
            frame = game_control.get_screenshot()
            if frame is None:
                continue
            frame = game_control.get_screenshot()
            frame = frame[..., ::-1]  # BGR -> RGB
            image = Image.fromarray(frame)
            batch_image = transform(image).unsqueeze(0)
        except Exception as e:
            logger.info(f"get screenshot failed, {e}")
            continue

        if count % 10 == 0 or count == 0:
            summoner_skill_init(game_control)
        else:
            action_logits, _ = model(batch_image)
            action = torch.argmax(torch.softmax(action_logits, dim=-1), dim=-1)
            move_and_action = ID2LABEL[str(action.item())]
            move_name, action_name = move_and_action.split('_')

            if move_name != '无移动':
                logger.info(f"send: {move_name}")
                game_control.send(int(ACTION2ID[move_name]))
                time.sleep(0.01)

            if action_name not in ['无动作', '发起集合', '发起进攻', '发起撤退', '无方向']:
                logger.info(f"send: {action_name}")
                game_control.send(int(ACTION2ID[action_name]))
                time.sleep(0.01)
            time.sleep(1)
        count = count + 1
