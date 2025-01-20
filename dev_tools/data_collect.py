import json
import time

from pathlib import Path
from PIL import Image
from utils.control import GAME
from loguru import logger
from utils.control import GameControl
from utils.tools import get_now_time
from utils.control import KeyboardListener
from utils.general import *


if __name__ == "__main__":
    logger.info("start init keyboard listening...")
    keyboard_listener = KeyboardListener()
    keyboard_listener.start()
    # keyboard_listener.listener.join()

    logger.info("start init game control...")
    game_control = GameControl()

    data_path = Path("data") / get_now_time()
    if not data_path.exists():
        data_path.mkdir()
        logger.info(f"create data save path: {str(data_path)}")

    with open(data_path / "action.jsonl", mode="w", encoding="utf8") as train_writer:
        while True:
            # get frame firstly, then do action, make sure the frame will not delay
            frame = game_control.get_screenshot()
            action_ret = keyboard_listener.get_action()

            # all the action will be executed, but not all action will be record
            move, action = action_ret["move"], action_ret["action"]
            game_control.send(move)
            time.sleep(0.00005)
            game_control.send(action)

            if move == GAME.ACTION_NO_MOVE and action == GAME.ACTION_NO_ACTION:
                continue

            # screenshot: only the action is required, the screenshot and action will be record
            if -1 < move < 23 and -1 < action < 23:
                label_str = ID2ACTION[str(move)] + "_" + ID2ACTION[str(action)]
                label_id = int(LABEL2ID[label_str])
                try:
                    image_name = f"{int(time.time() * 1000)}.jpg"
                    action_ret.update({"image_name": image_name, "label": label_id})
                    train_writer.write(json.dumps(action_ret) + "\n")

                    frame = frame[..., ::-1]  # BGR -> RGB
                    image = Image.fromarray(frame)

                    image.save(data_path / image_name)
                    logger.info(f"saving path: {data_path}, image name: {image_name}, label: {label_id}")
                except Exception as e:
                    logger.info(f"save image, move, action data failed {e}, just skip.")
