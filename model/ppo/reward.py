import PIL
import cv2
import time
import numpy as np
# import pytesseract
from pytesseract import pytesseract
from imagededup.methods import CNN
from PIL import Image
from ppocronnx import TextSystem
from typing import Union, Optional
from loguru import logger
from adbutils import adb
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.control import GameControl
from utils.general import ID2LABEL, ACTION2ID, STATE_ID2LABEL, STATE_REWARD


class GameState:
    ATTACK_MINION_MONSTER_TOWER = 0  # 攻击小兵/野怪/推塔
    ATTACK_ENEMY_HERO = 1  # 攻击敌方英雄
    ATTACKED_BY_TOWER = 2  # 被塔攻击
    ATTACKED_BY_ENEMY = 3  # 被敌方攻击
    DEAD_STATE = 4  # 死亡状态
    NORMAL_STATE = 5  # 普通状态


class AutoGameEnvironment:
    """
    STATE_LABEL2ID = {'攻击小兵/野怪/推塔': 0, '攻击敌方英雄': 1, '被塔攻击': 2, '被敌方攻击': 3, '死亡状态': 4, '普通状态': 5}
    """

    def __init__(self, serial=None):
        self.serial = adb.device().serial if serial is None else serial
        self.game_control = GameControl(self.serial)
        self.text_sys = TextSystem()
        self.is_game_continue_page = False

        # get reward
        self.last_move_and_action = "无移动_无动作"
        self.last_reward = 0
        self.last_status_id = 5  # 如果前后状态一致并且动作一致，那么reward应该为0

        # similarity
        # C:\Users\xxxx/.cache\torch\hub\checkpoints\mobilenet_v3_small-047dcff4.pth
        self.similarity_model = CNN(verbose=False)

        # golden vector: np.save("minion_monster.npy", image_vector) np.save("tower.npy", image_vector)
        self.minion_monster_vector = np.load("../../assets/vector_minion_monster.npy")
        self.tower_vector = np.load("../../assets/vector_tower.npy")

    def reset(self) -> np.ndarray:
        if self.is_game_continue_page:
            self.game_control.adb_device.click(1168, 956)

        # 对战
        self.game_control.adb_device.click(1168, 956)
        time.sleep(2)

        # 单人训练
        self.game_control.adb_device.click(2216, 78)
        time.sleep(2)
        self.game_control.adb_device.click(1722, 563)
        time.sleep(2)

        # 选择英雄
        self.game_control.adb_device.click(600, 591)
        time.sleep(2)
        self.game_control.adb_device.click(1500, 965)
        time.sleep(2)

        # 挑选对手
        self.game_control.adb_device.click(2479, 1168)
        time.sleep(2)
        self.game_control.adb_device.click(425, 485)
        time.sleep(2)

        # 开始游戏
        self.game_control.adb_device.click(2479, 1168)
        time.sleep(10)

        # 点击掉技能介绍
        self.game_control.adb_device.click(1191, 32)
        time.sleep(1)

        self.is_game_continue_page = False
        return self.game_control.get_screenshot()

    def get_reword(self, obs):
        reward, status_id = 0, GameState.NORMAL_STATE

        # 死亡状态
        if self.is_dead_state(obs):  # 0.009095907211303711s
            status_id = GameState.DEAD_STATE
            reward = STATE_REWARD[STATE_ID2LABEL[status_id]]
            return reward, status_id

        # 被塔攻击
        if self.attack_by_tower(obs):  # 0.001001596450805664
            status_id = GameState.ATTACKED_BY_TOWER
            reward = STATE_REWARD[STATE_ID2LABEL[status_id]]
            return reward, status_id

        # 被敌方英雄攻击
        if self.attack_by_enemy(obs):  # 0.38913822174072266
            status_id = GameState.ATTACKED_BY_ENEMY
            reward = STATE_REWARD[STATE_ID2LABEL[status_id]]
            return reward, status_id

        # 攻击塔、小兵
        if self.is_attack_status(obs):  # 0.03480219841003418
            is_attack_minion_monster_tower = self.is_attack_minion_monster_tower(obs)
            if is_attack_minion_monster_tower:
                status_id = GameState.ATTACK_MINION_MONSTER_TOWER
                reward = STATE_REWARD[STATE_ID2LABEL[status_id]]
                return reward, status_id
            else:
                # 如果是攻击野怪和敌方英雄，则可以获得更高的reward
                status_id = GameState.ATTACK_ENEMY_HERO
                reward = STATE_REWARD[STATE_ID2LABEL[status_id]]
                return reward, status_id
        else:
            status_id = GameState.NORMAL_STATE
            reward = STATE_REWARD[STATE_ID2LABEL[status_id]]
            return reward, status_id

    def get_image_vector(self, image: np.ndarray):
        return self.similarity_model.encode_image(image_array=image).flatten()

    def is_attack_minion_monster_tower(self, obs: np.ndarray) -> bool:
        cv2_image = self.get_attack_target_from_image(obs)
        image_vector = self.get_image_vector(cv2_image)

        is_minion_monster = self.compare_vectors(self.minion_monster_vector, image_vector)
        is_tower = self.compare_vectors(self.tower_vector, image_vector)

        if is_minion_monster or is_tower:
            return True
        return False

    def is_attack_enemy_hero(self, obs: np.ndarray) -> bool:
        return not self.is_attack_minion_monster_tower(obs)

    def attack_by_tower(self, obs: np.ndarray) -> bool:
        """
        check whether the hero is attacked by tower
        """
        crop_image = self.get_hero_attack_by_tower_label(obs)

        hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        blurred = cv2.GaussianBlur(mask, (9, 9), 2)

        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=100)
        if circles is not None:
            return True
        return False

    @staticmethod
    def detect_red(image, red_threshold=0.01):
        """
        检测图像中是否有红色部分
        :param image: 输入图像
        :param red_threshold: 红色像素比例阈值
        :return: 如果有红色部分返回True，否则返回False
        """
        # 将图像从BGR转换为HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 定义红色的HSV范围
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # 创建掩膜
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 计算红色像素的比例
        red_pixels = np.sum(mask > 0)
        total_pixels = image.shape[0] * image.shape[1]
        red_ratio = red_pixels / total_pixels

        return red_ratio > red_threshold

    def attack_by_enemy(self, obs: np.ndarray):
        hero_image = self.get_self_hero_target_from_image(obs)
        if self.contains_digits(hero_image):
            return True
        return False

    def is_dead_state(self, image: np.ndarray):
        if self.is_black_screen(image=image):
            return True
        return False

    def is_attack_status(self, image: np.ndarray):
        """
        you'd better input cropped image to save time
        """
        try:
            # crop image
            height, width = image.shape[:2]
            left, upper, right, lower = int(width * 0.56), int(height * 0.02), int(width * 0.68), int(height * 0.04)
            crop_image = self.get_target_bound(image, left, upper, right, lower)

            if self.detect_red(crop_image):
                return True
        except Exception as e:
            logger.error(e)
        return False

    def is_game_finished(self, image: np.ndarray):
        """
        check whether the game is finished
        """
        try:
            # crop image
            height, width = image.shape[:2]
            left, upper, right, lower = int(width * 0.36), int(height * 0.88), int(width * 0.49), int(height * 0.95)
            crop_image = self.get_target_bound(image, left, upper, right, lower)

            # ocr, get ocr text
            result = self.text_sys.detect_and_ocr(crop_image)
            for boxed_result in result:
                if boxed_result.ocr_text == "继续":
                    return True
        except Exception as e:
            logger.error(e)
        return False

    def contains_digits(self, image: np.ndarray):
        try:
            result = self.text_sys.detect_and_ocr(image)
            if any(boxed_result.ocr_text.isdigit() for boxed_result in result):
                return True
        except Exception as e:
            logger.error(e)
        return False

    @staticmethod
    def get_target_bound(image: Optional[np.ndarray], left: int, upper: int, right: int, lower: int):
        crop_image = np.ones(image.shape, dtype=np.uint8) * 255
        if image is not None:
            crop_image = image[upper:lower, left:right]
        return crop_image

    @staticmethod
    def get_click_window_continue(image: Union[PIL.Image.Image, np.ndarray]):
        crop_image = None
        if isinstance(image, PIL.Image.Image):
            width, height = image.size
            left, upper, right, lower = int(width * 0.40), int(height * 0.85), int(width * 0.60), int(height * 0.95)
            crop_image = image.crop((left, upper, right, lower))
        elif isinstance(image, np.ndarray):
            height, width = image.shape[:2]
            left, upper, right, lower = int(width * 0.40), int(height * 0.85), int(width * 0.60), int(height * 0.95)
            crop_image = image[upper:lower, left:right]
        return crop_image

    @staticmethod
    def get_attack_target_from_image(image: Union[PIL.Image.Image, np.ndarray]):
        crop_image = None
        if isinstance(image, PIL.Image.Image):
            width, height = image.size
            left, upper, right, lower = int(width * 0.675), int(height * 0.005), int(width * 0.705), int(height * 0.08)
            crop_image = image.crop((left, upper, right, lower))
        elif isinstance(image, np.ndarray):
            height, width = image.shape[:2]
            left, upper, right, lower = int(width * 0.675), int(height * 0.005), int(width * 0.705), int(height * 0.08)
            crop_image = image[upper:lower, left:right]
        return crop_image

    @staticmethod
    def get_self_hero_target_from_image(image: Union[PIL.Image.Image, np.ndarray]):
        """
        Image:
        image = Image.open(r"1724745801617.jpg")
        hero_image = env.get_self_hero_target_from_image(image)
        hero_image.show()

        cv2:
        image = cv2.imread(r"1724745801617.jpg")
        hero_image = env.get_self_hero_target_from_image(image)
        cv2.imshow("Image", hero_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        crop_image = None
        if isinstance(image, PIL.Image.Image):
            width, height = image.size
            left, upper, right, lower = int(width * 0.42), int(height * 0.4), int(width * 0.58), int(height * 0.62)
            crop_image = image.crop((left, upper, right, lower))
        elif isinstance(image, np.ndarray):
            height, width = image.shape[:2]
            left, upper, right, lower = int(width * 0.42), int(height * 0.4), int(width * 0.58), int(height * 0.62)
            crop_image = image[upper:lower, left:right]
        return crop_image

    @staticmethod
    def get_hero_attack_by_tower_label(image: Union[PIL.Image.Image, np.ndarray]):
        """
        Image:
        image = Image.open(r"1724745801617.jpg")
        hero_image = env.get_self_hero_target_from_image(image)
        hero_image.show()

        cv2:
        image = cv2.imread(r"1724745801617.jpg")
        hero_image = env.get_self_hero_target_from_image(image)
        cv2.imshow("Image", hero_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        crop_image = None
        if isinstance(image, PIL.Image.Image):
            width, height = image.size
            left, upper, right, lower = int(width * 0.42), int(height * 0.4), int(width * 0.58), int(height * 0.62)
            crop_image = image.crop((left, upper, right, lower))
        elif isinstance(image, np.ndarray):
            height, width = image.shape[:2]
            left, upper, right, lower = int(width * 0.49), int(height * 0.49), int(width * 0.51), int(height * 0.54)
            crop_image = image[upper:lower, left:right]
        return crop_image

    @staticmethod
    def is_black_screen(image: [PIL.Image.Image, np.ndarray],
                        threshold=72):
        """
        black screen: mean brightness is 59.69, normal image:85.17
        """
        mean_brightness = 255
        if isinstance(image, PIL.Image.Image):
            gray_image = image.convert('L')
            gray_array = np.array(gray_image)
            mean_brightness = np.mean(gray_array)
        elif isinstance(image, np.ndarray):
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray_array = np.array(gray_image)
            mean_brightness = np.mean(gray_array)
        return mean_brightness < threshold

    @staticmethod
    def compare_vectors(vector1, vector2, threshold=0.9):
        """
        比较两个特征向量的相似度。

        :param vector1: 第一个图片的特征向量
        :param vector2: 第二个图片的特征向量
        :param threshold: 相似度阈值（0-1之间）
        :return: 余弦相似度 >= threshold
        """
        similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        return similarity >= threshold

    @staticmethod
    def convert_cv2_to_pil(cv2_image: np.ndarray):
        """
        将 OpenCV 图像转换为 PIL 图像。

        :param cv2_image: OpenCV 图像（NumPy 数组，BGR 格式）
        :return: PIL 图像（RGB 格式）
        """
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        return pil_image

    @staticmethod
    def convert_pil_to_cv2(pil_image: PIL.Image.Image):
        """
        将 PIL 图像转换为 OpenCV 图像。

        :param pil_image: PIL 图像（RGB 格式）
        :return: OpenCV 图像（NumPy 数组，BGR 格式）
        """
        rgb_image = np.array(pil_image)
        cv2_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        return cv2_image

    def step(self, action: int = 128):
        move_and_action = ID2LABEL[str(action)]
        move_name, action_name = move_and_action.split('_')

        # direction
        self.game_control.send(int(ACTION2ID[move_name]))
        time.sleep(0.001)

        # action
        self.game_control.send(int(ACTION2ID[action_name]))
        time.sleep(0.001)

        next_frame = self.game_control.get_screenshot()

        zero_reward_name = ["发起进攻", "发起撤退", "发起集合", "无移动", "无动作"]
        if move_name in zero_reward_name or action_name in zero_reward_name:
            self.last_reward = 0
        else:
            self.last_reward, self.last_status_id = self.get_reword(next_frame)
        #     if self.last_move_and_action == move_and_action:
        #         cur_reward, cur_status_id = self.get_reword(next_frame)
        #
        #         # 如果动作相同情况下，然后得到的状态还是相同的，那么reward需要设置为0，此时为同一个动作持续在做
        #         if self.last_status_id == cur_status_id:
        #             self.last_reward, self.last_status_id = 0, cur_status_id
        #         else:
        #             self.last_reward, self.last_status_id = cur_reward, cur_status_id
        #     else:
        #         self.last_reward, self.last_status_id = self.get_reword(next_frame)
        # self.last_move_and_action = move_and_action

        done = self.is_game_finished(next_frame)
        if done:
            self.is_game_continue_page = True
        return next_frame, self.last_reward, done


if __name__ == '__main__':
    env = AutoGameEnvironment()
    image = env.game_control.get_screenshot()
    image = env.get_self_hero_target_from_image(image)
