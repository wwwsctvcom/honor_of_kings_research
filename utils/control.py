import time
from typing import Optional
import cv2

import scrcpy
import numpy as np
from pynput import keyboard
from pynput.keyboard import Key, KeyCode
from scrcpy import Client
from adbutils import adb
from loguru import logger

"""
x	    y	    说明	            x百分比	            y百分比
2376	1104	总分辨率		
160	    444	    金币	            0.067340067340067	0.402173913043478
296	    440 	物品1	        0.124579124579125	0.398550724637681
296 	566	    物品2	        0.124579124579125	0.51268115942029
470	    864 	摇杆	            0.197811447811448	0.782608695652174
1187	992	    回城	            0.499579124579125	0.898550724637681
1328	992	    加血	            0.558922558922559	0.898550724637681
1484	992	    闪现  	        0.624579124579125	0.898550724637681
1656	970	    1技能	        0.696969696969697	0.878623188405797
1775	764	    2技能	        0.747053872053872	0.692028985507246
1982	645 	3技能	        0.834175084175084	0.584239130434783
1970	432	    装备技能	        0.829124579124579	0.391304347826087
1553	855	    +1技能	        0.653619528619529	0.77445652173913
1671	651 	+2技能	        0.703282828282828	0.589673913043478
1878	530	    +3技能	        0.79040404040404	0.480072463768116
1986	944	    普通攻击	        0.835858585858586	0.855072463768116
1818	1006	左边攻击(英雄)	0.765151515151515	0.911231884057971
2072	786	    右边攻击(防御塔)	0.872053872053872	0.71195652173913
2222	156 	命令进攻	        0.935185185185185	0.141304347826087
2222	250	    命令撤退	        0.935185185185185	0.226449275362319
2222	336	    命令集合	        0.935185185185185	0.304347826086957
"""


class GAME:
    """
    "攻击": 0,
    "补刀": 1,
    "推塔": 2,
    "一技能": 3,
    "二技能": 4,
    "三技能": 5,
    "召唤师技能": 6,
    "回城": 7,
    "发起进攻": 8,
    "发起撤退": 9,
    "发起集合": 10,
    "上移": 11,
    "右移": 12,
    "下移": 13,
    "左移": 14,
    "左上移": 15,
    "左下移": 16,
    "右下移": 17,
    "右上移": 18,
    "移动停": 19,
    "无移动": 20,
    "无动作": 21,
    "恢复": 21
    """
    ACTION_ATTACK = 0
    ACTION_LAST_HIT = 1
    ACTION_PUSH_TOWER = 2
    ACTION_SKILL_ONE = 3
    ACTION_SKILL_TWO = 4
    ACTION_SKILL_THREE = 5
    ACTION_SUMMONER_SKILL = 6
    ACTION_RETURN = 7
    ACTION_INITIATE_ATTACK = 8
    ACTION_INITIATE_RETREAT = 9
    ACTION_INITIATE_GATHER = 10

    ACTION_MOVE_UP = 11
    ACTION_MOVE_RIGHT = 12
    ACTION_MOVE_DOWN = 13
    ACTION_MOVE_LEFT = 14
    ACTION_MOVE_LEFT_UP = 15
    ACTION_MOVE_LEFT_DOWN = 16
    ACTION_MOVE_RIGHT_DOWN = 17
    ACTION_MOVE_RIGHT_UP = 18
    ACTION_MOVE_STOP = 19
    ACTION_NO_MOVE = 20

    ACTION_NO_ACTION = 21
    ACTION_RECOVER = 22

    ACTION_ADD_SKILL_ONE = 23
    ACTION_ADD_SKILL_TWO = 24
    ACTION_ADD_SKILL_THREE = 25

    ACTION_BOUGHT_EQUIPMENT_ONE = 26
    ACTION_BOUGHT_EQUIPMENT_TWO = 27


class GameControl:
    """
    用于控制王者荣耀的界面
    """

    def __init__(self, serial=None):
        self.move_steps_delay = 0.005
        self.move_step_length = 10
        self.adb_device = adb.device()
        self.serial = serial
        if self.serial is None:
            self.client = Client(self.adb_device.serial)
        else:
            self.client = Client(self.serial)
        self.client.start(threaded=True, daemon_threaded=True)
        self.control = self.client.control
        time.sleep(3)
        self.width, self.height = self.adb_device.window_size()
        self.joystick_center_x, self.joystick_center_y = self.width * 0.19, self.height * 0.78

    def get_screenshot(self) -> Optional[np.ndarray]:
        if self.client.last_frame is not None:
            return self.client.last_frame

    def click(self, x, y, duration=0.0001) -> None:
        self.control.touch(x, y, scrcpy.ACTION_DOWN)
        time.sleep(duration)
        self.control.touch(x, y, scrcpy.ACTION_UP)

    def click_back_home(self):
        """
        回城
        """
        percent = (0.50, 0.91)
        x = int(self.width * percent[0])
        y = int(self.height * percent[1])
        self.click(x, y)

    def click_recover(self):
        """
        恢复
        """
        percent = (0.55, 0.89)
        x = int(self.width * percent[0])
        y = int(self.height * percent[1])
        self.click(x, y)

    def click_summoner_skill(self):
        """
        召唤师技能
        """
        percent = (0.624579124579125, 0.898550724637681)
        x = int(self.width * percent[0])
        y = int(self.height * percent[1])
        self.click(x, y)

    def click_money(self):
        percent = (0.08, 0.42)
        x = int(self.width * percent[0])
        y = int(self.height * percent[1])
        self.click(x, y)

    def bought_equipment_1(self):
        """
        购买商品1
        """
        percent = (0.13, 0.39)
        x = int(self.width * percent[0])
        y = int(self.height * percent[1])
        self.click(x, y, duration=0.1)

    def bought_equipment_2(self):
        percent = (0.124579124579125, 0.51268115942029)
        x = int(self.width * percent[0])
        y = int(self.height * percent[1])
        self.click(x, y, duration=0.1)

    def add_skill_1(self):
        """
        升级技能1
        """
        percent = (0.653619528619529, 0.77445652173913)
        x = int(self.width * percent[0])
        y = int(self.height * percent[1])
        self.click(x, y)

    def click_skill_1(self):
        """
        技能1
        """
        percent = (0.696969696969697, 0.878623188405797)
        x = int(self.width * percent[0])
        y = int(self.height * percent[1])
        self.click(x, y)

    def add_skill_2(self):
        """
        技能2加点
        """
        percent = (0.703282828282828, 0.589673913043478)
        x = int(self.width * percent[0])
        y = int(self.height * percent[1])
        self.click(x, y)

    def click_skill_2(self):
        """
        技能2
        """
        percent = (0.747053872053872, 0.692028985507246)
        x = int(self.width * percent[0])
        y = int(self.height * percent[1])
        self.click(x, y)

    def add_skill_3(self):
        """
        技能2加点
        """
        percent = (0.79040404040404, 0.480072463768116)
        x = int(self.width * percent[0])
        y = int(self.height * percent[1])
        self.click(x, y)

    def click_skill_3(self):
        """
        技能3（大招）
        """
        percent = (0.834175084175084, 0.584239130434783)
        x = int(self.width * percent[0])
        y = int(self.height * percent[1])
        self.click(x, y)

    def click_attack(self):
        """
        普通攻击
        """
        percent = (0.835858585858586, 0.855072463768116)
        x = int(self.width * percent[0])
        y = int(self.height * percent[1])
        self.click(x, y)

    def click_initiate_attack(self):
        """
        发起进攻信号
        """
        percent = (0.935185185185185, 0.141304347826087)
        x = int(self.width * percent[0])
        y = int(self.height * percent[1])
        self.click(x, y)

    def click_initiate_retreat(self):
        """
        发起撤退信号
        """
        percent = (0.935185185185185, 0.226449275362319)
        x = int(self.width * percent[0])
        y = int(self.height * percent[1])
        self.click(x, y)

    def click_initiate_gather(self):
        """
        发起集合信号
        """
        percent = (0.935185185185185, 0.304347826086957)
        x = int(self.width * percent[0])
        y = int(self.height * percent[1])
        self.click(x, y)

    def move_up(self):
        x_next = self.width * 0.20
        y_next = self.height * 0.59
        self.control.swipe(start_x=self.joystick_center_x,
                           start_y=self.joystick_center_y,
                           end_x=x_next,
                           end_y=y_next,
                           move_step_length=self.move_step_length,
                           move_steps_delay=self.move_steps_delay
                           )

    def move_down(self):
        x_next = self.width * 0.20
        y_next = self.height * 0.98
        self.control.swipe(start_x=self.joystick_center_x,
                           start_y=self.joystick_center_y,
                           end_x=x_next,
                           end_y=y_next,
                           move_step_length=self.move_step_length,
                           move_steps_delay=self.move_steps_delay
                           )

    def move_left(self):
        x_next = self.width * 0.10
        y_next = self.height * 0.78
        self.control.swipe(start_x=self.joystick_center_x,
                           start_y=self.joystick_center_y,
                           end_x=x_next,
                           end_y=y_next,
                           move_step_length=self.move_step_length,
                           move_steps_delay=self.move_steps_delay
                           )

    def move_right(self):
        x_next = self.width * 0.29
        y_next = self.height * 0.78
        self.control.swipe(start_x=self.joystick_center_x,
                           start_y=self.joystick_center_y,
                           end_x=x_next,
                           end_y=y_next,
                           move_step_length=self.move_step_length,
                           move_steps_delay=self.move_steps_delay
                           )

    def move_right_up(self):
        x_next = self.width * 0.25
        y_next = self.height * 0.65
        self.control.swipe(start_x=self.joystick_center_x,
                           start_y=self.joystick_center_y,
                           end_x=x_next,
                           end_y=y_next,
                           move_step_length=self.move_step_length,
                           move_steps_delay=self.move_steps_delay
                           )

    def move_right_down(self):
        x_next = self.width * 0.25
        y_next = self.height * 0.90
        self.control.swipe(start_x=self.joystick_center_x,
                           start_y=self.joystick_center_y,
                           end_x=x_next,
                           end_y=y_next,
                           move_step_length=self.move_step_length,
                           move_steps_delay=self.move_steps_delay
                           )

    def move_left_up(self):
        x_next = self.width * 0.13
        y_next = self.height * 0.65
        self.control.swipe(start_x=self.joystick_center_x,
                           start_y=self.joystick_center_y,
                           end_x=x_next,
                           end_y=y_next,
                           move_step_length=self.move_step_length,
                           move_steps_delay=self.move_steps_delay
                           )

    def move_left_down(self):
        x_next = self.width * 0.13
        y_next = self.height * 0.91
        self.control.swipe(start_x=self.joystick_center_x,
                           start_y=self.joystick_center_y,
                           end_x=x_next,
                           end_y=y_next,
                           move_step_length=self.move_step_length,
                           move_steps_delay=self.move_steps_delay
                           )

    def move_stop(self):
        self.control.touch(0, 0, scrcpy.ACTION_UP)

    def send(self, action: int):
        match action:
            case GAME.ACTION_ATTACK:
                self.click_attack()
            case GAME.ACTION_LAST_HIT:
                pass
            case GAME.ACTION_PUSH_TOWER:
                pass
            case GAME.ACTION_SKILL_ONE:
                self.click_skill_1()
            case GAME.ACTION_SKILL_TWO:
                self.click_skill_2()
            case GAME.ACTION_SKILL_THREE:
                self.click_skill_3()
            case GAME.ACTION_SUMMONER_SKILL:
                self.click_summoner_skill()
            case GAME.ACTION_RETURN:
                self.click_back_home()
            case GAME.ACTION_INITIATE_ATTACK:
                self.click_initiate_attack()
            case GAME.ACTION_INITIATE_RETREAT:
                self.click_initiate_retreat()
            case GAME.ACTION_INITIATE_GATHER:
                self.click_initiate_gather()
            case GAME.ACTION_MOVE_UP:
                self.move_up()
            case GAME.ACTION_MOVE_RIGHT:
                self.move_right()
            case GAME.ACTION_MOVE_DOWN:
                self.move_down()
            case GAME.ACTION_MOVE_LEFT:
                self.move_left()
            case GAME.ACTION_MOVE_LEFT_UP:
                self.move_left_up()
            case GAME.ACTION_MOVE_LEFT_DOWN:
                self.move_left_down()
            case GAME.ACTION_MOVE_RIGHT_DOWN:
                self.move_right_down()
            case GAME.ACTION_MOVE_RIGHT_UP:
                self.move_right_up()
            case GAME.ACTION_MOVE_STOP:
                self.move_stop()
            case GAME.ACTION_NO_MOVE:
                pass
            case GAME.ACTION_NO_ACTION:
                pass
            case GAME.ACTION_RECOVER:
                self.click_recover()
            case GAME.ACTION_ADD_SKILL_ONE:
                self.add_skill_1()
            case GAME.ACTION_ADD_SKILL_TWO:
                self.add_skill_2()
            case GAME.ACTION_ADD_SKILL_THREE:
                self.add_skill_3()
            case GAME.ACTION_BOUGHT_EQUIPMENT_ONE:
                self.bought_equipment_1()
            case GAME.ACTION_BOUGHT_EQUIPMENT_TWO:
                self.bought_equipment_2()
            case _:
                pass


class KeyboardListener:
    """
    wasd用于控制人物的上下左右移动
    b回城
    n购买物品1
    m购买物品2

    小键盘：
        123分别代表技能123
        ->方向右键代表召唤师技能1
        0代表召唤师技能2
        enter代表普通攻击
    """
    pressed_q, pressed_w, pressed_a, pressed_s, pressed_d = False, False, False, False, False
    pressed_b, pressed_n, pressed_m = False, False, False

    (pressed_1, pressed_2, pressed_3,
     pressed_4, pressed_5, pressed_6,
     pressed_0, pressed_enter, pressed_recover) = False, False, False, False, False, False, False, False, False

    def __init__(self):
        self.current_keys = set()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)

    def on_press(self, key):
        try:
            if isinstance(key, KeyCode):
                if key.char == "q":
                    self.pressed_q = True

                if key.char == "w":
                    self.pressed_w = True

                if key.char == "a":
                    self.pressed_a = True

                if key.char == "s":
                    self.pressed_s = True

                if key.char == "d":
                    self.pressed_d = True

                if key.char == "b":
                    self.pressed_b = True

                if key.char == 'n':
                    self.pressed_n = True

                if key.char == 'm':
                    self.pressed_m = True

                if str(key) == str(KeyCode.from_vk(96)):
                    self.pressed_0 = True

                if str(key) == str(KeyCode.from_vk(97)):
                    self.pressed_1 = True

                if str(key) == str(KeyCode.from_vk(98)):
                    self.pressed_2 = True

                if str(key) == str(KeyCode.from_vk(99)):
                    self.pressed_3 = True

                if str(key) == str(KeyCode.from_vk(100)):
                    self.pressed_4 = True

                if str(key) == str(KeyCode.from_vk(101)):
                    self.pressed_5 = True

                if str(key) == str(KeyCode.from_vk(102)):
                    self.pressed_6 = True
            elif isinstance(key, Key):
                if str(key) == "Key.right":
                    self.pressed_recover = True
                if str(key) == "Key.enter":
                    self.pressed_enter = True
                if str(key) == "Key.esc":
                    self.listener.stop()
        except Exception as e:
            logger.info(f"keyboard failed {e}")

    def on_release(self, key):
        try:
            if isinstance(key, KeyCode):
                if key.char == "q":
                    self.pressed_q = False

                if key.char == "w":
                    self.pressed_w = False

                if key.char == "a":
                    self.pressed_a = False

                if key.char == "s":
                    self.pressed_s = False

                if key.char == "d":
                    self.pressed_d = False

                if key.char == "b":
                    self.pressed_b = False

                if key.char == 'n':
                    self.pressed_n = False

                if key.char == 'm':
                    self.pressed_m = False

                if str(key) == str(KeyCode.from_vk(96)):
                    self.pressed_0 = False

                if str(key) == str(KeyCode.from_vk(97)):
                    self.pressed_1 = False

                if str(key) == str(KeyCode.from_vk(98)):
                    self.pressed_2 = False

                if str(key) == str(KeyCode.from_vk(99)):
                    self.pressed_3 = False

                if str(key) == str(KeyCode.from_vk(100)):
                    self.pressed_4 = False

                if str(key) == str(KeyCode.from_vk(101)):
                    self.pressed_5 = False

                if str(key) == str(KeyCode.from_vk(102)):
                    self.pressed_6 = False
            elif isinstance(key, Key):
                if str(key) == "Key.right":
                    self.pressed_recover = False
                if str(key) == "Key.enter":
                    self.pressed_enter = False
                if str(key) == "Key.esc":
                    self.listener.stop()
        except Exception as e:
            logger.info(f"keyboard failed {e}")

    def get_action(self):
        move_action = {"move": GAME.ACTION_NO_MOVE, "action": GAME.ACTION_NO_ACTION}

        # direction
        if self.pressed_q:
            move_action["move"] = GAME.ACTION_MOVE_STOP
        elif self.pressed_w and not self.pressed_a and not self.pressed_s and not self.pressed_d:
            move_action["move"] = GAME.ACTION_MOVE_UP
        elif not self.pressed_w and self.pressed_a and not self.pressed_s and not self.pressed_d:
            move_action["move"] = GAME.ACTION_MOVE_LEFT
        elif not self.pressed_w and not self.pressed_a and self.pressed_s and not self.pressed_d:
            move_action["move"] = GAME.ACTION_MOVE_DOWN
        elif not self.pressed_w and not self.pressed_a and not self.pressed_s and self.pressed_d:
            move_action["move"] = GAME.ACTION_MOVE_RIGHT

        elif self.pressed_w and self.pressed_a and not self.pressed_s and not self.pressed_d:
            move_action["move"] = GAME.ACTION_MOVE_LEFT_UP

        elif not self.pressed_w and self.pressed_a and self.pressed_s and not self.pressed_d:
            move_action["move"] = GAME.ACTION_MOVE_LEFT_DOWN

        elif self.pressed_w and not self.pressed_a and not self.pressed_s and self.pressed_d:
            move_action["move"] = GAME.ACTION_MOVE_RIGHT_UP

        elif not self.pressed_w and not self.pressed_a and self.pressed_s and self.pressed_d:
            move_action["move"] = GAME.ACTION_MOVE_RIGHT_DOWN

        # skill
        if self.pressed_0:
            move_action["action"] = GAME.ACTION_SUMMONER_SKILL
        elif self.pressed_1:
            move_action["action"] = GAME.ACTION_SKILL_ONE
        elif self.pressed_2:
            move_action["action"] = GAME.ACTION_SKILL_TWO
        elif self.pressed_3:
            move_action["action"] = GAME.ACTION_SKILL_THREE

        # upgrade skill
        elif self.pressed_4:
            move_action["action"] = GAME.ACTION_ADD_SKILL_ONE
        elif self.pressed_5:
            move_action["action"] = GAME.ACTION_ADD_SKILL_TWO
        elif self.pressed_6:
            move_action["action"] = GAME.ACTION_ADD_SKILL_THREE

        # normal attack
        elif self.pressed_enter:
            move_action["action"] = GAME.ACTION_ATTACK
        elif self.pressed_b:
            move_action["action"] = GAME.ACTION_RETURN
        elif self.pressed_recover:
            move_action["action"] = GAME.ACTION_RECOVER

        # bought equipment
        elif self.pressed_n:
            move_action["action"] = GAME.ACTION_BOUGHT_EQUIPMENT_ONE
        elif self.pressed_m:
            move_action["action"] = GAME.ACTION_BOUGHT_EQUIPMENT_TWO

        return move_action

    def start(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()
