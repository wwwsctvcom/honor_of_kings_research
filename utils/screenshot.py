import io
import os
import numpy as np
import PIL
import adbutils
import win32con
import win32gui
import win32ui
import subprocess
import weakref
import psutil
import socket
import sys
import time
import struct
import threading
import scrcpy
from scrcpy import Client
from typing import Optional
from pathlib import Path
from loguru import logger
from PIL import Image
from utils.general import *
from adbutils import adb
from pyminitouch import MNTDevice


class ControlDevice(MNTDevice):
    """
    examples: control_device = ControlDevice()
              control_device.tap([(100, 200)])
    """

    def __init__(self):
        os.system("adb kill-server")
        os.system("adb start-server")
        super().__init__(adb.device().serial)

    def send(self, action: str):
        self.connection.send(action)


class ScrcpyWindowCapture:

    def __init__(self,
                 window_name: str = None,
                 window_max_size: int = None):
        self._window_name = window_name
        if self._window_name is None:
            self._window_name = adbutils.adb.device().shell("getprop ro.product.odm_dlkm.model")
        logger.info(f"window name: {self._window_name}")

        if window_max_size is None:
            self._window_max_size = 960

        self._scrcpy_path = "scrcpy" if self.is_scrcpy_available() else Path(SCRCPY_PATH) / "scrcpy.exe"
        logger.info(f"scrcpy path {self._scrcpy_path}")
        self._p: subprocess.Popen

        # start scrcpy
        self.desktop_win = None
        self._start()

    @staticmethod
    def is_scrcpy_available():
        try:
            result = subprocess.run(["scrcpy", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            if result.returncode == 0:
                return True
            else:
                return False
        except FileNotFoundError:
            logger.error("scrcpy is not found in the environment variables.")
            return False

    def is_recording(self) -> bool:
        return bool(self._p and self._p.poll() is None)

    def _start(self):
        if self.check_env():
            self._p = subprocess.Popen(
                [self._scrcpy_path, "--window-borderless", "--max-size", str(self._window_max_size)],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            self._finalizer = weakref.finalize(self._p, self._p.kill)

            # wait for opening window
            count = 0
            while True:
                self.desktop_win = win32gui.FindWindow(0, self._window_name)
                if self.desktop_win != 0 or count > 20:
                    logger.info(f"wait for window: {self._window_name} handle valid...")
                    break
                count += 1
                time.sleep(1)

    def check_env(self) -> bool:
        return self._scrcpy_path is not None

    def get_screenshot(self):
        left, top, right, bot = win32gui.GetWindowRect(self.desktop_win)
        width = right - left
        height = bot - top

        # create window device context
        win_dc_handle = win32gui.GetWindowDC(self.desktop_win)
        win_dc = win32ui.CreateDCFromHandle(win_dc_handle)
        mem_dc = win_dc.CreateCompatibleDC()

        # create bitmap
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(win_dc, width, height)
        mem_dc.SelectObject(bitmap)

        # copy window content to bitmap
        mem_dc.BitBlt((0, 0), (width, height), win_dc, (0, 0), win32con.SRCCOPY)

        bitmap_info = bitmap.GetInfo()
        bitmap_str = bitmap.GetBitmapBits(True)
        image_from_buffer = Image.frombuffer('RGB',
                                             (bitmap_info['bmWidth'], bitmap_info['bmHeight']),
                                             bitmap_str, 'raw', 'BGRX')
        img = image_from_buffer.crop((0, 0, bitmap_info['bmWidth'], bitmap_info['bmHeight']))

        # release
        win32gui.DeleteObject(bitmap.GetHandle())
        mem_dc.DeleteDC()
        win_dc.DeleteDC()
        win32gui.ReleaseDC(self.desktop_win, win_dc_handle)
        return img

    def _stop(self):
        self._finalizer.detach()
        if self._p:
            ps_proc = psutil.Process(self._p.pid)
            for child in ps_proc.children(recursive=True):
                child.terminate()
            ps_proc.terminate()
            gone, still_alive = psutil.wait_procs([ps_proc] + ps_proc.children(), timeout=3)
            for p in still_alive:
                p.kill()

    def release(self):
        self._stop()


class ScrcpyWindowCaptureSocket(Client):
    """
    scrcpy: pip install scrcpy-client
    """

    def __init__(self):
        super().__init__()
        self.serial = adbutils.device().serial
        logger.info(f"Serial: {self.serial}")
        self.server = scrcpy.Client(device=self.serial, bitrate=100000000, connection_timeout=10 * 1000)
        self.server.start(threaded=True, daemon_threaded=True)

    def get_screenshot(self) -> Optional[np.ndarray]:
        if self.server.last_frame is not None:
            return self.server.last_frame

    def release(self):
        self.server.stop()


class MinicapWindowCapture:

    def __init__(self, minicap_path: str = None):
        self.buffer_sizer = 4096
        self.d = adb.device()

        if minicap_path is None:
            self.minicap_path = MINICAP_PATH

        self.cpu_abi = self.d.shell("getprop ro.product.cpu.abi")  # arm64-v8a
        logger.info(self.cpu_abi)
        self.sdk_version = self.d.shell("getprop ro.build.version.sdk")  # 34
        logger.info(self.sdk_version)

        self.minicap_bin = Path(self.minicap_path) / "bin/minicap"
        self.minicap_so = Path(self.minicap_path) / f"lib/android-{self.sdk_version}" / self.cpu_abi / "minicap.so"

        self.d.sync.push(self.minicap_bin, "/data/local/tmp/minicap")

        self.d.sync.push(self.minicap_so, "/data/local/tmp/minicap.so")

        # start minicap server
        self.start_minicap_server()
        self.forward_minicap()

        # wait for server
        self.is_server_start = False
        self.wait_for_server()

        # connect server
        self.host, self.port = "localhost", 1717
        self._socket = None
        self.connect()

        # get image data
        self.lock = threading.Lock()
        self.latest_frame: PIL.Image = None
        self.update_frame_thread = threading.Thread(target=self.read_frame)
        self.update_frame_thread.start()

    def connect(self):
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error as e:
            logger.error(f"create socket failed, {e}")
            sys.exit(1)
        self._socket.connect((self.host, self.port))

    def wait_for_server(self):
        count = 0
        while not self.is_server_start:
            time.sleep(1)
            count += 1
            if count > 20:
                logger.error("open minicap server failed.")
                sys.exit(1)

    def forward_minicap(self):
        forward_cmd = "adb forward tcp:1717 localabstract:minicap"
        logger.info(forward_cmd)

        forward_proc = subprocess.Popen(forward_cmd,
                                        shell=True,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)

        stdout_thread = threading.Thread(target=self._reader, args=(forward_proc.stdout,))
        stdout_thread.start()

        stderr_thread = threading.Thread(target=self._reader, args=(forward_proc.stderr,))
        stderr_thread.start()

    def start_minicap_server(self):
        width, height = self.d.window_size()
        server_cmd = f"adb shell LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/minicap -P {width}x{height}@{width}x{height}/0"
        logger.info(server_cmd)

        server_proc = subprocess.Popen(server_cmd,
                                       shell=True,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)

        stdout_thread = threading.Thread(target=self._reader, args=(server_proc.stdout,))
        stdout_thread.start()

        stderr_thread = threading.Thread(target=self._reader, args=(server_proc.stderr,))
        stderr_thread.start()

    def _reader(self, pipe):
        try:
            with pipe:
                for line in iter(pipe.readline, b''):
                    if "Allocating" in line.decode('utf-8'):
                        self.is_server_start = True
                    logger.info(line.decode('utf-8'))
        except Exception as e:
            logger.error(f"Error read: {e}")

    @staticmethod
    def parse_banner(chunk):
        banner = {'version': chunk[0],
                  'length': chunk[1],
                  'pid': struct.unpack('<I', chunk[2:6])[0],
                  'realWidth': struct.unpack('<I', chunk[6:10])[0],
                  'realHeight': struct.unpack('<I', chunk[10:14])[0],
                  'virtualWidth': struct.unpack('<I', chunk[14:18])[0],
                  'virtualHeight': struct.unpack('<I', chunk[18:22])[0],
                  'orientation': chunk[22] * 90,
                  'quirks': chunk[23]
                  }
        return banner

    def read_frame(self):
        read_banner_bytes = 0
        banner_length = 24
        read_frame_bytes = 0
        frame_body_length = 0
        frame_body = bytearray()

        while True:
            try:
                chunk = self._socket.recv(4096)
            except socket.error as e:
                logger.error(f"socket failed, {e}")
                sys.exit(1)

            cursor = 0
            while cursor < len(chunk):
                if read_banner_bytes < banner_length:
                    banner = self.parse_banner(chunk[:banner_length])
                    cursor = len(chunk)
                    read_banner_bytes = banner_length
                    logger.info(banner)
                elif read_frame_bytes < 4:
                    frame_body_length += (chunk[cursor] << (read_frame_bytes * 8))
                    cursor += 1
                    read_frame_bytes += 1
                else:
                    if len(chunk) - cursor >= frame_body_length:
                        frame_body.extend(chunk[cursor:cursor + frame_body_length])

                        if frame_body[0] != 0xFF or frame_body[1] != 0xD8:
                            logger.error(f'frame body does not start with JPG header {frame_body}')
                            return

                        with self.lock:
                            self.latest_frame = Image.open(io.BytesIO(frame_body))
                        cursor += frame_body_length
                        frame_body_length = read_frame_bytes = 0
                        frame_body = bytearray()
                    else:
                        frame_body.extend(chunk[cursor:])

                        frame_body_length -= len(chunk) - cursor
                        read_frame_bytes += len(chunk) - cursor
                        cursor = len(chunk)

    def get_screenshot(self):
        return self.latest_frame

    def release(self):
        self._socket.close()
        self.update_frame_thread.join()
        self.latest_frame = None


if __name__ == "__main__":
    scrcpy = ScrcpyWindowCapture()
    scrcpy.get_screenshot()

    minicap = MinicapWindowCapture()
    minicap.get_screenshot()

    scrcpy = ScrcpyWindowCaptureSocket()
    scrcpy.get_screenshot()
