from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from scipy import spatial
from socialtouch.config import Config
from time import time
from threading import (Lock, Thread)
from tqdm import tqdm

import cv2
import numpy as np
import pygame as pg
import pandas as pd
import serial
import torch


class Interface:
    def __init__(self, port: str, baudrate: int, rx: int, tx: int, n: int, min: int, max: int) -> None:
        self.port, self.baudrate = port, baudrate
        self.rx, self.tx, self.cells, self.n = rx, tx, rx * tx,n
        self.min, self.max = min, max
        
        self.serial = serial.Serial(self.port, baudrate=self.baudrate)
        self.buffer = np.zeros((self.tx, self.rx * self.n))
        self.lock = Lock()
        self.thread = Thread(target=self.update)
        
        self.start()
        
    def connect   (self) -> None : self.serial.write("connected\r\n".encode())
    def disconnect(self) -> None : self.serial.write("disconnected\r\n".encode())
    def calibrate (self) -> None : self.serial.write("c\r\n".encode())
    def readline  (self) -> bytes: return self.serial.readline().strip(b"\r\n")
    
    def read(self) -> np.ndarray:
        with self.lock: return self.buffer.copy()

    def update(self) -> None:
        while self.run:
            buffer = [char for char in self.readline()]
            if len(buffer) != self.cells * self.n: continue
            buffers = [np.array(buffer[i * self.cells:(i + 1) * self.cells]).reshape(self.tx, self.rx) for i in range(self.n)]
            buffer = (np.hstack(buffers).clip(self.min, self.max) - self.min) / (self.max - self.min) * 255
            with self.lock: self.buffer[:, :] = buffer[:, :]

    def start(self) -> None:
        self.connect()
        self.calibrate()
        self.run = True
        self.thread.start()

    def stop(self) -> None:
        self.run = False
        self.thread.join()
        self.disconnect() 


class Model:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.ckpt = torch.load(str(self.path.resolve()), map_location="cpu")
        self.conf: Config = self.ckpt["conf"]
        self.model = self.conf.model.make(self.conf.dataset.height, self.conf.dataset.width, 32, self.conf.dataset.window_size).eval()
        self.model.load_state_dict(self.ckpt["model"])
        self.sequence_buffer = torch.zeros((1, self.conf.dataset.window_size, 1, self.conf.dataset.height, self.conf.dataset.width))

    @torch.inference_mode()
    def __call__(self, buffer: np.ndarray) -> tuple(np.ndarray, int):
        bh, bw = buffer.shape[-2:]
        sh, sw = self.sequence_buffer.shape[-2:]
        
        h, w = min(bh, sh), min(bw, sw)
        self.sequence_buffer[:, :-1] = self.sequence_buffer[:, 1:].clone()
        self.sequence_buffer[:, -1, :, :h, :w] = torch.from_numpy(buffer[None, :h, :w] / 255.0)

        probs = torch.softmax(self.model(self.sequence_buffer), dim=1).squeeze(0)
        probs = probs.cpu().numpy()

        return probs, np.argmax(probs)


class Player:
    def __init__(self, path: Path, k: int, thresh: float) -> None:
        self.path = path
        self.k, self.thresh = k, thresh
        self.load = lambda p, e: pd.read_csv(str(Path(self.path, p).resolve())).drop(columns=e)
        self.kd = spatial.KDTree(self.load("AudioRated.csv", ["ID", "Filename"]).values)
        self.proj = self.load("GesturesRated.csv", ["id", "gesture_name"]).values
        self.audios = self.preload()

    def preload(self) -> None:
        audios = []
        with tqdm(sorted(self.path.glob("*.mp3")), desc="Loading Audio") as pbar:
            for path in pbar:
                pbar.set_postfix(path=path)
                audios.append(pg.mixer.Sound(str(path.resolve())))
        return audios

    def __call__(self, probs: np.ndarray, idx: int) -> None:
        if probs[idx] < self.thresh: return
        self.audios[np.random.choice(self.kd.query(probs @ self.proj, self.k)[1])].play()


class Clock:
    def __init__(self) -> None:
        self.last, self.fps, self.timer, self.calib = time(), 0, 0, 0 

    def update(self) -> None:
        current = time()
        dt = current - self.last
        self.last = current
        self.fps = 1 / dt
        self.timer += dt
        self.calib += dt


class App:
    def __init__(self, args: Namespace) -> None:
        for key, val in args._get_kwargs():
            setattr(self, key, val)
        
        self.init()
        self.screen = pg.display.set_mode((self.rx * self.n * self.scale, self.tx * self.scale)) if self.display else None
        self.font = pg.font.SysFont("Ubuntu", 30) if self.display else None

        self.interface = None if self.random else Interface(self.port, self.baud, self.rx, self.tx, self.n, self.min, self.max)
        self.model = Model(Path(self.checkpoint))
        self.player = Player(Path(self.data), self.k, self.thresh)
        self.clock = Clock()

    def init(self) -> None:
        if self.display:
            pg.init()
            pg.font.init()
        pg.mixer.init()

    def draw(self, buffer: np.ndarray, probs: np.ndarray, idx: int) -> None:
        if self.display:
            buffer = buffer.astype(np.uint8)[:, :, None].repeat(3, axis=-1)
            buffer = cv2.resize(buffer, (self.rx * self.n * self.scale, self.tx * self.scale), interpolation=cv2.INTER_CUBIC)
            self.screen.blit(pg.image.frombuffer(buffer.tobytes(), buffer.shape[1::-1], "RGB"), (0, 0))
            self.screen.blit(self.font.render(f"{self.clock.fps:.2f} FPS",                False, (255, 255, 255)), (25, 25))
            self.screen.blit(self.font.render(f"G{idx:02d} {probs[idx] * 100:.2f}%", False, (255, 255, 255)), (25, 25 + 32))
            pg.display.update()
            pg.display.flip()
        else: print(f"{self.clock.fps:.2f} FPS | G{idx:02d} {probs[idx] * 100:.2f}%", end="\r")

    def activity(self, buffer: np.ndarray) -> bool:
        return np.sum(buffer / 255.0) / np.product(buffer.shape) > self.calib_thresh

    def quit(self) -> None:
        if self.interface is not None: self.interface.stop()
        pg.quit()
        exit(0)

    def events(self) -> None:
        if self.display:
            for event in pg.event.get():
                if event.type == pg.QUIT: self.quit()
                if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE: self.quit()

    def run(self) -> None:
        try:
            while True:
                self.events()
                
                buffer = np.random.random((self.tx, self.rx * self.n)) if self.random else self.interface.read()
                activity = self.activity(buffer)
                probs, idx = self.model(buffer)
                
                if self.clock.timer > self.timer:
                    self.clock.timer = 0
                    self.player(probs, idx)

                if activity: self.clock.calib = 0
                if self.clock.calib > self.calib and self.interface is not None:
                    self.clock.calib = 0
                    self.interface.calibrate()

                self.draw(buffer, probs, idx)
                self.clock.update()
        
        except KeyboardInterrupt:
            self.quit()


if __name__ == "__main__":
    from argparse import ArgumentParser


    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint",   type=str,   default="res/racnet_48_1.pt", help="Path to the model checkpoint"                                          )
    parser.add_argument("-d", "--data",         type=str,   default="audio",              help="Path to the data directory"                                            )
    parser.add_argument("-p", "--port",         type=str,   default="/dev/ttyUSB0",       help="Port for the muca"                                                     )
    parser.add_argument("-b", "--baud",         type=int,   default=2_000_000,            help="Baud rate for the muca"                                                )
    parser.add_argument("-m", "--min",          type=int,   default=10,                   help="Minimum threshold for the raw values"                                  )
    parser.add_argument("-M", "--max",          type=int,   default=100,                  help="Maximum threshold for the raw values"                                  )
    parser.add_argument(      "--rx",           type=int,   default=12,                   help="Number of columns"                                                     )
    parser.add_argument(      "--tx",           type=int,   default=19,                   help="Number of rows"                                                        )
    parser.add_argument(      "--n",            type=int,   default=2,                    help="Number of muca"                                                        )
    parser.add_argument(      "--scale",        type=int,   default=30,                   help="Number of pixel per cell"                                              )
    parser.add_argument(      "--thresh",       type=float, default=0.7,                  help="Probability threshold to accept a prediction as valid"                 )
    parser.add_argument(      "--k",            type=int,   default=10,                   help="Number of neihgbour to consider in the kd-tree"                        )
    parser.add_argument(      "--timer",        type=float, default=0.5,                  help="Time to wait before sounds"                                            )
    parser.add_argument(      "--calib",        type=float, default=10.0,                 help="Time to wait before auto calibration"                                  )
    parser.add_argument(      "--calib_thresh", type=float, default=0.2,                  help="Calibration threshold in percent"                                      )
    parser.add_argument(      "--display",      action="store_true",                      help="Display buffer on screen (require video server)"                       )
    parser.add_argument(      "--random",       action="store_true",                      help="Generate random buffer and do not use the interface (useful for debug)")
    args = parser.parse_args()

    App(args).run()