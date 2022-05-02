from __future__ import annotations

from pathlib import Path
from sklearn.metrics import (classification_report, confusion_matrix)
from socialtouch.config import Config
from socialtouch.data import (RandomFlipX, RandomFlipY, RandomReverse, RandomScale, ToTensor)
from time import time
from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import Module
from torch.nn.functional import cross_entropy
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchbooster.dataset import Split
from torchbooster.metrics import (accuracy, RunningAverage)
from torchbooster.scheduler import BaseScheduler
from torchbooster.utils import iter_loader
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torchinfo
import torchbooster.distributed as dist
import torchbooster.utils as utils
import torchvision.transforms as T


def test(conf: Config, model: Module, loader: DataLoader) -> None:
    model.eval()
    
    run_acc = RunningAverage()
    predictions, targets = [], []
    with tqdm(loader, desc="Test") as pbar:
        for X, labels in pbar:
            X, labels = conf.env.make(X, labels)
            
            with torch.no_grad(), autocast(conf.env.fp16):
                logits = model(X)
                acc = accuracy(logits, labels)
            
            run_acc.update(acc.item())
            pbar.set_postfix(acc=f"{run_acc.value * 100:.2f}%")

            predictions += logits.argmax(dim=1).cpu().numpy().tolist()
            targets += labels.cpu().numpy().tolist()
    
    names = list(map(str, range(loader.dataset.n_class)))
    cf = confusion_matrix(targets, predictions)
    cf = pd.DataFrame(cf / np.max(cf), index=range(loader.dataset.n_class), columns=names)
    report = classification_report(targets, predictions, target_names=names, digits=4)

    plt.figure(figsize=(10, 8))
    sn.heatmap(cf, annot=False)
    plt.savefig(f"res/cf_{conf.model.name}_{conf.dataset.window_size}_{conf.dataset.hop_size}.png")
    plt.close("all")

    with open("res/benchmark.csv", "a") as fp:
        fp.write(f"\n{conf.model.name};{conf.dataset.window_size};{conf.dataset.hop_size};{run_acc.value * 100:.2f}")
    
    with open(f"res/{conf.model.name}_{conf.dataset.window_size}_{conf.dataset.hop_size}.txt", "w") as fp:
        fp.write(str(report))

    torch.save({"conf": conf, "model": model.state_dict()}, f"res/{conf.model.name}_{conf.dataset.window_size}_{conf.dataset.hop_size}.pt")


def fit(
    conf: Config,
    model: Module,
    optim: Optimizer,
    scheduler: BaseScheduler,
    scaler: GradScaler,
    loader: DataLoader,
) -> None:
    model.train()

    loader = iter_loader(loader)
    with tqdm(range(conf.n_iter), desc="Train") as pbar:
        run_loss, run_acc = RunningAverage(), RunningAverage()
        for step in pbar:
            epoch, (X, labels) = next(loader)
            X, labels = conf.env.make(X, labels)
                
            with autocast(conf.env.fp16):
                logits = model(X)
                loss = cross_entropy(logits, labels)
                acc = accuracy(logits, labels)

            utils.step(loss, optim, scheduler=scheduler, scaler=scaler)

            run_loss.update(loss.item())
            run_acc.update(acc.item())
            pbar.set_postfix(
                epoch=f"{epoch}",
                loss=f"{run_loss.value:.2e}",
                acc=f"{run_acc.value * 100:.2f}%",
            )


def benchmark(conf: Config, model: Module, shape: tuple, n: int) -> float:
    x = conf.env.make(torch.zeros(shape))
    dts = np.zeros((n, ))
    for i in range(n):
        start = time()
        torch.softmax(model(x), dim=1)
        dts[i] = time() - start
    return dts.min()


def main(conf: Config) -> None:
    train_transform = T.Compose([RandomReverse, RandomFlipY, RandomFlipX, ToTensor, RandomScale])
    train_set = conf.dataset.make(Split.TRAIN, train_transform)
    train_loader = conf.loader.make(train_set, shuffle=True)

    test_transform = ToTensor
    test_set = conf.dataset.make(Split.TEST, test_transform)
    test_loader = conf.loader.make(test_set, shuffle=False)

    shape = (1, conf.dataset.window_size, 1, conf.dataset.height, conf.dataset.width)
    model = conf.env.make(conf.model.make(
        conf.dataset.height,
        conf.dataset.width,
        train_set.n_class,
        conf.dataset.window_size,
    ))

    torchinfo.summary(model, shape)
    dt = benchmark(conf, model, shape, 10)
    print(f"Model's Perormances: {dt * 1000:.2f} ms | {1 / dt:.2f} FPS")

    optim = conf.optim.make(model.parameters())
    scheduler = conf.scheduler.make(optim)
    scaler = GradScaler(enabled=conf.env.fp16)

    fit(conf, model, optim, scheduler, scaler, train_loader)
    test(conf, model, test_loader)


if __name__ == "__main__":
    models       = ["rnet", "rcnet", "racnet", "tacnet"]
    window_sizes = [6, 12, 24, 36, 48, 60]
    hop_sizes    = [1, 3]
    
    for model in models:
        conf = Config.load(Path("configs", f"{model}.yml"))
        
        utils.seed(conf.seed)
        utils.boost(enable=True)
        
        for window_size in window_sizes:
            conf.dataset.window_size = window_size

            for hop_size in hop_sizes:
                conf.dataset.hop_size = hop_size

                dist.launch(
                    main,
                    conf.env.n_gpu,
                    conf.env.n_machine,
                    conf.env.machine_rank,
                    conf.env.dist_url,
                    args=(conf, )
                )

    df = pd.read_csv("res/benchmark.csv", sep=";")
    sn.set_theme(style="darkgrid")
    plt.figure(figsize=(12, 8))
    sn.lineplot(data=df[df.hop_size == 1], x="window_size", y="acc", hue="model")
    plt.savefig("window_size.png")
    plt.close("all")