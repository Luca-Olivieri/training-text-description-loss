from config import *
from utils import pretty_metrics

from pathlib import Path
import logging
import os
from torch.utils.tensorboard import SummaryWriter
import torchmetrics as tm
from typing import Literal, Optional

def get_logger(
        log_dir: Path,
        exp_name: str
) -> logging.Logger:
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d] %(message)s"
    
    logger_name = "main_logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # log to StdOut
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    # log to file
    os.makedirs(log_dir, exist_ok=True) 
    log_filename = os.path.join(log_dir, exp_name+".log")
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)

    return logger

def get_tb_logger(
        tb_dir: Path,
        exp_name: str
) -> SummaryWriter:
    tb_exp_dir = os.path.join(tb_dir, exp_name)
    os.makedirs(tb_exp_dir, exist_ok=True)
    return SummaryWriter(log_dir=tb_exp_dir)

logger = get_logger(CONFIG["log_dir"], CONFIG["exp_name"])
tb_writer = get_tb_logger(CONFIG["tb_dir"], CONFIG["exp_name"])

def log_segnet_scores(
          title: str,
          loss: float,
          metrics_score: dict[dict, tm.Metric],
          tb_log_counter: Optional[int],
          tb_phase: Optional[Literal["train", "val"]],
          suffix: Optional[str] = None,
          metrics_prefix: Optional[str] = None
) -> None:
    log_str = f"[{title}] {metrics_prefix}loss: {loss:.4f}"
    tb_writer.add_scalar(f"{tb_phase}/loss", loss, tb_log_counter) if tb_log_counter is not None else None
    for m, s in pretty_metrics(metrics_score).items():
            log_str += f", {metrics_prefix}{m}: {s}"
            tb_writer.add_scalar(f"{tb_phase}/{m}", s, tb_log_counter)  if tb_log_counter is not None else None
    log_str += suffix if suffix is not None else ""
    logger.info(log_str)
