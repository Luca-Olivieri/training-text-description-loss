from config import *
from utils import pretty_metrics
from viz import format_to_title

from pathlib import Path
import logging
from logging import Logger
import os
from torch.utils.tensorboard import SummaryWriter
import torchmetrics as tm
from typing import Literal, Optional
from torch.utils.data import Dataset, DataLoader

def get_logger(
        log_dir: Optional[Path],
        exp_name: str
) -> logging.Logger:
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d] %(message)s"
    
    logger_name = "main_logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    logger.propagate = False
    
    # log to StdOut
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    # log to file
    if log_dir:
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

# logger = get_logger(CONFIG["log_dir"], CONFIG["exp_name"])
# tb_writer = get_tb_logger(CONFIG["tb_dir"], CONFIG["exp_name"])

def log_intro(
        logger: Logger,
        exp_name: str,
        exp_desc: Optional[str],
        config: dict[str, Any],
        train_ds: Dataset,
        val_ds: Dataset,
        train_dl: DataLoader,
        val_dl: DataLoader
) -> None:
    _stacklevel = 2
    logger.info(format_to_title(exp_name, pad_symbol='='), stacklevel=_stacklevel)
    logger.info(exp_desc, stacklevel=_stacklevel) if exp_desc is not None else None
    logger.info(format_to_title("Config"), stacklevel=_stacklevel)
    logger.info(config, stacklevel=_stacklevel)
    logger.info(format_to_title("Data"), stacklevel=_stacklevel)
    logger.info(f"- Training data: {len(train_ds)} samples, in {len(train_dl)} mini-batches of size {train_dl.batch_size}", stacklevel=_stacklevel)
    logger.info(f"- Validation data: {len(val_ds)} samples, in {len(val_dl)} mini-batches of size {val_dl.batch_size}", stacklevel=_stacklevel)

def log_scores(
        logger: Logger,
        tb_writer: SummaryWriter,
        title: str,
        loss: float,
        metrics_score: Optional[dict[dict, tm.Metric]],
        tb_log_counter: Optional[int],
        tb_phase: Optional[Literal["train", "val"]],
        suffix: Optional[str] = None,
        metrics_prefix: Optional[str] = None
) -> None:
    _stacklevel = 2
    log_str = f"[{title}] {metrics_prefix}loss: {loss:.4f}"
    tb_writer.add_scalar(f"{tb_phase}/loss", loss, tb_log_counter) if tb_log_counter is not None and tb_writer is not None else None

    if metrics_score:
        for m, s in pretty_metrics(metrics_score).items():
                log_str += f", {metrics_prefix}{m}: {s}"
                tb_writer.add_scalar(f"{tb_phase}/{m}", s, tb_log_counter)  if tb_log_counter is not None else None
    log_str += suffix if suffix is not None else ""
    logger.info(log_str, stacklevel=_stacklevel)

def log_title(
        logger: Logger,
        text: str,
        pad_symbol: str = '-'
) -> None:
    _stacklevel = 2
    logger.info(format_to_title(text, pad_symbol=pad_symbol), stacklevel=_stacklevel)
