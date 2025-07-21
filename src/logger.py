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

class LogManager():
    
    def __init__(
            self,
            exp_name: str,
            exp_desc: Optional[str] = None,
            file_logs_dir_path: Optional[Path] = None,
            tb_logs_dir_path: Optional[Path] = None,
    ) -> None:
        
        self.exp_name = exp_name
        self.exp_desc = exp_desc if exp_desc else None
        
        # logs to StdOut and (optionally) to .log file.
        self.main_logger: Logger = self.get_logger(
            file_logs_dir_path=file_logs_dir_path,
        )

        # logs to TensorBoad
        if tb_logs_dir_path:
            self.tb_logger: SummaryWriter = self.get_tb_logger(
                tb_logs_dir_path=tb_logs_dir_path,
            )
        else:
            self.tb_logger = None

    def get_logger(
            self,
            file_logs_dir_path: Optional[Path],
    ) -> logging.Logger:
        fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d] %(message)s"
        
        logger_name = "main_logger"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        # prevent messages from being propagated to the root logger
        logger.propagate = False

        # avoid adding duplicate handlers if get_logger is called multiple times
        if logger.hasHandlers():
            return logger
        
        # log to StdOut
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

        # log to file
        if file_logs_dir_path:
            os.makedirs(file_logs_dir_path, exist_ok=True)
            log_filename = os.path.join(file_logs_dir_path, self.exp_name+".log")
            file_handler = logging.FileHandler(log_filename)
            file_handler.setFormatter(logging.Formatter(fmt))
            logger.addHandler(file_handler)

        return logger

    def get_tb_logger(
            self,
            tb_logs_dir_path: Path,
    ) -> SummaryWriter:
        tb_exp_dir = os.path.join(tb_logs_dir_path, self.exp_name)
        os.makedirs(tb_exp_dir, exist_ok=True)
        return SummaryWriter(log_dir=tb_exp_dir)
    
    def log_title(
            self,
            text: str,
            pad_symbol: str = '-'
    ) -> None:
        _stacklevel = 2
        self.main_logger.info(format_to_title(text, pad_symbol=pad_symbol), stacklevel=_stacklevel)

    def log_line(
            self,
            text: str
    ) -> None:
        _stacklevel = 2
        self.main_logger.info(text, stacklevel=_stacklevel)

    def log_intro(
            self,
            config: dict[str, Any],
            train_ds: Dataset,
            val_ds: Dataset,
            train_dl: DataLoader,
            val_dl: DataLoader
    ) -> None:
        _stacklevel = 2
        self.main_logger.info(format_to_title(self.exp_name, pad_symbol='='), stacklevel=_stacklevel)
        self.main_logger.info(self.exp_desc, stacklevel=_stacklevel) if self.exp_desc else None
        self.main_logger.info(format_to_title("Config"), stacklevel=_stacklevel)
        self.main_logger.info(config, stacklevel=_stacklevel)
        self.main_logger.info(format_to_title("Data"), stacklevel=_stacklevel)
        self.main_logger.info(f"- Training data: {len(train_ds)} samples, in {len(train_dl)} mini-batches of size {train_dl.batch_size}", stacklevel=_stacklevel)
        self.main_logger.info(f"- Validation data: {len(val_ds)} samples, in {len(val_dl)} mini-batches of size {val_dl.batch_size}", stacklevel=_stacklevel)

    def log_scores(
            self,
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
        self.tb_logger.add_scalar(f"{tb_phase}/loss", loss, tb_log_counter) if self.tb_logger else None

        if metrics_score:
            for m, s in pretty_metrics(metrics_score).items():
                    log_str += f", {metrics_prefix}{m}: {s}"
                    self.tb_logger.add_scalar(f"{tb_phase}/{m}", s, tb_log_counter)  if self.tb_logger else None
        log_str += suffix if suffix else ""
        self.main_logger.info(log_str, stacklevel=_stacklevel)


    def close_loggers(self) -> None:
        
        # close StdOut and file loggers.
        logging.shutdown()

        # close TensorBoard loggers
        self.tb_logger.close() if self.tb_logger else None
