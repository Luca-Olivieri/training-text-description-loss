from pathlib import Path
import logging
import os
import tensorboard as tb
from torch.utils.tensorboard import SummaryWriter

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
