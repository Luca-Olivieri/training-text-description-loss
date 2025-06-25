from pathlib import Path
import logging
import os
import tensorboard as tb
from tensorboard import SummaryWriter

def get_tb_logger(
        tb_dir: Path,
        exp_name: str
) -> SummaryWriter:
    tb_exp_dir = os.path.join(tb_dir, exp_name)
    os.makedirs(tb_exp_dir, exist_ok=True)
    return SummaryWriter(log_dir=tb_exp_dir)
