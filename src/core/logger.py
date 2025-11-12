"""
Logging and Experiment Tracking Module.

This module provides a unified logging infrastructure for machine learning experiments,
combining traditional file/console logging with TensorBoard visualization capabilities.

The primary class, LogManager, handles:
- Standard output and file-based logging with structured formatting
- TensorBoard integration for metric visualization and experiment tracking
- Experiment metadata logging (configuration, dataset statistics)
- Training/validation metric logging with automatic TensorBoard synchronization
- Statistical analysis of metric differences during training

Typical Usage:
    ```python
    from pathlib import Path
    
    log_manager = LogManager(
        exp_name="my_experiment",
        exp_desc="Training a segmentation model",
        file_logs_dir_path=Path("logs/experiments"),
        tb_logs_dir_path=Path("logs_tb/experiments")
    )
    
    # Log experiment intro
    log_manager.log_intro(config, train_ds, val_ds, train_dl, val_dl)
    
    # Log training metrics
    log_manager.log_scores(
        title="Epoch 1",
        loss=0.345,
        metrics_score={"iou": 0.78, "dice": 0.82},
        tb_log_counter=1,
        tb_phase="train"
    )
    
    # Clean up
    log_manager.close_loggers()
    ```

Dependencies:
    - torch: For tensor operations and data loading
    - torchmetrics: For metric computation and tracking
    - tensorboard: For experiment visualization
"""

from core.config import *
from core.viz import format_to_title, pretty_metrics
from core.torch_utils import nanstd

from pathlib import Path
import logging
from logging import Logger
import os
from torch.utils.tensorboard import SummaryWriter
import torchmetrics as tm
from typing import Literal, Optional
from torch.utils.data import Dataset, DataLoader
import torch
import re

from core._types import Any

class LogManager():
    """
    A comprehensive logging manager for experiment tracking and monitoring.
    
    This class provides unified logging capabilities to both standard output/file and TensorBoard,
    enabling consistent experiment tracking across different logging backends.
    
    Attributes:
        exp_name (str): The name of the experiment.
        exp_desc (Optional[str]): A description of the experiment.
        main_logger (Logger): Logger instance for stdout and file logging.
        tb_logger (Optional[SummaryWriter]): TensorBoard logger instance for metric visualization.
    """
    
    def __init__(
            self,
            exp_name: str,
            exp_desc: Optional[str] = None,
            file_logs_dir_path: Optional[Path] = None,
            tb_logs_dir_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize the LogManager with experiment details and logging paths.
        
        Args:
            exp_name: The name of the experiment, used for log file naming and identification.
            exp_desc: Optional description of the experiment for documentation purposes.
            file_logs_dir_path: Optional directory path where log files will be saved.
                If None, logs only to stdout.
            tb_logs_dir_path: Optional directory path for TensorBoard logs.
                If None, TensorBoard logging is disabled.
        """
        
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
        """
        Create and configure a logger for stdout and optional file logging.
        
        Creates a logger with formatted output that includes timestamp, log level,
        filename, line number, and message. Prevents duplicate handlers if called
        multiple times.
        
        Args:
            file_logs_dir_path: Optional directory path for saving log files.
                If provided, logs will be written to a file named "{exp_name}.log"
                in this directory.
        
        Returns:
            A configured Logger instance with appropriate handlers.
        """
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
        """
        Create and configure a TensorBoard SummaryWriter.
        
        Creates a TensorBoard logger in a subdirectory named after the experiment
        within the provided TensorBoard logs directory.
        
        Args:
            tb_logs_dir_path: Directory path where TensorBoard logs will be stored.
        
        Returns:
            A configured SummaryWriter instance for TensorBoard logging.
        """
        tb_exp_dir = os.path.join(tb_logs_dir_path, self.exp_name)
        os.makedirs(tb_exp_dir, exist_ok=True)
        return SummaryWriter(log_dir=tb_exp_dir)
    
    def log_title(
            self,
            text: str,
            pad_symbol: str = '-'
    ) -> None:
        """
        Log a formatted title with padding symbols for visual separation.
        
        Args:
            text: The title text to be logged.
            pad_symbol: Character used for padding around the title (default: '-').
        """
        _stacklevel = 2
        self.main_logger.info(format_to_title(text, pad_symbol=pad_symbol), stacklevel=_stacklevel)

    def log_line(
            self,
            text: str
    ) -> None:
        """
        Log a single line of text to the main logger.
        
        Args:
            text: The text to be logged.
        """
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
        """
        Log the introductory information for an experiment.
        
        Logs experiment name, description, configuration, and dataset/dataloader
        statistics in a formatted structure. This provides a comprehensive overview
        at the start of an experiment.
        
        Args:
            config: Dictionary containing experiment configuration parameters.
            train_ds: Training dataset instance.
            val_ds: Validation dataset instance.
            train_dl: Training dataloader instance.
            val_dl: Validation dataloader instance.
        """
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
        """
        Log loss and metrics scores to both main logger and TensorBoard.
        
        Formats and logs training/validation metrics including loss and additional
        evaluation metrics. Simultaneously logs to TensorBoard for visualization
        if TensorBoard logger is available.
        
        Args:
            title: Title or identifier for this logging entry.
            loss: The loss value to be logged.
            metrics_score: Optional dictionary of metric names to torchmetrics.Metric
                objects containing computed scores.
            tb_log_counter: Optional step/iteration counter for TensorBoard x-axis.
            tb_phase: Phase identifier for TensorBoard logging ("train" or "val").
            suffix: Optional suffix string to append to the log message.
            metrics_prefix: Optional prefix to prepend to metric names in the log output.
        """
        _stacklevel = 2
        log_str = f"[{title}] {metrics_prefix}loss: {loss:.4f}"
        self.tb_logger.add_scalar(f"{tb_phase}/loss", loss, tb_log_counter) if self.tb_logger else None

        if metrics_score:
            for m, s in pretty_metrics(metrics_score).items():
                log_str += f", {metrics_prefix}{m}: {s}"
                self.tb_logger.add_scalar(f"{tb_phase}/{m}", s, tb_log_counter)  if self.tb_logger else None
        log_str += suffix if suffix else ""
        self.main_logger.info(log_str, stacklevel=_stacklevel)

    def log_metric_diffs(
            self,
            title: str,
            metric_diffs: dict[str, Optional[float]],
            tb_log_counter: Optional[int],
    ) -> None:
        """
        Log statistical analysis of metric differences.
        
        Computes and logs the mean and standard deviation of metric differences,
        useful for tracking training dynamics and convergence patterns. Handles
        NaN values gracefully using nanmean and nanstd.
        
        Args:
            title: Title or identifier for this logging entry.
            metric_diffs: Dictionary mapping metric names to their difference values.
                Values can be None/NaN for missing data.
            tb_log_counter: Optional step/iteration counter for TensorBoard logging.
        """
        metric_diffs_values = torch.Tensor(list(metric_diffs.values()))
        mean = metric_diffs_values.nanmean()
        std = nanstd(metric_diffs_values, dim=0)
        log_str = f"[{title}] train_metric_diff_mean: {mean:.4f}, train_metric_diff_std: {std:.4f}"
        self.log_line(log_str)
        self.tb_logger.add_scalar(f"train/metric_diff_mean", mean, tb_log_counter) if self.tb_logger else None
        self.tb_logger.add_scalar(f"train/metric_diff_std", std, tb_log_counter) if self.tb_logger else None

    def close_loggers(self) -> None:
        """
        Properly close and cleanup all logger resources.
        
        Shuts down the main logging system and closes the TensorBoard writer
        to ensure all logs are flushed and resources are released properly.
        Should be called at the end of an experiment run.
        """
        
        # close StdOut and file loggers.
        logging.shutdown()

        # close TensorBoard loggers
        self.tb_logger.close() if self.tb_logger else None

def parse_train_logs_metric(
        filepath: Path,
        metric_name: str
) -> dict:
    """
    Parses a log file to extract batch metrics for each epoch and step.

    Args:
        filepath: The path to the .log file.

    Returns:
        A nested dictionary where the outer keys are epoch numbers and the
        inner keys are step numbers, with train metric as the value.
        Example: {1: {5: 0.9739, 10: 0.9783}, 2: {...}}
    """
    # Regex to capture epoch, step, and metric
    log_pattern = re.compile(fr"\[epoch: (\d+)/\d+, step: (\d+)/\d+ .*?{metric_name}: ([\d.]+)")

    metrics = {}

    try:
        with open(filepath, 'r') as f:
            for line in f:
                match = log_pattern.search(line)
                if match:
                    # Extract captured groups
                    epoch = int(match.group(1))
                    step = int(match.group(2))
                    batch_acc = float(match.group(3))

                    # Populate the nested dictionary
                    if epoch not in metrics:
                        metrics[epoch] = {}
                    
                    metrics[epoch][step] = batch_acc
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return {}
        
    return metrics

def parse_val_logs_metric(
        filepath: Path,
        metric_name: str
) -> dict:
    """
    Parses a log file to extract validation accuracies for each epoch.

    It handles two cases:
    1. The initial validation before training starts, assigned to epoch 0.
    2. End-of-epoch validations, assigned to their corresponding epoch number.

    Args:
        filepath: The path to the .log file.

    Returns:
        A dictionary mapping epoch number to validation accuracy.
        Example: {0: 0.8989, 1: 0.9150, 2: 0.9230}
    """
    # Pattern for the initial validation line (before epoch 1)
    initial_val_pattern = re.compile(
        fr"\[Before any weight update, VALIDATION\].*?{metric_name}: ([\d.]+)"
    )
    
    # Pattern for end-of-epoch validation lines (assumes a similar format)
    epoch_val_pattern = re.compile(
        fr"\[epoch: (\d+)/\d+.*?VALIDATION\].*?{metric_name}: ([\d.]+)"
    )

    val_accuracies = {}

    try:
        with open(filepath, 'r') as f:
            for line in f:
                # First, check for the initial validation line
                initial_match = initial_val_pattern.search(line)
                if initial_match:
                    val_acc = float(initial_match.group(1))
                    val_accuracies[0] = val_acc
                    continue # Move to the next line

                # If not initial, check for a standard end-of-epoch validation line
                epoch_match = epoch_val_pattern.search(line)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    val_acc = float(epoch_match.group(2))
                    val_accuracies[epoch] = val_acc

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return {}
        
    return val_accuracies
