"""
Author: Bin Cao (bcao686@connect.hkust-gz.edu.cn)
Affiliations:
- Hong Kong University of Science and Technology (Guangzhou)
- City University of Hong Kong

Description: Best Model Checkpoint Management Module for Prdnet.
This module implements automatic saving and updating of model parameters based on
validation performance. It tracks the best validation MAE and manages model checkpoints.

Key Features:
- Track best validation MAE across all epochs
- Save model checkpoints when new best validation is achieved
- Automatic test set evaluation for best validation models
- Strategic test evaluation scheduling (best-val-triggered vs 30-epoch-interval)
- Comprehensive checkpoint metadata management
"""

import os
import torch
import json
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import time


class BestModelTracker:
    """Track and manage best validation model checkpoints."""
    
    def __init__(self, output_dir: str, save_top_k: int = 3, 
                 monitor_metric: str = "mae", mode: str = "min"):
        """
        Initialize best model tracker.
        
        Args:
            output_dir: Directory to save checkpoints
            save_top_k: Number of best models to keep
            monitor_metric: Metric to monitor for best model selection
            mode: "min" for metrics where lower is better, "max" for higher is better
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        # Initialize tracking variables
        self.best_metric_value = float('inf') if mode == "min" else float('-inf')
        self.best_epoch = -1
        self.best_model_path = None
        self.checkpoint_history = []
        
        # Test evaluation tracking
        self.last_test_epoch = -1
        self.test_evaluation_log = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load existing checkpoint history if available
        self._load_checkpoint_history()
    
    def _load_checkpoint_history(self):
        """Load existing checkpoint history from file."""
        history_file = self.output_dir / "checkpoint_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.checkpoint_history = data.get('checkpoints', [])
                    self.best_metric_value = data.get('best_metric_value', 
                                                    float('inf') if self.mode == "min" else float('-inf'))
                    self.best_epoch = data.get('best_epoch', -1)
                    self.best_model_path = data.get('best_model_path', None)
                    self.test_evaluation_log = data.get('test_evaluation_log', [])
                    self.last_test_epoch = data.get('last_test_epoch', -1)
                    
                    self.logger.info(f"Loaded checkpoint history. Best {self.monitor_metric}: "
                                   f"{self.best_metric_value} at epoch {self.best_epoch}")
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint history: {e}")
    
    def _save_checkpoint_history(self):
        """Save checkpoint history to file."""
        history_file = self.output_dir / "checkpoint_history.json"
        data = {
            'checkpoints': self.checkpoint_history,
            'best_metric_value': self.best_metric_value,
            'best_epoch': self.best_epoch,
            'best_model_path': self.best_model_path,
            'test_evaluation_log': self.test_evaluation_log,
            'last_test_epoch': self.last_test_epoch,
            'monitor_metric': self.monitor_metric,
            'mode': self.mode,
            'save_top_k': self.save_top_k
        }
        
        try:
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save checkpoint history: {e}")
    
    def is_better_metric(self, current_value: float) -> bool:
        """Check if current metric value is better than the best so far."""
        if self.mode == "min":
            return current_value < self.best_metric_value
        else:
            return current_value > self.best_metric_value
    
    def should_evaluate_test(self, epoch: int, is_new_best: bool) -> Tuple[bool, str]:
        """
        Determine if test set should be evaluated and why.
        
        Args:
            epoch: Current epoch number
            is_new_best: Whether this epoch achieved a new best validation metric
        
        Returns:
            Tuple of (should_evaluate, reason)
        """
        # Always evaluate test set when new best validation is achieved
        if is_new_best:
            return True, "best-val-triggered"
        
        # Evaluate every 30 epochs
        if epoch % 30 == 0 and epoch > 0:
            return True, "30-epoch-interval"
        
        return False, "no-evaluation"
    
    def update_and_save(self, epoch: int, model: torch.nn.Module, 
                       optimizer: torch.optim.Optimizer, 
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       metrics: Dict[str, float], 
                       additional_info: Optional[Dict[str, Any]] = None) -> Tuple[bool, bool, str]:
        """
        Update best model tracking and save checkpoint if needed.
        
        Args:
            epoch: Current epoch number
            model: Model to potentially save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            metrics: Dictionary of validation metrics
            additional_info: Additional information to save with checkpoint
        
        Returns:
            Tuple of (is_new_best, should_evaluate_test, test_reason)
        """
        current_metric = metrics.get(self.monitor_metric)
        if current_metric is None:
            self.logger.warning(f"Metric '{self.monitor_metric}' not found in metrics")
            return False, False, "no-evaluation"
        
        is_new_best = self.is_better_metric(current_metric)
        should_eval_test, test_reason = self.should_evaluate_test(epoch, is_new_best)
        
        if is_new_best:
            # Update best metric tracking
            self.best_metric_value = current_metric
            self.best_epoch = epoch
            
            # Save new best model
            checkpoint_name = f"best_model_epoch_{epoch}_{self.monitor_metric}_{current_metric:.6f}.pt"
            checkpoint_path = self.output_dir / checkpoint_name
            
            # Prepare checkpoint data
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'metrics': metrics,
                'best_metric_value': self.best_metric_value,
                'monitor_metric': self.monitor_metric,
                'timestamp': time.time(),
                'additional_info': additional_info or {}
            }
            
            # Save checkpoint
            try:
                torch.save(checkpoint_data, checkpoint_path)
                self.best_model_path = str(checkpoint_path)
                
                # Update checkpoint history
                checkpoint_info = {
                    'epoch': epoch,
                    'path': str(checkpoint_path),
                    'metric_value': current_metric,
                    'metrics': metrics,
                    'timestamp': time.time()
                }
                self.checkpoint_history.append(checkpoint_info)
                
                # Keep only top-k checkpoints
                self._cleanup_old_checkpoints()
                
                # Save updated history
                self._save_checkpoint_history()
                
                self.logger.info(f"New best {self.monitor_metric}: {current_metric:.6f} "
                               f"at epoch {epoch}. Checkpoint saved: {checkpoint_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to save checkpoint: {e}")
                return False, should_eval_test, test_reason
        
        return is_new_best, should_eval_test, test_reason
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only top-k best models."""
        if len(self.checkpoint_history) <= self.save_top_k:
            return
        
        # Sort checkpoints by metric value
        if self.mode == "min":
            sorted_checkpoints = sorted(self.checkpoint_history, 
                                      key=lambda x: x['metric_value'])
        else:
            sorted_checkpoints = sorted(self.checkpoint_history, 
                                      key=lambda x: x['metric_value'], reverse=True)
        
        # Keep only top-k
        checkpoints_to_keep = sorted_checkpoints[:self.save_top_k]
        checkpoints_to_remove = sorted_checkpoints[self.save_top_k:]
        
        # Remove old checkpoint files
        for checkpoint in checkpoints_to_remove:
            try:
                checkpoint_path = Path(checkpoint['path'])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    self.logger.info(f"Removed old checkpoint: {checkpoint_path.name}")
            except Exception as e:
                self.logger.warning(f"Could not remove checkpoint {checkpoint['path']}: {e}")
        
        # Update checkpoint history
        self.checkpoint_history = checkpoints_to_keep
    
    def log_test_evaluation(self, epoch: int, test_metrics: Dict[str, float], 
                           reason: str):
        """Log test set evaluation results."""
        test_log_entry = {
            'epoch': epoch,
            'test_metrics': test_metrics,
            'reason': reason,
            'timestamp': time.time()
        }
        
        self.test_evaluation_log.append(test_log_entry)
        self.last_test_epoch = epoch
        
        # Save updated history
        self._save_checkpoint_history()
        
        test_mae = test_metrics.get('mae', 'N/A')
        self.logger.info(f"Test evaluation at epoch {epoch} ({reason}): "
                        f"Test MAE = {test_mae}")
    
    def load_best_model(self, model: torch.nn.Module, 
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Dict[str, Any]:
        """
        Load the best model checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
        
        Returns:
            Dictionary with checkpoint information
        """
        if self.best_model_path is None or not Path(self.best_model_path).exists():
            raise FileNotFoundError(f"Best model checkpoint not found: {self.best_model_path}")
        
        try:
            checkpoint = torch.load(self.best_model_path, map_location='cpu')
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state if provided
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if provided
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                if checkpoint['scheduler_state_dict'] is not None:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.logger.info(f"Loaded best model from epoch {checkpoint['epoch']} "
                           f"with {self.monitor_metric}: {checkpoint['best_metric_value']:.6f}")
            
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"Failed to load best model: {e}")
            raise
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of checkpoint tracking."""
        return {
            'best_metric_value': self.best_metric_value,
            'best_epoch': self.best_epoch,
            'best_model_path': self.best_model_path,
            'total_checkpoints': len(self.checkpoint_history),
            'test_evaluations': len(self.test_evaluation_log),
            'last_test_epoch': self.last_test_epoch,
            'monitor_metric': self.monitor_metric,
            'mode': self.mode
        }
