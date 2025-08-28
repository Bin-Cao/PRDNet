"""
Author: Bin Cao (bcao686@connect.hkust-gz.edu.cn)
Affiliations:
- Hong Kong University of Science and Technology (Guangzhou)
- City University of Hong Kong

Description: Main training module for Prdnet models.
"""

from functools import partial
from typing import Any, Dict, Union

import ignite
import torch

from ignite.contrib.handlers import TensorboardLogger
try:
    from ignite.contrib.handlers.stores import EpochOutputStore
except Exception as exp:
    from ignite.handlers.stores import EpochOutputStore
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.tensorboard_logger import (
    global_step_from_engine,
)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.contrib.metrics import ROC_AUC, RocCurve
from ignite.metrics import (
    Accuracy,
    Precision,
    Recall,
    ConfusionMatrix,
)
import pickle as pk
import numpy as np
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError
from torch import nn

from prdnet.data import get_train_val_loaders
from prdnet.config import TrainingConfig
from prdnet.model import Prdnet
from prdnet.checkpoint import BestModelTracker

from jarvis.db.jsonutils import dumpjson
import json
import pprint

import os


# torch config
torch.set_default_dtype(torch.float32)

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def activated_output_transform(output):
    """Exponentiate output."""
    y_pred, y = output
    y_pred = torch.exp(y_pred)
    y_pred = y_pred[:, 1]
    return y_pred, y


def make_standard_scalar_and_pca(output):
    """Use standard scalar and PCS for multi-output data."""
    sc = pk.load(open(os.path.join(tmp_output_dir, "sc.pkl"), "rb"))
    y_pred, y = output
    y_pred = torch.tensor(sc.transform(y_pred.cpu().numpy()), device=device)
    y = torch.tensor(sc.transform(y.cpu().numpy()), device=device)
    return y_pred, y


def thresholded_output_transform(output):
    """Round off output."""
    y_pred, y = output
    y_pred = torch.round(torch.exp(y_pred))
    # print ('output',y_pred)
    return y_pred, y


def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def setup_optimizer(params, config: TrainingConfig):
    """Set up optimizer for param groups with advanced optimizers."""
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
    elif config.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.9,
            alpha=0.99,
        )
    else:
        # Default to AdamW
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    return optimizer


def train_dgl(
    config: Union[TrainingConfig, Dict[str, Any]],
    model: nn.Module = None,
    train_val_test_loaders=[],
    test_only=False,
    use_save=True,
    mp_id_list=None,
):
    """
    `config` should conform to prdnet.conf.TrainingConfig, and
    if passed as a dict with matching keys, pydantic validation is used
    """
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)
            print('error in converting to training config!')
    import os
    
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    checkpoint_dir = os.path.join(config.output_dir)
    deterministic = False
    classification = False
    # Save configuration to file
    tmp = config.dict()
    f = open(os.path.join(config.output_dir, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()
    global tmp_output_dir
    tmp_output_dir = config.output_dir
    if config.classification_threshold is not None:
        classification = True
    if config.random_seed is not None:
        deterministic = True
        ignite.utils.manual_seed(config.random_seed)

    line_graph = True
    if not train_val_test_loaders:
        # use input standardization for all real-valued feature sets
        (
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
            mean_train,
            std_train,
        ) = get_train_val_loaders(
            dataset=config.dataset,
            target=config.target,
            n_train=config.n_train,
            n_val=config.n_val,
            n_test=config.n_test,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            batch_size=config.batch_size,
            atom_features=config.atom_features,
            neighbor_strategy=config.neighbor_strategy,
            standardize=config.atom_features != "cgcnn",
            line_graph=line_graph,
            id_tag=config.id_tag,
            pin_memory=config.pin_memory,
            workers=config.num_workers,
            save_dataloader=config.save_dataloader,
            use_canonize=config.use_canonize,
            filename=config.filename,
            cutoff=config.cutoff,
            max_neighbors=config.max_neighbors,
            output_features=config.model.output_features,
            classification_threshold=config.classification_threshold,
            target_multiplication_factor=config.target_multiplication_factor,
            standard_scalar_and_pca=config.standard_scalar_and_pca,
            keep_data_order=config.keep_data_order,
            output_dir=config.output_dir,
            matrix_input=config.matrix_input,
            pyg_input=config.pyg_input,
            use_lattice=config.use_lattice,
            use_angle=config.use_angle,
            use_save=use_save,
            mp_id_list=mp_id_list,
        )
    else:
        train_loader = train_val_test_loaders[0]
        val_loader = train_val_test_loaders[1]
        test_loader = train_val_test_loaders[2]
        prepare_batch = train_val_test_loaders[3]

        # Calculate mean and std from training data
        import torch
        all_targets = []
        for batch in train_loader:
            try:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    targets = batch[-1]  # Last element should be targets
                else:
                    # Try to extract targets from batch
                    targets = getattr(batch, 'y', None)
                    if targets is None:
                        targets = batch.get('target', batch.get('targets', None))

                if targets is not None:
                    if torch.is_tensor(targets):
                        targets_np = targets.cpu().numpy()
                        # Ensure we have a flat array of scalars
                        if targets_np.ndim > 1:
                            targets_np = targets_np.flatten()
                        all_targets.extend(targets_np.tolist())
                    else:
                        # Handle non-tensor targets
                        if hasattr(targets, '__iter__') and not isinstance(targets, str):
                            all_targets.extend([float(t) for t in targets])
                        else:
                            all_targets.append(float(targets))
            except Exception as e:
                print(f"Warning: Could not extract targets from batch: {e}")
                continue

        if all_targets:
            import numpy as np
            all_targets = np.array(all_targets, dtype=float)
            mean_train = float(np.mean(all_targets))
            std_train = float(np.std(all_targets))
            
        else:
            # Fallback values
            mean_train = 0.0
            std_train = 1.0
            print("Warning: Could not calculate statistics from training data, using defaults")
    prepare_batch = partial(prepare_batch, device=device)
    if classification:
        config.model.classification = True
    # define network, optimizer, scheduler
    _model = {
        "prdnet" : Prdnet,
    }
    if std_train is None:
        std_train = 1.0
    if model is None:
        net = _model.get(config.model.name)(config.model)
    else:
        net = model

    net.to(device)
    if config.distributed:
        import torch.distributed as dist

        # Use existing distributed setup if already initialized
        if not dist.is_initialized():
            # Fallback initialization if not already done
            rank = getattr(config, 'rank', 0)
            world_size = getattr(config, 'world_size', 1)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            dist.init_process_group("nccl", rank=rank, world_size=world_size)

        # Wrap model with DistributedDataParallel
        device_ids = [getattr(config, 'local_rank', 0)] if torch.cuda.is_available() else None
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=device_ids
        )
    params = group_decay(net)
    optimizer = setup_optimizer(params, config)

    # Enhanced learning rate scheduling
    if config.scheduler == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda _: 1.0
        )
    elif config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0,
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=50,  # More frequent steps
            gamma=0.8,     # More aggressive decay
        )
    elif config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.01,
        )
    elif config.scheduler == "cosine_warm":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,
            T_mult=2,
            eta_min=config.learning_rate * 0.01,
        )
    elif config.scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95,
        )
    elif config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
        )
    else:
        # Default to cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.01,
        )

    # Enhanced loss functions
    class HuberLoss(nn.Module):
        def __init__(self, delta=1.0):
            super().__init__()
            self.delta = delta

        def forward(self, input, target):
            residual = torch.abs(input - target)
            condition = residual < self.delta
            squared_loss = 0.5 * residual ** 2
            linear_loss = self.delta * residual - 0.5 * self.delta ** 2
            return torch.where(condition, squared_loss, linear_loss).mean()

    class SmoothL1Loss(nn.Module):
        def __init__(self, beta=1.0):
            super().__init__()
            self.beta = beta

        def forward(self, input, target):
            return nn.functional.smooth_l1_loss(input, target, beta=self.beta)

    class LogCoshLoss(nn.Module):
        def forward(self, input, target):
            diff = input - target
            return torch.mean(torch.log(torch.cosh(diff)))

    criteria = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "huber": HuberLoss(delta=1.0),
        "smooth_l1": SmoothL1Loss(beta=1.0),
        "log_cosh": LogCoshLoss(),
    }
    criterion = criteria.get(config.criterion, nn.MSELoss())
    # set up training engine and evaluators
    # Create custom MAE metric that properly denormalizes
    def denormalized_mae_transform(output):
        y_pred, y = output
        # Denormalize both predictions and targets
        y_pred_denorm = y_pred * std_train + mean_train
        y_denorm = y * std_train + mean_train
        return y_pred_denorm, y_denorm

    metrics = {
        "loss": Loss(criterion),
        "mae": MeanAbsoluteError(output_transform=denormalized_mae_transform),
        "neg_mae": -1.0 * MeanAbsoluteError(output_transform=denormalized_mae_transform)
    }
    # Create custom trainer that computes MAE during training
    def train_step(engine, batch):
        net.train()
        optimizer.zero_grad()

        # Prepare batch
        if prepare_batch:
            prepared_data = prepare_batch(batch, device=device, non_blocking=False)
            # prepare_batch returns (input_data, targets)
            input_data, y = prepared_data
        else:
            # Fallback for when prepare_batch is not provided
            input_data = batch
            y = batch.y if hasattr(batch, 'y') else batch[1]

        # Forward pass
        y_pred = net(input_data)

        # Compute loss
        loss = criterion(y_pred, y)

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        if hasattr(config, 'grad_clip_norm') and config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)

        optimizer.step()

        # Compute denormalized MAE for monitoring
        with torch.no_grad():
            y_pred_denorm = y_pred * std_train + mean_train
            y_denorm = y * std_train + mean_train
            mae = torch.mean(torch.abs(y_pred_denorm - y_denorm))

        return {
            'loss': loss.item(),
            'mae': mae.item(),
            'y_pred': y_pred,
            'y': y
        }

    from ignite.engine import Engine
    trainer = Engine(train_step)

    # Add event handlers to accumulate metrics during training
    @trainer.on(Events.EPOCH_STARTED)
    def reset_epoch_metrics(engine):
        engine.state.epoch_mae_sum = 0.0
        engine.state.epoch_loss_sum = 0.0
        engine.state.epoch_batch_count = 0

    @trainer.on(Events.ITERATION_COMPLETED)
    def accumulate_metrics(engine):
        output = engine.state.output
        engine.state.epoch_mae_sum += output['mae']
        engine.state.epoch_loss_sum += output['loss']
        engine.state.epoch_batch_count += 1
    evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )

    # Create test evaluator for periodic test set evaluation
    test_evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )

    # Initialize best model tracker for checkpoint management
    best_model_tracker = BestModelTracker(
        output_dir=config.output_dir,
        save_top_k=3,
        monitor_metric="mae",
        mode="min"
    )
    train_evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )
    if test_only:
        checkpoint_tmp = torch.load('/your_model_path.pt')
        to_load = {
            "model": net,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "trainer": trainer,
        }
        Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint_tmp)
        net.eval()
        targets = []
        predictions = []
        import time
        t1 = time.time()
        with torch.no_grad():
            for dat in test_loader:
                g, lg, _, target = dat
                try:
                    out_data = net([g.to(device), lg.to(device), _.to(device)])
                    success_flag=1
                except: # just in case
                    print('error for this data')
                    print(g)
                    success_flag=0
                if success_flag > 0:
                    out_data = out_data.cpu().numpy().tolist()
                    target = target.cpu().numpy().flatten().tolist()
                    if len(target) == 1:
                        target = target[0]
                    targets.append(target)
                    predictions.append(out_data)
        t2 = time.time()
        f.close()
        from sklearn.metrics import mean_absolute_error
        targets = np.array(targets) * std_train + mean_train
        predictions = np.array(predictions) * std_train + mean_train
        print("Test MAE:", mean_absolute_error(targets, predictions))
        print("Total test time:", t2-t1)
        return mean_absolute_error(targets, predictions)
    # ignite event handlers:
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    # apply learning rate scheduler
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )

    if config.write_checkpoint:
        # model checkpointing
        to_save = {
            "model": net,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "trainer": trainer,
        }
        handler = Checkpoint(
            to_save,
            DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=2,
            global_step_transform=lambda *_: trainer.state.epoch,
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)
        # evaluate save
        to_save = {"model": net}
        handler = Checkpoint(
            to_save,
            DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=5,
            filename_prefix='best',
            score_name="neg_mae",
            global_step_transform=lambda *_: trainer.state.epoch,
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)
    if config.progress:
        pbar = ProgressBar()
        # Show both loss and MAE in the progress bar
        def progress_transform(x):
            if isinstance(x, dict) and 'loss' in x and 'mae' in x:
                return {"loss": f"{x['loss']:.4f}", "mae": f"{x['mae']:.4f}"}
            else:
                return {"loss": f"{x:.4f}" if isinstance(x, (int, float)) else str(x)}

        pbar.attach(trainer, output_transform=progress_transform)

        # Also attach progress bar to evaluator to show validation metrics
        val_pbar = ProgressBar(desc="Validation")
        val_pbar.attach(evaluator, output_transform=lambda x: {"val_mae": f"{x['mae']:.4f}" if 'mae' in x else "N/A"})

    history = {
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
    }

    if config.store_outputs:
        # in history["EOS"]
        eos = EpochOutputStore()
        eos.attach(evaluator)
        train_eos = EpochOutputStore()
        train_eos.attach(train_evaluator)

    # collect evaluation performance
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        """Print training and validation metrics to console."""
        # train_evaluator.run(train_loader)
        # evaluator.run(val_loader)

        # tmetrics = train_evaluator.state.metrics
        # vmetrics = evaluator.state.metrics
        # for metric in metrics.keys():
        #     tm = tmetrics[metric]
        #     vm = vmetrics[metric]
        #     if metric == "roccurve":
        #         tm = [k.tolist() for k in tm]
        #         vm = [k.tolist() for k in vm]
        #     if isinstance(tm, torch.Tensor):
        #         tm = tm.cpu().numpy().tolist()
        #         vm = vm.cpu().numpy().tolist()

        #     history["train"][metric].append(tm)
        #     history["validation"][metric].append(vm)

        # train_evaluator.run(train_loader)
        evaluator.run(val_loader)

        vmetrics = evaluator.state.metrics
        for metric in metrics.keys():
            vm = vmetrics[metric]
            t_metric = metric
            if metric == "roccurve":
                vm = [k.tolist() for k in vm]
            if isinstance(vm, torch.Tensor):
                vm = vm.cpu().numpy().tolist()

            history["validation"][metric].append(vm)

        
        
        # Get train metrics from the training process
        # Access the trainer's state to get accumulated metrics
        if hasattr(engine.state, 'epoch_mae_sum') and engine.state.epoch_batch_count > 0:
            train_mae_avg = engine.state.epoch_mae_sum / engine.state.epoch_batch_count
            train_loss_avg = engine.state.epoch_loss_sum / engine.state.epoch_batch_count
        else:
            # Fallback: compute using evaluator (slower)
            train_evaluator.run(train_loader)
            eval_metrics = train_evaluator.state.metrics
            train_mae_avg = eval_metrics['mae']
            train_loss_avg = eval_metrics['loss']

        # Store train metrics
        tmetrics = {
            'mae': train_mae_avg,
            'loss': train_loss_avg,
            'neg_mae': -train_mae_avg
        }

        for metric in ['mae', 'loss', 'neg_mae']:
            if metric in tmetrics:
                history["train"][metric].append(tmetrics[metric])

        # Enhanced checkpoint management and strategic test evaluation
        current_epoch = len(history["validation"]["mae"])

        # Update best model tracker and determine if test evaluation is needed
        is_new_best, should_eval_test, test_reason = best_model_tracker.update_and_save(
            epoch=current_epoch,
            model=net,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=vmetrics,
            additional_info={
                'train_metrics': tmetrics,
                'config': config.dict() if hasattr(config, 'dict') else str(config)
            }
        )

        # Strategic test set evaluation
        test_metrics = None
        if should_eval_test and test_loader is not None:
            print(f"Test Evaluation at Epoch {current_epoch} ({test_reason})")
            test_evaluator.run(test_loader)
            test_metrics = test_evaluator.state.metrics

            # Store test metrics in history
            if "test" not in history:
                history["test"] = {metric: [] for metric in metrics.keys()}

            for metric in metrics.keys():
                tm = test_metrics[metric]
                if metric == "roccurve":
                    tm = [k.tolist() for k in tm]
                if isinstance(tm, torch.Tensor):
                    tm = tm.cpu().numpy().tolist()
                history["test"][metric].append(tm)

            # Log test evaluation
            best_model_tracker.log_test_evaluation(current_epoch, test_metrics, test_reason)

            # Display test results
            test_mae = test_metrics.get('mae', 'N/A')
            if is_new_best:
                print(f"NEW BEST MODEL - Test MAE: {test_mae:.6f}")
            else:
                print(f"Test Evaluation - Test MAE: {test_mae:.6f}")
        elif is_new_best:
            print(f"NEW BEST VALIDATION MODEL at epoch {current_epoch} - "
                  f"Val MAE: {vmetrics.get('mae', 'N/A'):.6f}")

        # Save training history
        history_file = os.path.join(config.output_dir, "training_history.json")
        try:
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save training history: {e}")


        # for metric in metrics.keys():
        #    history["train"][metric].append(tmetrics[metric])
        #    history["validation"][metric].append(vmetrics[metric])

        if config.store_outputs:
            history["EOS"] = eos.data
            history["trainEOS"] = train_eos.data
            dumpjson(
                filename=os.path.join(config.output_dir, "history_val.json"),
                data=history["validation"],
            )
            dumpjson(
                filename=os.path.join(config.output_dir, "history_train.json"),
                data=history["train"],
            )
        # Enhanced epoch summary with checkpoint and test information
        if config.progress:
            epoch_num = len(history["validation"]["mae"])
            summary = f"\nEpoch {epoch_num}: Train_MAE={tmetrics['mae']:.4f}, Val_MAE={vmetrics['mae']:.4f}"

            # Add best model indicator
            if is_new_best:
                summary += " ⭐ NEW BEST!"

            # Add test MAE if evaluated this epoch
            if test_metrics is not None:
                test_mae = test_metrics.get('mae', 'N/A')
                summary += f", Test_MAE={test_mae:.4f} ({test_reason})"
            elif "test" in history and len(history["test"]["mae"]) > 0:
                # Show last test MAE if available
                latest_test_mae = history["test"]["mae"][-1]
                summary += f", Last_Test_MAE={latest_test_mae:.4f}"

            # Add best model summary
            best_summary = best_model_tracker.get_summary()
            summary += f" | Best_Val_MAE={best_summary['best_metric_value']:.4f}@Epoch{best_summary['best_epoch']}"

            print(summary)

    if config.n_early_stopping is not None:
        if classification:
            my_metrics = "accuracy"
        else:
            my_metrics = "neg_mae"

        def default_score_fn(engine):
            score = engine.state.metrics[my_metrics]
            return score

        es_handler = EarlyStopping(
            patience=config.n_early_stopping,
            score_function=default_score_fn,
            trainer=trainer,
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, es_handler)
    # optionally log results to tensorboard
    if config.log_tensorboard:

        tb_logger = TensorboardLogger(
            log_dir=os.path.join(config.output_dir, "tb_logs", "test")
        )
        for tag, evaluator in [
            ("training", train_evaluator),
            ("validation", evaluator),
        ]:
            tb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names=["loss", "mae"],
                global_step_transform=global_step_from_engine(trainer),
            )

    trainer.run(train_loader, max_epochs=config.epochs)

    if config.log_tensorboard:
        test_loss = evaluator.state.metrics["loss"]
        tb_logger.writer.add_hparams(config, {"hparam/test_loss": test_loss})
        tb_logger.close()
    if config.write_predictions and classification:
        net.eval()
        f = open(
            os.path.join(config.output_dir, "prediction_results_test_set.csv"),
            "w",
        )
        f.write("id,target,prediction\n")
        targets = []
        predictions = []
        with torch.no_grad():
            ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
            for dat, id in zip(test_loader, ids):
                g, lg, target = dat
                out_data = net([g.to(device), lg.to(device)])
                # out_data = torch.exp(out_data.cpu())
                top_p, top_class = torch.topk(torch.exp(out_data), k=1)
                target = int(target.cpu().numpy().flatten().tolist()[0])

                f.write("%s, %d, %d\n" % (id, (target), (top_class)))
                targets.append(target)
                predictions.append(
                    top_class.cpu().numpy().flatten().tolist()[0]
                )
        f.close()
        from sklearn.metrics import roc_auc_score

        print("predictions", predictions)
        print("targets", targets)
        print(
            "Test ROCAUC:",
            roc_auc_score(np.array(targets), np.array(predictions)),
        )

    if (
        config.write_predictions
        and not classification
        and config.model.output_features > 1
    ):
        net.eval()
        mem = []
        with torch.no_grad():
            ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
            for dat, id in zip(test_loader, ids):
                g, lg, target = dat
                out_data = net([g.to(device), lg.to(device)])
                out_data = out_data.cpu().numpy().tolist()
                if config.standard_scalar_and_pca:
                    sc = pk.load(open("sc.pkl", "rb"))
                    out_data = list(
                        sc.transform(np.array(out_data).reshape(1, -1))[0]
                    )  # [0][0]
                target = target.cpu().numpy().flatten().tolist()
                info = {}
                info["id"] = id
                info["target"] = target
                info["predictions"] = out_data
                mem.append(info)
        dumpjson(
            filename=os.path.join(
                config.output_dir, "multi_out_predictions.json"
            ),
            data=mem,
        )
    if (
        config.write_predictions
        and not classification
        and config.model.output_features == 1
    ):
        net.eval()
        f = open(
            os.path.join(config.output_dir, "prediction_results_test_set.csv"),
            "w",
        )
        f.write("id,target,prediction\n")
        targets = []
        predictions = []
        with torch.no_grad():
            for dat in test_loader:
                g, lg, _, target = dat
                out_data = net([g.to(device), lg.to(device), lg.to(device)])
                out_data = out_data.cpu().numpy().tolist()
                target = target.cpu().numpy().flatten().tolist()
                if len(target) == 1:
                    target = target[0]
                targets.append(target)
                predictions.append(out_data)
        f.close()
        from sklearn.metrics import mean_absolute_error

        # Denormalize predictions and targets for final evaluation
        targets_denorm = np.array(targets) * std_train + mean_train
        predictions_denorm = np.array(predictions) * std_train + mean_train

        print(
            "Test MAE:",
            mean_absolute_error(targets_denorm, predictions_denorm),
        )
        if config.store_outputs and not classification:
            x = []
            y = []
            for i in history["EOS"]:
                x.append(i[0].cpu().numpy().tolist())
                y.append(i[1].cpu().numpy().tolist())
            x = np.array(x, dtype="float").flatten()
            y = np.array(y, dtype="float").flatten()
            f = open(
                os.path.join(
                    config.output_dir, "prediction_results_train_set.csv"
                ),
                "w",
            )
            # TODO: Add IDs
            f.write("target,prediction\n")
            for i, j in zip(x, y):
                f.write("%6f, %6f\n" % (j, i))
                line = str(i) + "," + str(j) + "\n"
                f.write(line)
            f.close()

    # Print final training summary
    print("\n" + "="*80)
    print("TRAINING COMPLETED - FINAL SUMMARY")
    print("="*80)

    final_summary = best_model_tracker.get_summary()
    print(f"Best Validation MAE: {final_summary['best_metric_value']:.6f} at Epoch {final_summary['best_epoch']}")
    print(f"Total Checkpoints Saved: {final_summary['total_checkpoints']}")
    print(f"Test Evaluations Performed: {final_summary['test_evaluations']}")

    if final_summary['test_evaluations'] > 0:
        print(f"Last Test Evaluation: Epoch {final_summary['last_test_epoch']}")

        # Show test results for best model if available
        if best_model_tracker.test_evaluation_log:
            best_val_test_results = None
            for test_log in best_model_tracker.test_evaluation_log:
                if test_log['reason'] == 'best-val-triggered':
                    best_val_test_results = test_log

            if best_val_test_results:
                test_mae = best_val_test_results['test_metrics'].get('mae', 'N/A')
                print(f"Test MAE for Best Validation Model: {test_mae:.6f}")

    print(f"Best Model Checkpoint: {final_summary['best_model_path']}")
    print("="*80)

    return history


