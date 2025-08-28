"""
Author: Bin Cao (bcao686@connect.hkust-gz.edu.cn)
Affiliations:
- Hong Kong University of Science and Technology (Guangzhou)
- City University of Hong Kong

Description: Shared utilities and configuration classes.
"""
import json
import glob
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt

from pydantic_settings import BaseSettings as PydanticBaseSettings
from jarvis.db.jsonutils import loadjson


class BaseSettings(PydanticBaseSettings):
    """Add configuration to default Pydantic BaseSettings."""

    class Config:
        """Configure BaseSettings behavior."""

        extra = "forbid"
        use_enum_values = True
        env_prefix = "jv_"


def plot_learning_curve(
    results_dir: Union[str, Path], key: str = "mae", plot_train: bool = False
):
    """Plot learning curves based on json history files."""
    if isinstance(results_dir, str):
        results_dir = Path(results_dir)

    with open(results_dir / "history_val.json", "r") as f:
        val = json.load(f)

    p = plt.plot(val[key], label=results_dir.name)

    if plot_train:
        # plot the training trace in the same color, lower opacity
        with open(results_dir / "history_train.json", "r") as f:
            train = json.load(f)

        c = p[0].get_color()
        plt.plot(train[key], alpha=0.5, c=c)

    plt.xlabel("epochs")
    plt.ylabel(key)

    return train, val


def check_early_stopping_reached(
    validation_file="history_val.json", n_early_stopping=30
):
    """Check if early stopping reached."""
    early_stopping_reached = False
    maes = loadjson(validation_file)["mae"]
    best_mae = 1e9
    no_improvement = 0
    best_epoch = len(maes)
    for ii, i in enumerate(maes):
        if i > best_mae:
            no_improvement += 1
            if no_improvement == n_early_stopping:
                print("Reached Early Stopping at", i, "epoch=", ii)
                early_stopping_reached = True
                best_mae = i
                best_epoch = ii
                break
        else:
            no_improvement = 0
            best_mae = i
    return early_stopping_reached, best_mae, best_epoch


def check_all_folders(path="."):
    """Check results for all sub folders of a dataset run."""
    for i in glob.glob(path + "/*/history_val.json"):
        print(i)
        (
            early_stopping_reached,
            best_mae,
            best_epoch,
        ) = check_early_stopping_reached(validation_file=i)
        print(
            "early_stopping_reached,best_mae,best_epoch",
            early_stopping_reached,
            best_mae,
            best_epoch,
        )
        print()
