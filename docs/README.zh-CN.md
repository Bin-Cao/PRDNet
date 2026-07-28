<p align="center"><img src="logo.svg" width="92" alt="PRDNet 标志" /></p>

# PRDNet：伪粒子射线衍射网络

<p align="center">
  <a href="README.en.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE) [![ICLR 2026](https://img.shields.io/badge/ICLR-OpenReview-4b44ce.svg)](https://openreview.net/forum?id=OfmurJrzlT) [![GitHub stars](https://img.shields.io/github/stars/Bin-Cao/PRDNet?style=flat&logo=github)](https://github.com/Bin-Cao/PRDNet/stargazers)

PRDNet 是面向晶体性质预测的物理先验图神经网络。它将图表示与倒易空间中可学习的伪粒子射线衍射机制结合，在保持晶体对称性不变性的同时捕获长程结构相互作用。

## 特点

- 基于图神经网络的晶体结构表示
- 融合倒易空间伪粒子射线衍射物理
- 多头注意力与对称性不变表示

## 快速开始

```bash
git clone https://github.com/Bin-Cao/PRDNet.git
cd PRDNet
pip install -r requirements.txt
python -c "import prdnet; print('PRDNet installed successfully!')"
```

## 数据准备

请使用 ASE 数据库格式。每个结构可包含 `formation_energy`、`band_gap`、`bulk_modulus` 或 `shear_modulus` 等数值标签。

```python
from ase.db import connect
from ase.build import bulk

db = connect("my_data.db")
db.write(bulk("Si", "diamond", a=5.43), formation_energy=-5.42, band_gap=1.12)
```

可使用 [CPPbenchmark / Materials Project](https://huggingface.co/datasets/caobin/CPPbenchmark)、JARVIS-DFT，或任意兼容 ASE 的数据库。

## 训练

在 `trainer.py` 中设置数据库路径和预测目标后运行：

```bash
python trainer.py
```

多 GPU 训练：

```bash
torchrun --nproc_per_node=4 trainer.py
```

常用配置包括 `epochs`、`batch_size`、`learning_rate`、`conv_layers`、`node_features`、`use_diffraction` 与 `diffraction_max_hkl`。完整配置和排错说明请参见[英文主 README](../README.md)。

## 引用

```bibtex
@article{cao2025beyond,
  title={Beyond Structure: Invariant Crystal Property Prediction with Pseudo-Particle Ray Diffraction},
  author={Cao, Bin and Liu, Yang and Zhang, Longhan and Wu, Yifan and Li, Zhixun and Luo, Yuyu and Cheng, Hong and Ren, Yang and Zhang, Tong-Yi},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

## 许可证

MIT，详见 [LICENSE](../LICENSE)。
