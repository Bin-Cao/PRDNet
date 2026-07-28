<p align="center"><img src="logo.svg" width="92" alt="PRDNet logo" /></p>

# PRDNet: Pseudo-particle Ray Diffraction Network

<p align="center">
  <a href="README.en.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE) [![ICLR 2026](https://img.shields.io/badge/ICLR-OpenReview-4b44ce.svg)](https://openreview.net/forum?id=OfmurJrzlT) [![GitHub stars](https://img.shields.io/github/stars/Bin-Cao/PRDNet?style=flat&logo=github)](https://github.com/Bin-Cao/PRDNet/stargazers)

PRDNet is a physics-informed graph neural network for crystal property prediction. It combines graph representations with a learnable pseudo-particle ray-diffraction mechanism in reciprocal space, capturing long-range structural interactions while preserving crystallographic symmetry invariance.

## Highlights

- Graph neural networks for crystal structures
- Pseudo-particle ray-diffraction physics in reciprocal space
- Multi-head attention and symmetry-invariant representations

## Quick Start

```bash
git clone https://github.com/Bin-Cao/PRDNet.git
cd PRDNet
pip install -r requirements.txt
python -c "import prdnet; print('PRDNet installed successfully!')"
```

## Data

Use ASE database files. Each structure can include numeric target properties such as `formation_energy`, `band_gap`, `bulk_modulus`, or `shear_modulus`.

```python
from ase.db import connect
from ase.build import bulk

db = connect("my_data.db")
db.write(bulk("Si", "diamond", a=5.43), formation_energy=-5.42, band_gap=1.12)
```

Datasets: [CPPbenchmark / Materials Project](https://huggingface.co/datasets/caobin/CPPbenchmark), JARVIS-DFT, or any ASE-compatible database.

## Training

Set your database paths and target in `trainer.py`, then run:

```bash
python trainer.py
```

For multi-GPU training:

```bash
torchrun --nproc_per_node=4 trainer.py
```

Important configuration options include `epochs`, `batch_size`, `learning_rate`, `conv_layers`, `node_features`, `use_diffraction`, and `diffraction_max_hkl`. See the [main README](../README.md) for the complete configuration and troubleshooting guide.

## Citation

```bibtex
@article{cao2025beyond,
  title={Beyond Structure: Invariant Crystal Property Prediction with Pseudo-Particle Ray Diffraction},
  author={Cao, Bin and Liu, Yang and Zhang, Longhan and Wu, Yifan and Li, Zhixun and Luo, Yuyu and Cheng, Hong and Ren, Yang and Zhang, Tong-Yi},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

## License

MIT. See [LICENSE](../LICENSE).
