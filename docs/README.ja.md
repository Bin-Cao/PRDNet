<p align="center"><img src="logo.svg" width="92" alt="PRDNet ロゴ" /></p>

# PRDNet：擬似粒子レイ回折ネットワーク

<p align="center">
  <a href="README.en.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE) [![ICLR 2026](https://img.shields.io/badge/ICLR-OpenReview-4b44ce.svg)](https://openreview.net/forum?id=OfmurJrzlT) [![GitHub stars](https://img.shields.io/github/stars/Bin-Cao/PRDNet?style=flat&logo=github)](https://github.com/Bin-Cao/PRDNet/stargazers)

PRDNet は、結晶物性予測のための物理情報を取り入れたグラフニューラルネットワークです。グラフ表現と逆空間における学習可能な擬似粒子レイ回折機構を組み合わせ、結晶対称性に対する不変性を保ちながら長距離の構造相互作用を捉えます。

## 特長

- 結晶構造のためのグラフニューラルネットワーク
- 逆空間における擬似粒子レイ回折物理
- マルチヘッド注意機構と対称性不変表現

## クイックスタート

```bash
git clone https://github.com/Bin-Cao/PRDNet.git
cd PRDNet
pip install -r requirements.txt
python -c "import prdnet; print('PRDNet installed successfully!')"
```

## データ

ASE データベース形式を使用してください。各構造には `formation_energy`、`band_gap`、`bulk_modulus`、`shear_modulus` などの数値ターゲットを含められます。

```python
from ase.db import connect
from ase.build import bulk

db = connect("my_data.db")
db.write(bulk("Si", "diamond", a=5.43), formation_energy=-5.42, band_gap=1.12)
```

[CPPbenchmark / Materials Project](https://huggingface.co/datasets/caobin/CPPbenchmark)、JARVIS-DFT、または ASE 互換データベースを利用できます。

## 学習

`trainer.py` でデータベースのパスと予測対象を設定してから実行します。

```bash
python trainer.py
```

複数 GPU を使用する場合：

```bash
torchrun --nproc_per_node=4 trainer.py
```

主な設定項目は `epochs`、`batch_size`、`learning_rate`、`conv_layers`、`node_features`、`use_diffraction`、`diffraction_max_hkl` です。詳しい設定とトラブルシューティングは[英語版メイン README](../README.md)を参照してください。

## 引用

```bibtex
@article{cao2025beyond,
  title={Beyond Structure: Invariant Crystal Property Prediction with Pseudo-Particle Ray Diffraction},
  author={Cao, Bin and Liu, Yang and Zhang, Longhan and Wu, Yifan and Li, Zhixun and Luo, Yuyu and Cheng, Hong and Ren, Yang and Zhang, Tong-Yi},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

## ライセンス

MIT。詳細は [LICENSE](../LICENSE) を参照してください。
