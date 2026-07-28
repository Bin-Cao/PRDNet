<p align="center"><img src="logo.svg" width="92" alt="PRDNet 로고" /></p>

# PRDNet: 의사 입자 광선 회절 네트워크

<p align="center">
  <a href="README.en.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE) [![ICLR 2026](https://img.shields.io/badge/ICLR-OpenReview-4b44ce.svg)](https://openreview.net/forum?id=OfmurJrzlT) [![GitHub stars](https://img.shields.io/github/stars/Bin-Cao/PRDNet?style=flat&logo=github)](https://github.com/Bin-Cao/PRDNet/stargazers)

PRDNet은 결정 물성 예측을 위한 물리 정보 기반 그래프 신경망입니다. 그래프 표현과 역공간의 학습 가능한 의사 입자 광선 회절 메커니즘을 결합하여, 결정 대칭성 불변성을 유지하면서 장거리 구조 상호작용을 포착합니다.

## 주요 기능

- 결정 구조를 위한 그래프 신경망
- 역공간 의사 입자 광선 회절 물리 통합
- 멀티헤드 어텐션 및 대칭성 불변 표현

## 빠른 시작

```bash
git clone https://github.com/Bin-Cao/PRDNet.git
cd PRDNet
pip install -r requirements.txt
python -c "import prdnet; print('PRDNet installed successfully!')"
```

## 데이터

ASE 데이터베이스 형식을 사용하세요. 각 구조에는 `formation_energy`, `band_gap`, `bulk_modulus`, `shear_modulus` 등의 수치 타깃을 포함할 수 있습니다.

```python
from ase.db import connect
from ase.build import bulk

db = connect("my_data.db")
db.write(bulk("Si", "diamond", a=5.43), formation_energy=-5.42, band_gap=1.12)
```

[CPPbenchmark / Materials Project](https://huggingface.co/datasets/caobin/CPPbenchmark), JARVIS-DFT 또는 ASE 호환 데이터베이스를 사용할 수 있습니다.

## 학습

`trainer.py`에서 데이터베이스 경로와 예측 대상을 설정한 뒤 실행합니다.

```bash
python trainer.py
```

다중 GPU 학습:

```bash
torchrun --nproc_per_node=4 trainer.py
```

주요 설정은 `epochs`, `batch_size`, `learning_rate`, `conv_layers`, `node_features`, `use_diffraction`, `diffraction_max_hkl`입니다. 전체 설정 및 문제 해결 방법은 [영문 메인 README](../README.md)를 참고하세요.

## 인용

```bibtex
@article{cao2025beyond,
  title={Beyond Structure: Invariant Crystal Property Prediction with Pseudo-Particle Ray Diffraction},
  author={Cao, Bin and Liu, Yang and Zhang, Longhan and Wu, Yifan and Li, Zhixun and Luo, Yuyu and Cheng, Hong and Ren, Yang and Zhang, Tong-Yi},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

## 라이선스

MIT. 자세한 내용은 [LICENSE](../LICENSE)를 참고하세요.
