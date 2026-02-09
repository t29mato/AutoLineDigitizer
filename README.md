# Line Chart Data Extractor

A Streamlit web application for extracting data from line chart images using [LineFormer](https://github.com/TheJaeLal/LineFormer) with automatic axis detection via [ChartDete](https://github.com/pengyu965/ChartDete/) and EasyOCR.

## Features

- **Line Extraction**: Automatic line detection using LineFormer
- **Axis Detection**: Automatic axis label reading via ChartDete + OCR
- **Export Formats**:
  - [starry-digitizer](https://starrydigitizer.vercel.app/) ZIP
  - [WebPlotDigitizer](https://apps.automeris.io/wpd4/) TAR

## Quick Start

```bash
# Clone with submodules
git clone --recursive https://github.com/YOUR_USERNAME/LineFormer.git
cd LineFormer

# Or if already cloned, initialize submodules
git submodule update --init --recursive

# Install dependencies
pip install -r requirements_streamlit.txt

# Download models (see Model Weights section below)

# Run the app
streamlit run streamlit_app.py
```

### Model Weights

Download the following model weights:

| Model | Size | Source |
|-------|------|--------|
| LineFormer (`iter_3000.pth`) | 543MB | [Original Repository](https://drive.google.com/drive/folders/1K_zLZwgoUIAJtfjwfCU5Nv33k17R0O5T?usp=sharing) |
| ChartDete (`chartdete/checkpoint.pth`) | 1.4GB | [ChartDete Repository](https://github.com/pengyu965/ChartDete/) |

Place the model files in the following locations:
```
LineFormer/
├── iter_3000.pth              # LineFormer model
└── chartdete/
    └── checkpoint.pth         # ChartDete model
```

---

## Image Attribution

Demo images are used under the following licenses:

- **demo/10.3390_nano14040384_4i.png**: Figure 4(i) from [Wang et al., Nanomaterials, 2024](https://doi.org/10.3390/nano14040384), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Components

- **LineFormer**: [ICDAR 2023 Paper](https://link.springer.com/chapter/10.1007/978-3-031-41734-4_24) by Jay Lal et al.
- **ChartDete**: MIT License, Copyright (c) 2023 Pengyu Yan
- **MMDetection**: Apache License 2.0, Copyright (c) 2018-2023 OpenMMLab
