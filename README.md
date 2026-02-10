# AutoLineDigitizer

A Streamlit web application for automatic line chart data extraction using [LineFormer](https://github.com/TheJaeLal/LineFormer) with automatic axis detection via [ChartDete](https://github.com/pengyu965/ChartDete/) and [EasyOCR](https://github.com/JaidedAI/EasyOCR).

## Demo

<video src="https://github.com/t29mato/AutoLineDigitizer/releases/download/v0.1.0/AutoLineDigitizerDemo.mp4" controls width="100%"></video>

| Input | Output |
|-------|--------|
| ![Input](demo/10.3390_nano14040384_4i.png) | ![Output](demo/10.3390_nano14040384_4i_result.png) |

### Export to Digitizer Tools

| [StarryDigitizer](https://starrydigitizer.vercel.app/) | [WebPlotDigitizer](https://apps.automeris.io/wpd4/) |
|------------------|------------------|
| ![StarryDigitizer](demo/10.3390_nano14040384_4i_sd.png) | ![WebPlotDigitizer](demo/10.3390_nano14040384_4i_wpd.png) |

## Features

- **Line Extraction**: Automatic line detection using LineFormer
- **Axis Detection**: Automatic axis label reading via ChartDete + OCR
- **Export Formats**:
  - [StarryDigitizer](https://starrydigitizer.vercel.app/) ZIP
  - [WebPlotDigitizer](https://apps.automeris.io/wpd4/) TAR

## Project Structure

```
LineFormer/
├── src/                    # Source code
│   ├── app.py              # Streamlit application
│   └── chartdete_infer.py  # ChartDete inference wrapper
├── config/                 # Configuration files
│   └── chartdete_config.py # ChartDete model config
├── models/                 # Model weights (not in git)
│   ├── iter_3000.pth       # LineFormer model
│   └── checkpoint.pth      # ChartDete model
├── submodules/             # Git submodules
│   ├── lineformer/         # LineFormer repository
│   └── chartdete/          # ChartDete repository
├── demo/                   # Demo images
├── requirements.txt        # Python dependencies
├── LICENSE
└── README.md
```

## Quick Start

```bash
# Clone with submodules
git clone --recursive https://github.com/YOUR_USERNAME/LineFormer.git
cd LineFormer

# Or if already cloned, initialize submodules
git submodule update --init --recursive

# Install dependencies
pip install -r requirements.txt

# Download models (see Model Weights section below)

# Run the app
streamlit run src/app.py
```

### Model Weights

Download the following model weights and place them in the `models/` directory:

| Model | Size | Source |
|-------|------|--------|
| LineFormer (`iter_3000.pth`) | 543MB | [Original Repository](https://drive.google.com/drive/folders/1K_zLZwgoUIAJtfjwfCU5Nv33k17R0O5T?usp=sharing) |
| ChartDete (`checkpoint.pth`) | 1.4GB | [ChartDete Repository](https://github.com/pengyu965/ChartDete/) |

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
