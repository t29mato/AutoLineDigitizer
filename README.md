# AutoLineDigitizer

A desktop application for automatic line chart data extraction using [LineFormer](https://github.com/TheJaeLal/LineFormer) with automatic axis detection via [ChartDete](https://github.com/pengyu965/ChartDete/) and [EasyOCR](https://github.com/JaidedAI/EasyOCR).

## Demo

### Video

https://github.com/user-attachments/assets/7ecb641e-f939-40a5-ad7b-54b64937fdd4

### Input / Output

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

## Download

Download the latest version from the [Releases](https://github.com/t29mato/AutoLineDigitizer/releases) page.

| Platform | File |
|----------|------|
| macOS (Apple Silicon) | `AutoLineDigitizer-macOS.zip` |
| Windows | `AutoLineDigitizer-Windows.zip` |

> **Note:** Intel Mac is not currently supported. Apple Silicon (M1/M2/M3/M4) only.

### Installation

#### macOS

1. Download and unzip `AutoLineDigitizer-macOS.zip`
2. Move `AutoLineDigitizer.app` to Applications
3. On first launch, macOS will show a warning: **"AutoLineDigitizer.app cannot be opened because the developer cannot be verified."**
   - Click **Cancel** (not "Move to Trash")
   - Go to **System Settings → Privacy & Security** → scroll down and click **"Open Anyway"**
   - Or: **Right-click** the app → **Open** → click **Open** in the dialog
4. Models will be downloaded automatically on first launch

#### Windows

1. Download and unzip `AutoLineDigitizer-Windows.zip`
2. Run `AutoLineDigitizer\AutoLineDigitizer.exe`
3. Models will be downloaded automatically on first launch

### Manual Model Download (Proxy / Firewall environments)

If auto-download fails (e.g., due to a corporate proxy), you can download the models manually via your browser and import them into the app.

#### Base Models (required)

1. Download `iter_3000.pth` and `checkpoint.pth` from [GitHub Releases](https://github.com/t29mato/AutoLineDigitizer/releases/tag/models)
2. In the app, click **Import Models** button that appears under the Line Model dropdown
3. Select the downloaded `.pth` files

#### Battery Fine-tuned Models (optional)

1. Download the model file from [HuggingFace](https://huggingface.co/t29mato/lineformer-battery-finetuned/tree/main)
   - `lineformer_battery_iter_5000.pth` — Battery (iter_5000)
   - `lineformer_battery_best_iter_1300.pth` — Battery (best)
2. Select the Battery model from the **Line Model** dropdown
3. Click the **Import Model** button that appears
4. Select the downloaded `.pth` file

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
