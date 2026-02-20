# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import certifi
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Get the directory containing this spec file
spec_dir = os.path.dirname(os.path.abspath(SPEC))

# Collect all distutils submodules (including setuptools._distutils)
distutils_imports = collect_submodules('distutils') + collect_submodules('setuptools._distutils')

# Collect all mmcv submodules to ensure runner, parallel, cnn, etc. are included
mmcv_imports = collect_submodules('mmcv')

a = Analysis(
    ['desktop_app.py'],
    pathex=[
        spec_dir,
        os.path.join(spec_dir, 'submodules', 'chartdete'),
        os.path.join(spec_dir, 'submodules', 'lineformer'),
        os.path.join(spec_dir, 'submodules', 'lineformer', 'mmdetection'),
        os.path.join(spec_dir, 'src'),
    ],
    binaries=[],
    datas=[
        # SSL certificates for HTTPS downloads
        (certifi.where(), 'certifi'),
        # Config files
        ('config', 'config'),
        # LineFormer submodule (config files and line_utils)
        ('submodules/lineformer/lineformer_swin_t_config.py', 'submodules/lineformer'),
        ('submodules/lineformer/line_utils.py', 'submodules/lineformer'),
        ('submodules/lineformer/infer.py', 'submodules/lineformer'),
        # LineFormer mmdetection
        ('submodules/lineformer/mmdetection/mmdet', 'submodules/lineformer/mmdetection/mmdet'),
        # ChartDete submodule
        ('submodules/chartdete/mmdet', 'submodules/chartdete/mmdet'),
        ('submodules/chartdete/configs', 'submodules/chartdete/configs'),
        # src module
        ('src/chartdete_infer.py', 'src'),
    ],
    hiddenimports=[
        'mmdet',
        'mmdet.models',
        'mmdet.models.roi_heads',
        'mmdet.models.roi_heads.cascade_roi_head_LGF',
        'torch',
        'torchvision',
        'cv2',
        'numpy',
        'easyocr',
        'PIL',
        'skimage',
        'scipy',
        'bresenham',
        'terminaltables',
        'matplotlib',
        'pycocotools',
    ] + distutils_imports + mmcv_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AutoLineDigitizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AutoLineDigitizer',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='AutoLineDigitizer.app',
        icon=None,
        bundle_identifier='com.lineformer.autolinedigitizer',
    )
