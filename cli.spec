# -*- mode: python ; coding: utf-8 -*-

block_cipher = None
import os
import shutil
import glob

a = Analysis(
    ['horguesc/cli.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(
    a.pure, 
    a.zipped_data,
    cipher=block_cipher
)

# シングルファイルEXEの代わりにCOLLECTを使用する
exe = EXE(
    pyz,
    a.scripts,
    [],  # 空のリストにしてバイナリや他のファイルをcollにまとめる
    exclude_binaries=True,  # バイナリを別に分ける
    name='horguesc',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='horguesc/resources/icon.ico' if os.path.exists('horguesc/resources/icon.ico') else None,
)

# COLLECTを追加する
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='horguesc',
    debug=False,
)


# Define distribution directory
dist_dir = os.path.join('dist', 'horguesc')

# Copy dataloader.exe and required DLLs to distribution directory
# Adjusted to match Visual Studio's default Release output directory
dataloader_paths = [
    os.path.join('Release', 'dataloader.exe'),  # Standard Release output
    os.path.join('dataloader', 'Release', 'dataloader.exe'),  # Alternative path
]

dataloader_found = False
for path in dataloader_paths:
    if os.path.exists(path):
        print(f"Found dataloader.exe at: {path}")
        shutil.copy2(path, dist_dir)
        dataloader_found = True
        
        # Copy any DLL files in the same directory
        dll_dir = os.path.dirname(path)
        for dll_file in glob.glob(os.path.join(dll_dir, "*.dll")):
            print(f"Copying dependency: {dll_file}")
            shutil.copy2(dll_file, dist_dir)
        break

if not dataloader_found:
    print("Warning: dataloader.exe not found at any expected location.")
    print("Please build the dataloader project first and make sure the output is in one of:")
    for path in dataloader_paths:
        print(f"- {path}")

# Copy sample configuration file if it exists
config_path = os.path.join('config', 'horguesc.ini')
if os.path.exists(config_path):
    shutil.copy2(config_path, dist_dir)
else:
    print("Warning: horguesc.ini not found at expected location.")

# Create README file with usage instructions
with open(os.path.join(dist_dir, 'README.txt'), 'w') as f:
    f.write("""horguesc Distribution Package

This package contains two executables:

1. dataloader.exe - Data loading tool for horse racing database
   Usage: dataloader.exe [setup|update|realtime]

2. horguesc.exe - Command-line interface for horse racing analysis
   Usage: horguesc.exe [train|test|predict] [options]
   
   Common commands:
   - horguesc.exe train --config path/to/config.ini
   - horguesc.exe test --model path/to/model
   - horguesc.exe predict --model path/to/model --input path/to/input

Before using these tools, make sure to configure the horguesc.ini file 
with your database and application settings.
""")

print(f"Distribution package created in {dist_dir}")