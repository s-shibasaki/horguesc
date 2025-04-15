"""
Core functionality for the horguesc package.
"""

from dataclasses import dataclass
import subprocess
import json
import numpy as np
import os
import tempfile
import struct


@dataclass
class Record:
    id: int
    value: float


def hello():
    """Return a greeting message."""
    return "Hello from horguesc!"


def load_dataset(exe_path, params=None):
    """
    Load a dataset using a C++ executable.

    Parameters:
        exe_path (str): Path to the C++ executable.
        params (dict, optional): Additional parameters to pass to the executable.

    Returns:
        dict or numpy.ndarray: The loaded dataset.
    """
    if params is None:
        params = {}

    # 一時ファイルを作成してデータを保存
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        output_path = tmp_file.name

    # コマンドライン引数を構築
    cmd_args = [exe_path, f"--output={output_path}"]
    for key, value in params.items():
        cmd_args.append(f"--{key}={value}")

    try:
        # C++実行ファイルを実行
        result = subprocess.run(cmd_args, check=True)

        # 成功した場合、一時ファイルからデータを読み込む
        with open(output_path, "rb") as f:
            version_bytes = f.read(4)
            version = struct.unpack("I", version_bytes)[0]

            if version != 1:
                raise ValueError(f"Unsupported version: {version}")

            record_count_bytes = f.read(4)
            record_count = struct.unpack("I", record_count_bytes)[0]

            records = []
            for i in range(record_count):
                id_bytes = f.read(4)
                record_id = struct.unpack("I", id_bytes)[0]

                value_bytes = f.read(4)
                value = struct.unpack("f", value_bytes)[0]

                record = Record(id=record_id, value=value)
                records.append(record)

        return records

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"C++ executable failed with error: {e.stderr}")

    finally:
        # 一時ファイルを削除
        if os.path.exists(output_path):
            os.remove(output_path)
