"""
规定：instruction.json 放置在与数据路径相同的目录下。

- 数据路径为文件（如 HDF5/MCAP）时，instruction.json 为该文件所在目录下的 instruction.json。
- 数据路径为目录时，instruction.json 为该目录下的 instruction.json。
"""
import os

INSTRUCTION_FILENAME = "instruction.json"


def get_instruction_path_for_data_path(data_path: str) -> str:
    """
    根据数据路径返回 instruction.json 的绝对路径。

    规则：instruction.json 与数据路径在同一目录下。
    data_path 可为数据文件路径（如 /path/to/episode_0.hdf5）或数据所在目录路径。
    """
    if not data_path or not data_path.strip():
        raise ValueError("data_path 不能为空")
    base = os.path.abspath(data_path.strip())
    if os.path.isfile(base):
        base = os.path.dirname(base)
    return os.path.join(base, INSTRUCTION_FILENAME)
