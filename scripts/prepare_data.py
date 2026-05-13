from __future__ import annotations

import argparse
import tarfile
import time
from pathlib import Path

import requests
from tqdm import tqdm


DATASET_URL = "http://dags.stanford.edu/data/iccv09Data.tar.gz"


def download(url: str, destination: Path) -> None:
    """下载 Stanford Background Dataset 压缩包，并显示进度条。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with destination.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def extract(archive: Path, target_dir: Path) -> None:
    """解压数据集到目标目录，支持 tar.gz 直接展开。"""
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=target_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="准备 Stanford Background Dataset。")
    parser.add_argument("--root", default="data", help="数据集根目录。")
    parser.add_argument("--url", default=DATASET_URL, help="数据集下载地址。")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载，直接解压本地压缩包。")
    args = parser.parse_args()

    root = Path(args.root)
    archive = root / "iccv09Data.tar.gz"
    if not archive.exists() and not args.skip_download:
        print(f"正在下载数据集到 {archive} ...")
        download(args.url, archive)
    if not archive.exists():
        raise FileNotFoundError(f"未找到压缩包：{archive}")
    extract_dir = root / "raw"
    if extract_dir.exists():
        # 遵循项目规则：不直接删除既有文件。若 raw 已存在，则解压到新的时间戳目录，
        # 由用户确认后自行清理旧目录，避免误删已有数据或实验文件。
        extract_dir = root / f"raw_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"正在解压到 {extract_dir} ...")
    extract(archive, extract_dir)
    print("数据准备完成。")


if __name__ == "__main__":
    main()
