#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
opt_atomic.py

使用多进程并行打包图片为 .tar，并通过扫描已有 .tar 来实现断点续传（checkpoint）。
每个子进程负责将一个 samples_per_tar 大小的图片块打包成一个 .tar.tmp，完成后原子重命名为 .tar。
重启时只需扫描 output_root 下已有的 .tar 文件，跳过对应的块索引，避免重复处理。
"""

import os
import uuid
import tarfile
import json
from PIL import Image
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# === 参数配置 ===
input_root      = "/home/iv/images/new_aws_pics/2021"   # 输入图片根目录
output_root     = "/home/iv/images/new_aws_tar_v2"        # 输出 .tar 保存根目录
samples_per_tar = 10000                                 # 每个 .tar 包含的图片数量
tars_per_group  = 500                                   # 每 tars_per_group 个 .tar 归到一个子目录 group_xxx
max_workers     = 48                                     # 并行进程数，建议设置为 CPU 核数的一半或相近
image_suffixes  = {".jpg", ".jpeg", ".png", ".webp"}    # 支持的图片后缀

# 可以按需修改 start_group / start_tar，以支持跨作业继续
start_group = 0
start_tar   = 5227

os.makedirs(output_root, exist_ok=True)


def get_finished_indices(output_root: str, tars_per_group: int) -> set[int]:
    """
    扫描 output_root 下所有 group_* 目录，收集已存在的 .tar 文件对应的 tar_index。
    tar_index = int(“000123.tar”.replace(".tar",""))
    返回一个包含所有已完成 tar_index 的集合。
    """
    finished = set()
    # 列出所有以 “group_” 开头的子目录
    for entry in os.listdir(output_root):
        if not entry.startswith("group_"):
            continue
        group_dir = os.path.join(output_root, entry)
        if not os.path.isdir(group_dir):
            continue
        # 列出本组下所有 .tar 文件
        for fname in os.listdir(group_dir):
            if not fname.endswith(".tar"):
                continue
            try:
                idx = int(fname[:-4])  # “000123.tar” 去掉 “.tar” 后转 int → 123
                finished.add(idx)
            except ValueError:
                # 如果文件名不是五位数字.tar，则略过
                continue
    return finished


def process_and_write_atomic(paths_block: list[str],
                             tar_index: int,
                             output_root: str,
                             tars_per_group: int) -> bool:
    """
    子进程任务：将一个长度为 samples_per_tar 的图片路径列表 paths_block
    打包成一个 .tar.tmp，完成后原子重命名为 .tar，并生成同目录下的统计 .json 文件。

    返回 True 表示该块打包并写入成功；False 表示失败。
    """
    # 计算本 tar_index 应该存放到哪个 group_{:05d} 目录
    group_id   = tar_index // tars_per_group
    group_dir  = os.path.join(output_root, f"group_{group_id:05d}")
    os.makedirs(group_dir, exist_ok=True)

    final_tar = os.path.join(group_dir, f"{tar_index:05d}.tar")
    tmp_tar   = final_tar + ".tmp"

    # 如果 final_tar 已存在，直接跳过
    if os.path.exists(final_tar):
        return True

    # 1. 打包到 tmp 文件
    success = 0
    try:
        with tarfile.open(tmp_tar, "w") as tar:
            for path in paths_block:
                try:
                    with Image.open(path) as img:
                        img = img.convert("RGB")
                        width, height = img.size
                        buf = BytesIO()
                        img.save(buf, format="JPEG")
                        img_bytes = buf.getvalue()
                    key = str(uuid.uuid4())
                    # 写入 JPEG
                    info_jpg = tarfile.TarInfo(name=f"{key}.jpg")
                    info_jpg.size = len(img_bytes)
                    tar.addfile(info_jpg, BytesIO(img_bytes))
                    # 写入 JSON 元信息
                    meta = {
                        "filename": os.path.basename(path),
                        "width": width,
                        "height": height,
                    }
                    json_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
                    info_json = tarfile.TarInfo(name=f"{key}.json")
                    info_json.size = len(json_bytes)
                    tar.addfile(info_json, BytesIO(json_bytes))
                    success+=1
                except Exception as e_img:
                    print(f"❌ [Chunk {tar_index}] 处理图片出错: {path} → {e_img}")
                    # 如果单张图片解码或写入失败，则跳过该图片，继续下一个
                    continue
    except Exception as e_tar:
        # 打包阶段出错，删除临时文件并返回 False
        if os.path.exists(tmp_tar):
            os.remove(tmp_tar)
        print(f"❌ [Chunk {tar_index}] 打包到 tmp 文件失败: {e_tar}")
        return False

    # 2. 原子性重命名：tmp → final
    try:
        os.rename(tmp_tar, final_tar)
    except Exception as e_rename:
        # 如果重命名失败，删除 tmp 并返回 False
        if os.path.exists(tmp_tar):
            os.remove(tmp_tar)
        print(f"❌ [Chunk {tar_index}] 重命名 .tmp 失败: {e_rename}")
        return False

    # 3. 写统计 JSON，与 final_tar 同目录下同名 .json
    try:
        count_info = {
            "tar_file": os.path.basename(final_tar),
            "image_count": success
        }
        count_json_path = final_tar.replace(".tar", ".json")
        with open(count_json_path, "w", encoding="utf-8") as f_json:
            json.dump(count_info, f_json, ensure_ascii=False, indent=2)
    except Exception as e_count:
        print(f"⚠️ [Chunk {tar_index}] 写统计 JSON 失败: {e_count}")

    print(f"✅ [Chunk {tar_index}] 写入成功: {final_tar} ({len(paths_block)} 张图片)")
    return True


def main():
    # 1. 按照扩展名收集所有图片路径（静态列表），并排序，保证 chunk 分配固定
    all_paths = []
    for dirpath, _, filenames in os.walk(input_root):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in image_suffixes:
                all_paths.append(os.path.join(dirpath, fname))
    all_paths.sort()

    total_images = len(all_paths)
    if total_images == 0:
        print("⚠️ 未找到任何支持格式的图片。")
        return

    print(f"📂 共发现 {total_images} 张待处理图片 (含子目录)。")

    # 2. 计算总共多少个块（chunk）
    total_chunks = (total_images + samples_per_tar - 1) // samples_per_tar
    print(f"🔢 预计分 {total_chunks} 个块，每块 ≈ {samples_per_tar} 张图片。")

    # 3. 扫描已有 .tar，生成已完成的 tar_index 集合
    finished_indices = get_finished_indices(output_root, tars_per_group)
    if finished_indices:
        print(f"🔁 检测到已有 {len(finished_indices)} 个已完成的 .tar 块，将自动跳过它们。")
    else:
        print("🔄 未检测到已完成的 .tar块，将全部重新打包。")

    # 4. 按块并行提交到进程池
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        pbar = tqdm(total=total_chunks, desc="Chunks", unit="tar")
        for chunk_idx in range(total_chunks):
            tar_index = start_group * tars_per_group + start_tar + chunk_idx
            if tar_index in finished_indices:
                # 跳过已完成的 tar_index，并更新进度条
                pbar.update(1)
                continue

            # 该块的图片索引范围：[chunk_idx * samples_per_tar : (chunk_idx+1) * samples_per_tar)
            start_i = chunk_idx * samples_per_tar
            end_i   = min(start_i + samples_per_tar, total_images)
            paths_block = all_paths[start_i:end_i]

            # 提交子进程任务
            future = executor.submit(
                process_and_write_atomic,
                paths_block,
                tar_index,
                output_root,
                tars_per_group
            )
            futures.append(future)


        # 等待所有任务完成
        for fut in as_completed(futures):
            pbar.update(1)
            # 取回返回值 (True/False)，可用于检查失败的块
            success = fut.result()
            # 立即更新进度条 (任务已提交，相当于“占用一个槽”)
        pbar.close()

    print("✅ 全部任务完成。")


if __name__ == "__main__":
    main()
