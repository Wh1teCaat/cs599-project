"""一次性下载 BGE 模型到本地，供后续离线使用。

模型对照:
  - BAAI/bge-m3                → HF_MODEL_NAME (embedding, 2.2 GB)
  - BAAI/bge-reranker-v2-m3    → RERANK_MODEL_NAME (精排, 1.1 GB)
"""

import os
from huggingface_hub import snapshot_download

LOCAL_ROOT = "/mnt/e/huggingface/embedding_model"

models = [
    ("BAAI/bge-m3", f"{LOCAL_ROOT}/bge-m3"),
    ("BAAI/bge-reranker-v2-m3", f"{LOCAL_ROOT}/bge-reranker-v2-m3"),
]

for repo_id, local_dir in models:
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"✅ {repo_id} 已存在，跳过")
        continue

    print(f"⬇️  下载 {repo_id} → {local_dir} ...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # WSL 下避免符号链接
            resume_download=True,
        )
        print(f"   完成: {repo_id}")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
