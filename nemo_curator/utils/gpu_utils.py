# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU and CUDA library utilities.

Handles runtime discovery and loading of CUDA libraries (e.g. cuDNN) that are
installed as Python packages (``nvidia-cudnn-cu12``).  ONNX Runtime relies on
the system dynamic linker to locate ``libcudnn*.so`` files, but pip-installed
packages place them inside the virtual-environment ``site-packages`` tree which
is **not** on the default library search path.

Call :func:`ensure_cudnn_loaded` early — before any ``import onnxruntime`` — to
make those libraries visible to the linker.
"""

from __future__ import annotations

import ctypes
import logging
import math
import os

import torch
from loguru import logger as loguru_logger
from transformers import AutoConfig

logger = logging.getLogger(__name__)

_cudnn_loaded: bool = False


def ensure_cudnn_loaded() -> bool:
    """Discover and pre-load cuDNN from the ``nvidia-cudnn-cu12`` pip package.

    This function is **idempotent**: repeated calls are cheap no-ops after the
    first successful load.

    Returns
    -------
    bool
        ``True`` if cuDNN was successfully loaded (or was already loaded),
        ``False`` otherwise.
    """
    global _cudnn_loaded  # noqa: PLW0603

    if _cudnn_loaded:
        return True

    try:
        import nvidia.cudnn  # noqa: WPS433
    except ImportError:
        logger.debug(
            "nvidia-cudnn-cu12 is not installed; "
            "cuDNN must be available on the system LD_LIBRARY_PATH for GPU inference."
        )
        return False

    cudnn_lib_dir = os.path.join(list(nvidia.cudnn.__path__)[0], "lib")
    if not os.path.isdir(cudnn_lib_dir):
        logger.warning("nvidia.cudnn package found but lib directory missing: %s", cudnn_lib_dir)
        return False

    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if cudnn_lib_dir not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = cudnn_lib_dir + (":" + ld_path if ld_path else "")

    import glob

    cudnn_libs = sorted(glob.glob(os.path.join(cudnn_lib_dir, "libcudnn*.so*")))
    if not cudnn_libs:
        logger.warning("No libcudnn*.so* files found in %s", cudnn_lib_dir)
        return False

    cudnn_libs.sort(key=lambda p: (not os.path.basename(p).startswith("libcudnn.so."), p))

    for lib_path in cudnn_libs:
        try:
            ctypes.cdll.LoadLibrary(lib_path)
            logger.debug("Pre-loaded %s", lib_path)
        except OSError:
            logger.warning("Failed to load %s", lib_path, exc_info=True)

    _cudnn_loaded = True

    return True


def get_gpu_count() -> int:
    """
    Get number of available CUDA GPUs as a power of 2.

    Many models require tensor parallelism to use power-of-2 GPU counts.
    This returns the largest power of 2 <= available GPU count.

    Returns:
        Power of 2 GPU count, minimum 1.

    Raises:
        RuntimeError: If no CUDA GPUs are detected.
    """
    count = torch.cuda.device_count()
    if count == 0:
        msg = "No CUDA GPUs detected. At least one GPU is required for vLLM inference."
        raise RuntimeError(msg)
    tp_size = 2 ** int(math.log2(count)) if count >= 2 else 1  # noqa: PLR2004
    loguru_logger.info(
        f"Detected {count} GPU(s), using tensor_parallel_size={tp_size}"
    )
    return tp_size


def get_max_model_len_from_config(model: str, cache_dir: str | None = None) -> int | None:
    """
    Try to get max model length from HuggingFace AutoConfig.

    Args:
        model: Model identifier (e.g., "microsoft/phi-4")
        cache_dir: Optional cache directory for model config.

    Returns:
        Max model length if found, None otherwise.
    """
    try:
        config = AutoConfig.from_pretrained(model, trust_remote_code=True, cache_dir=cache_dir)
    except (OSError, ValueError, ImportError) as e:
        loguru_logger.warning(f"Could not auto-detect max_model_len for {model}: {e}")
        return None
    max_len = (
        getattr(config, "max_position_embeddings", None)
        or getattr(config, "n_positions", None)
        or getattr(config, "max_sequence_length", None)
    )
    if max_len is not None:
        loguru_logger.info(f"Auto-detected max_model_len={max_len} for {model}")

    return max_len
