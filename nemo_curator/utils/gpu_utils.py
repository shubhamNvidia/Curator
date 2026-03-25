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
import os

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

    # Update LD_LIBRARY_PATH so that any *subsequent* dlopen() calls by
    # onnxruntime (or other native libraries) can resolve cuDNN symbols.
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if cudnn_lib_dir not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = cudnn_lib_dir + (":" + ld_path if ld_path else "")

    # Eagerly load cuDNN shared libraries into the process address space.
    # Setting LD_LIBRARY_PATH alone is not enough once the process has started
    # because the dynamic linker caches its search paths at startup.
    # ONNX Runtime's CUDA provider uses dlopen() for sub-libraries like
    # libcudnn_adv.so.9, so we must pre-load all of them.
    import glob

    cudnn_libs = sorted(glob.glob(os.path.join(cudnn_lib_dir, "libcudnn*.so*")))
    if not cudnn_libs:
        logger.warning("No libcudnn*.so* files found in %s", cudnn_lib_dir)
        return False

    # Load the main library first (other libs depend on it), then the rest.
    # The main library matches "libcudnn.so.<version>" (no underscore after "libcudnn").
    cudnn_libs.sort(key=lambda p: (not os.path.basename(p).startswith("libcudnn.so."), p))

    for lib_path in cudnn_libs:
        try:
            ctypes.cdll.LoadLibrary(lib_path)
            logger.debug("Pre-loaded %s", lib_path)
        except OSError:
            logger.warning("Failed to load %s", lib_path, exc_info=True)

    _cudnn_loaded = True

    return True
