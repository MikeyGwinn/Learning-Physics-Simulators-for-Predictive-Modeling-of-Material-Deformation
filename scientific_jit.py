"""Keep compiled FEM forms isolated across heterogeneous HPC nodes."""
import hashlib
import json
import os
import platform
import sys
from pathlib import Path


def jit_options():
    import dolfinx
    from dolfinx.jit import get_options
    # Conda GCC can target a newer sysroot than the allocated HPC node.
    # Generated C extensions must link against this host's glibc/libm.
    compiler = Path("/usr/bin/gcc")
    if sys.platform.startswith("linux") and compiler.is_file():
        os.environ["CC"] = str(compiler)
        os.environ["LDSHARED"] = str(compiler) + " -shared"
    runtime = {"compiler": os.environ.get("CC"), "host": platform.node(), "libc": platform.libc_ver(),
               "machine": platform.machine(), "python": sys.version,
               "executable": str(Path(sys.executable).resolve()), "dolfinx": dolfinx.__version__}
    key = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()[:20]
    base = get_options()["cache_dir"]
    cache = base/"phasefield-nodes"/key
    cache.mkdir(parents=True, exist_ok=True)
    return {"cache_dir": cache, "timeout": 120}
