"""Print a local experiment environment report; no packages are installed or changed."""
import json
import platform
import sys

report = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch_installed": False,
    "gpu_benchmark_performed": False,
}
try:
    import torch
except ImportError:
    report["torch_status"] = "not installed"
else:
    report.update({
        "torch_installed": True,
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "hip_build": getattr(torch.version, "hip", None),
        "cuda_api_device_available": torch.cuda.is_available(),
        "devices": [],
    })
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(index)
            report["devices"].append({
                "index": index,
                "name": prop.name,
                "total_memory_bytes": prop.total_memory,
                "compute_capability": [prop.major, prop.minor],
            })
    if hasattr(torch.backends, "mps"):
        report["mps_available"] = torch.backends.mps.is_available()
print(json.dumps(report, ensure_ascii=False, indent=2))
