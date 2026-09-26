"""
Hardware Abstraction Layer (GPU-vendor agnostic).

Detects whatever GPU is actually installed (NVIDIA/AMD/Intel, via WMI) and
picks the compute backend that will run AI inference: ONNX Runtime's
DirectML Execution Provider (DirectX 12 compute). DirectML is NOT AMD-
specific — it's a Windows-native, vendor-agnostic compute API — so this same
code path already accelerates NVIDIA and Intel GPUs too, with no per-vendor
branching required. Falls back to multi-threaded CPU execution if no
DirectX 12 GPU is usable.

Module name/API kept as `amd_backend`/`amd_hardware` for backward
compatibility with the rest of the codebase (engine.py, generative_engine.py,
status_bar.py, main_window.py, tests) even though detection itself is
generic — renaming would touch many call sites for no functional benefit.

torch-directml is intentionally NOT used here: it is in maintenance mode and
does not publish builds for current PyTorch/Python versions, so it cannot be
relied on as the GPU path going forward.

Note on NVIDIA specifically: this gives GPU acceleration via DirectML, not
native CUDA/TensorRT. A CUDA-specific path would need bundling a second,
much larger onnxruntime-gpu build alongside onnxruntime-directml (the two
can't coexist in one Python env — they're mutually exclusive wheels under
the same `onnxruntime` import name) and is a real engineering project with
its own driver/CUDA-version fragility; deliberately out of scope here in
favor of the simpler, already-working cross-vendor DirectML path.
"""

import json
import logging
import platform
import re
import subprocess
import sys

import psutil

logger = logging.getLogger("RadeonVideoAI.Backend")

# Avoid a flashing console window when this windowed app launches powershell.
_NO_WINDOW_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

_NVIDIA_KEYWORDS = ("nvidia", "geforce", "rtx", "gtx", "quadro", "tesla")
_AMD_KEYWORDS = ("amd", "radeon")
_INTEL_KEYWORDS = ("intel", "arc(tm)", "iris")


def _detect_vendor(gpu_name: str) -> str:
    name = gpu_name.lower()
    if any(k in name for k in _NVIDIA_KEYWORDS):
        return "NVIDIA"
    if any(k in name for k in _AMD_KEYWORDS):
        return "AMD"
    if any(k in name for k in _INTEL_KEYWORDS):
        return "Intel"
    return "Desconocido"


class AMDHardwareManager:
    """Detects the system GPU (any vendor)/VRAM and the active onnxruntime execution provider."""

    def __init__(self):
        self._gpu_name = "Detectando GPU..."
        self._vram_total_mb = 0
        self._vendor = "Desconocido"
        self._cpu_name = platform.processor() or "CPU"
        self._backend_name = "CPU"
        self._providers = ["CPUExecutionProvider"]

        self._detect_system_gpus()
        self._detect_compute_backend()

    def _detect_system_gpus(self):
        """
        Queries WMI (Win32_VideoController) for the primary GPU adapter name,
        then reads the real VRAM size from the driver's registry key
        (HardwareInformation.qwMemorySize, a 64-bit value), since WMI's
        AdapterRAM field is 32-bit and clamps/wraps on any GPU with 4GB+ VRAM
        regardless of vendor (e.g. it reports ~4095MB for a 16GB card).
        """
        try:
            cmd = [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5,
                                  creationflags=_NO_WINDOW_FLAGS)
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout.strip())
                if isinstance(data, dict):
                    data = [data]

                # Prefer a discrete GPU (NVIDIA/AMD/Intel Arc) over an
                # integrated/basic adapter if more than one is listed.
                discrete_candidate = None
                fallback_candidate = None
                for item in data:
                    name = item.get("Name", "")
                    ram = item.get("AdapterRAM", 0) or 0
                    ram_mb = int(ram) // (1024 * 1024) if ram else 0

                    if _detect_vendor(name) != "Desconocido":
                        discrete_candidate = (name, ram_mb)
                        break
                    elif ram_mb > 1024 and not fallback_candidate:
                        fallback_candidate = (name, ram_mb)

                target = discrete_candidate or fallback_candidate or (data[0].get("Name", "Generic GPU"), 4096)
                self._gpu_name = target[0]
                self._vendor = _detect_vendor(self._gpu_name)

                registry_mb = self._read_vram_from_registry()
                nvidia_smi_mb = self._read_vram_from_nvidia_smi() if self._vendor == "NVIDIA" else 0
                wmi_looks_clamped = target[1] <= 4096

                if nvidia_smi_mb:
                    self._vram_total_mb = nvidia_smi_mb
                elif registry_mb and registry_mb > target[1]:
                    self._vram_total_mb = registry_mb
                elif wmi_looks_clamped:
                    # WMI's 32-bit field clamped and no better reading was
                    # available from the registry/vendor tool. Default
                    # conservatively rather than guessing a specific card's
                    # VRAM size — the tiling safety margin degrades tile
                    # size automatically if this turns out too optimistic,
                    # but it can't recover from starting too high.
                    logger.warning(
                        "No se pudo confirmar la VRAM real (WMI truncado); "
                        "usando un valor conservador por defecto."
                    )
                    self._vram_total_mb = 8192
                else:
                    self._vram_total_mb = target[1]

                logger.info(f"GPU detectada: {self._gpu_name} [{self._vendor}] (VRAM: {self._vram_total_mb} MB)")
        except Exception as e:
            logger.warning(f"No se pudo consultar WMI para la GPU: {e}")
            self._gpu_name = "GPU no identificada"
            self._vram_total_mb = 8192
            self._vendor = "Desconocido"

    @staticmethod
    def _read_vram_from_registry() -> int:
        """Reads HardwareInformation.qwMemorySize (bytes, 64-bit) from the GPU
        driver's registry class key, returning VRAM in MB, or 0 if unavailable.
        Populated by the driver of whichever vendor's GPU is installed."""
        try:
            ps_script = (
                "$base = 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\"
                "{4d36e968-e325-11ce-bfc1-08002be10318}'; "
                "Get-ChildItem $base -ErrorAction SilentlyContinue | ForEach-Object { "
                "  $p = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue; "
                "  if ($p.'HardwareInformation.qwMemorySize') { $p.'HardwareInformation.qwMemorySize' } "
                "}"
            )
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True, text=True, timeout=5,
                creationflags=_NO_WINDOW_FLAGS
            )
            values = []
            for line in res.stdout.splitlines():
                line = line.strip()
                if line.isdigit():
                    values.append(int(line))
            if values:
                return max(values) // (1024 * 1024)
        except Exception:
            pass
        return 0

    @staticmethod
    def _read_vram_from_nvidia_smi() -> int:
        """On NVIDIA hardware, nvidia-smi ships with every driver install and
        reports exact VRAM directly — more reliable than registry/WMI
        heuristics when available, so it's tried first for NVIDIA GPUs."""
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
                creationflags=_NO_WINDOW_FLAGS
            )
            match = re.search(r"\d+", res.stdout)
            if res.returncode == 0 and match:
                return int(match.group())
        except Exception:
            pass
        return 0

    def _detect_compute_backend(self):
        """
        Picks the onnxruntime execution provider that will run AI inference:
        DirectML (GPU, DirectX 12 — works on NVIDIA/AMD/Intel alike) first,
        multi-threaded CPU otherwise.
        """
        try:
            import onnxruntime as ort
            available = set(ort.get_available_providers())
        except ImportError:
            logger.error("onnxruntime no está instalado. Instala 'onnxruntime-directml'.")
            available = set()

        if "DmlExecutionProvider" in available:
            self._providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            self._backend_name = f"DirectML (DirectX 12) - {self._gpu_name}"
            logger.info(f"Aceleración ONNX Runtime DirectML activada para GPU {self._vendor} ({self._gpu_name}).")
        else:
            self._providers = ["CPUExecutionProvider"]
            cores = psutil.cpu_count(logical=False) or 8
            self._backend_name = f"CPU Multi-Threading ({self._cpu_name}, {cores} núcleos)"
            logger.warning(
                "DmlExecutionProvider no disponible; usando CPU. "
                "Instala/actualiza 'onnxruntime-directml' y el driver de GPU más reciente."
            )

    @property
    def providers(self) -> list:
        return self._providers

    @property
    def gpu_name(self) -> str:
        return self._gpu_name

    @property
    def vram_total_mb(self) -> int:
        return self._vram_total_mb

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def vendor(self) -> str:
        """GPU vendor: 'NVIDIA', 'AMD', 'Intel', or 'Desconocido'."""
        return self._vendor

    @property
    def is_amd(self) -> bool:
        return self._vendor == "AMD"

    def get_hardware_summary(self) -> dict:
        return {
            "gpu_name": self._gpu_name,
            "vendor": self._vendor,
            "vram_total_mb": self._vram_total_mb,
            "backend": self._backend_name,
            "providers": self._providers,
            "is_amd": self.is_amd,
            "cpu": self._cpu_name,
        }


# Global hardware manager singleton instance
amd_hardware = AMDHardwareManager()
