"""
AMD Hardware Abstraction Layer.
Detects the AMD Radeon GPU (name/VRAM via WMI) and the compute backend that
will actually run AI inference: ONNX Runtime's DirectML Execution Provider
(DirectX 12 compute, works on any modern AMD Radeon GPU including RDNA4 /
RX 9060 XT through the standard Adrenalin driver), falling back to
multi-threaded AMD Ryzen CPU execution if no DirectX 12 GPU is usable.

torch-directml is intentionally NOT used here: it is in maintenance mode and
does not publish builds for current PyTorch/Python versions, so it cannot be
relied on as the GPU path going forward.
"""

import json
import logging
import platform
import subprocess
import sys

import psutil

logger = logging.getLogger("RadeonVideoAI.Backend")

# Avoid a flashing console window when this windowed app launches powershell.
_NO_WINDOW_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


class AMDHardwareManager:
    """Detects AMD Radeon GPU/VRAM and the active onnxruntime execution provider."""

    def __init__(self):
        self._gpu_name = "Detectando GPU..."
        self._vram_total_mb = 0
        self._is_amd_gpu = False
        self._cpu_name = platform.processor() or "AMD Ryzen"
        self._backend_name = "CPU"
        self._providers = ["CPUExecutionProvider"]

        self._detect_system_gpus()
        self._detect_compute_backend()

    def _detect_system_gpus(self):
        """
        Queries WMI (Win32_VideoController) for the AMD Radeon adapter name,
        then reads the real VRAM size from the driver's registry key
        (HardwareInformation.qwMemorySize, a 64-bit value), since WMI's
        AdapterRAM field is 32-bit and clamps/wraps on any GPU with 4GB+ VRAM
        (e.g. it reports ~4095MB for a 16GB RX 9060 XT).
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

                amd_candidate = None
                fallback_candidate = None
                for item in data:
                    name = item.get("Name", "")
                    ram = item.get("AdapterRAM", 0) or 0
                    ram_mb = int(ram) // (1024 * 1024) if ram else 0

                    if "amd" in name.lower() or "radeon" in name.lower():
                        amd_candidate = (name, ram_mb)
                        break
                    elif ram_mb > 1024 and not fallback_candidate:
                        fallback_candidate = (name, ram_mb)

                target = amd_candidate or fallback_candidate or (data[0].get("Name", "Generic GPU"), 4096)
                self._gpu_name = target[0]
                self._is_amd_gpu = "radeon" in self._gpu_name.lower() or "amd" in self._gpu_name.lower()

                registry_mb = self._read_vram_from_registry()
                wmi_looks_clamped = target[1] <= 4096
                if registry_mb and registry_mb > target[1]:
                    self._vram_total_mb = registry_mb
                elif wmi_looks_clamped:
                    # Could not confirm via registry; assume the known target
                    # hardware (16GB RX 9060 XT) rather than the truncated 4GB.
                    self._vram_total_mb = 16384
                else:
                    self._vram_total_mb = target[1]

                logger.info(f"GPU detectada: {self._gpu_name} (VRAM: {self._vram_total_mb} MB)")
        except Exception as e:
            logger.warning(f"No se pudo consultar WMI para la GPU: {e}")
            self._gpu_name = "AMD Radeon Graphics"
            self._vram_total_mb = 16384
            self._is_amd_gpu = True

    @staticmethod
    def _read_vram_from_registry() -> int:
        """Reads HardwareInformation.qwMemorySize (bytes, 64-bit) from the GPU
        driver's registry class key, returning VRAM in MB, or 0 if unavailable."""
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

    def _detect_compute_backend(self):
        """
        Picks the onnxruntime execution provider that will run AI inference:
        DirectML (GPU, DirectX 12) first, multi-threaded Ryzen CPU otherwise.
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
            logger.info("Aceleración ONNX Runtime DirectML activada para GPU AMD Radeon.")
        else:
            self._providers = ["CPUExecutionProvider"]
            cores = psutil.cpu_count(logical=False) or 8
            self._backend_name = f"AMD Ryzen Multi-Threading (CPU Zen, {cores} núcleos)"
            logger.warning(
                "DmlExecutionProvider no disponible; usando CPU. "
                "Instala/actualiza 'onnxruntime-directml' y el driver AMD Adrenalin más reciente."
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
    def is_amd(self) -> bool:
        return self._is_amd_gpu

    def get_hardware_summary(self) -> dict:
        return {
            "gpu_name": self._gpu_name,
            "vram_total_mb": self._vram_total_mb,
            "backend": self._backend_name,
            "providers": self._providers,
            "is_amd": self._is_amd_gpu,
            "cpu": self._cpu_name,
        }


# Global hardware manager singleton instance
amd_hardware = AMDHardwareManager()
