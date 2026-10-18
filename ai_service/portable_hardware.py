"""Best-effort local hardware discovery for the portable launcher.

Only the Python standard library is used.  Detection never starts inference,
downloads software, changes device settings, or writes files.  GPU availability
is a hint: Ollama's actual runner decides which devices it can use.
"""

from __future__ import annotations

import csv
import ctypes
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
from typing import Any


_MIB = 1024 * 1024
_COMMAND_TIMEOUT_SEC = 8


def _run(arguments: list[str]) -> str | None:
    """Run a fixed local diagnostic command, never through a command shell."""
    options: dict[str, Any] = {}
    if os.name == "nt":
        options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=_COMMAND_TIMEOUT_SEC,
            check=False,
            **options,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode:
        return None
    return completed.stdout.decode("utf-8-sig", errors="replace").strip()


def _number(value: Any) -> int | None:
    try:
        result = int(float(str(value).strip()))
    except (ValueError, TypeError, OverflowError):
        return None
    return result if result >= 0 else None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _windows_ram() -> tuple[int | None, int | None]:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.total_physical // _MIB, status.available_physical // _MIB
    except (AttributeError, OSError):
        pass
    return None, None


def _linux_ram() -> tuple[int | None, int | None]:
    fields: dict[str, int] = {}
    for line in (_read_text(Path("/proc/meminfo")) or "").splitlines():
        match = re.match(r"^(\w+):\s+(\d+)\s+kB$", line)
        if match:
            fields[match.group(1)] = int(match.group(2))
    total = fields.get("MemTotal")
    available = fields.get("MemAvailable")
    if available is None and "MemFree" in fields:
        available = sum(fields.get(key, 0) for key in ("MemFree", "Buffers", "Cached"))
    return (
        total // 1024 if total is not None else None,
        available // 1024 if available is not None else None,
    )


def _mac_ram() -> tuple[int | None, int | None]:
    total_bytes = _number(_run(["/usr/sbin/sysctl", "-n", "hw.memsize"]))
    statistics = _run(["/usr/bin/vm_stat"]) or ""
    page_match = re.search(r"page size of (\d+) bytes", statistics)
    available_mib = None
    if page_match:
        pages: dict[str, int] = {}
        for line in statistics.splitlines():
            match = re.match(r"^([^:]+):\s+(\d+)\.?$", line)
            if match:
                pages[match.group(1)] = int(match.group(2))
        if "Pages free" in pages:
            # Reclaimable estimate; this is not a guarantee that a model fits.
            reclaimable = sum(
                pages.get(key, 0)
                for key in ("Pages free", "Pages inactive", "Pages speculative")
            )
            available_mib = reclaimable * int(page_match.group(1)) // _MIB
    return total_bytes // _MIB if total_bytes is not None else None, available_mib


def _nvidia_gpus() -> list[dict[str, Any]]:
    candidates: list[str] = []
    on_path = shutil.which("nvidia-smi")
    if on_path:
        candidates.append(on_path)
    if os.name == "nt":
        candidates.extend(
            [
                str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"),
                str(Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"),
            ]
        )
    for executable in dict.fromkeys(candidates):
        if not Path(executable).is_file():
            continue
        output = _run(
            [executable, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"]
        )
        if not output:
            continue
        devices = []
        for row in csv.reader(output.splitlines()):
            if len(row) != 3:
                continue
            total = _number(row[1])
            devices.append(
                {
                    "vendor": "NVIDIA",
                    "name": row[0].strip(),
                    "total_mib": total,
                    "free_mib": _number(row[2]),
                    "backend": "cuda",
                    "source": "nvidia-smi",
                    "memory_reliable": total is not None,
                }
            )
        if devices:
            return devices
    return []


def _windows_gpus() -> list[dict[str, Any]]:
    executable = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
    if not executable:
        fallback = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        executable = str(fallback) if fallback.is_file() else None
    if not executable:
        return []
    # Static script, with no interpolated user input or environment values.
    script = (
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); "
        "@(Get-CimInstance Win32_VideoController -ErrorAction Stop | "
        "Select-Object Name,AdapterRAM,AdapterCompatibility) | ConvertTo-Json -Compress"
    )
    output = _run([executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script])
    try:
        values = json.loads(output or "[]")
    except (ValueError, TypeError):
        return []
    if isinstance(values, dict):
        values = [values]
    if not isinstance(values, list):
        return []
    devices = []
    for value in values:
        if not isinstance(value, dict):
            continue
        name = str(value.get("Name") or "Unknown display adapter")
        identity = (name + " " + str(value.get("AdapterCompatibility") or "")).lower()
        vendor = "AMD" if any(word in identity for word in ("amd", "radeon", "advanced micro")) else "NVIDIA" if "nvidia" in identity else "Intel" if "intel" in identity else "unknown"
        reported_bytes = _number(value.get("AdapterRAM"))
        devices.append(
            {
                "vendor": vendor,
                "name": name,
                # Win32_VideoController.AdapterRAM is a 32-bit field, often
                # reports shared memory or wraps.  Never select a profile by it.
                "total_mib": None,
                "reported_adapter_ram_mib": reported_bytes // _MIB if reported_bytes else None,
                "free_mib": None,
                "backend": "cuda_unverified" if vendor == "NVIDIA" else "ollama_auto",
                "source": "windows_cim",
                "memory_reliable": False,
            }
        )
    return devices


def _linux_gpus() -> list[dict[str, Any]]:
    devices = []
    for card in sorted(Path("/sys/class/drm").glob("card*")):
        if not re.fullmatch(r"card\d+", card.name):
            continue
        device = card / "device"
        vendor_id = _read_text(device / "vendor")
        if vendor_id not in {"0x1002", "0x8086", "0x10de"}:
            continue
        vendor = {"0x1002": "AMD", "0x8086": "Intel", "0x10de": "NVIDIA"}[vendor_id]
        total_bytes = _number(_read_text(device / "mem_info_vram_total"))
        used_bytes = _number(_read_text(device / "mem_info_vram_used"))
        name = _read_text(device / "product_name") or f"{vendor} PCI {_read_text(device / 'device') or card.name}"
        devices.append(
            {
                "vendor": vendor,
                "name": name,
                "total_mib": total_bytes // _MIB if total_bytes else None,
                "free_mib": max(0, total_bytes - used_bytes) // _MIB if total_bytes is not None and used_bytes is not None else None,
                "backend": "ollama_auto",
                "source": "linux_sysfs",
                "memory_reliable": bool(total_bytes),
            }
        )
    return devices


def _mac_gpus(total_ram_mib: int | None, architecture: str) -> list[dict[str, Any]]:
    output = _run(["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"])
    try:
        displays = json.loads(output or "{}").get("SPDisplaysDataType", [])
    except (ValueError, TypeError, AttributeError):
        displays = []
    devices = []
    if isinstance(displays, list):
        for display in displays:
            if not isinstance(display, dict):
                continue
            name = str(display.get("sppci_model") or display.get("_name") or "Mac display adapter")
            is_apple = architecture.lower() in {"arm64", "aarch64"} or "apple" in name.lower()
            vendor = "Apple" if is_apple else "AMD" if any(x in name.lower() for x in ("amd", "radeon")) else "Intel" if "intel" in name.lower() else "unknown"
            devices.append(
                {
                    "vendor": vendor,
                    "name": name,
                    "total_mib": None,
                    "free_mib": None,
                    "unified_memory_mib": total_ram_mib if is_apple else None,
                    "backend": "metal" if is_apple else "ollama_auto",
                    "source": "system_profiler",
                    "memory_reliable": False,
                }
            )
    if not devices and architecture.lower() in {"arm64", "aarch64"}:
        chip = _run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple Silicon"
        devices.append(
            {
                "vendor": "Apple", "name": chip, "total_mib": None,
                "free_mib": None, "unified_memory_mib": total_ram_mib,
                "backend": "metal", "source": "architecture_sysctl",
                "memory_reliable": False,
            }
        )
    return devices


def detect_hardware() -> dict[str, Any]:
    """Return JSON-serializable best-effort diagnostics; unknown values are null."""
    system = platform.system()
    architecture = platform.machine()
    notes: list[str] = []
    if system == "Windows":
        total, available = _windows_ram()
    elif system == "Linux":
        total, available = _linux_ram()
    elif system == "Darwin":
        total, available = _mac_ram()
        notes.append("Available RAM on macOS is an estimate including reclaimable pages.")
    else:
        total, available = None, None
    if total is not None and available is not None:
        available = min(total, available)
    if total is None:
        notes.append("Total physical RAM could not be detected.")

    gpus = _nvidia_gpus()
    additional = _windows_gpus() if system == "Windows" else _linux_gpus() if system == "Linux" else _mac_gpus(total, architecture) if system == "Darwin" else []
    known_names = {device["name"].casefold() for device in gpus}
    for device in additional:
        if device["name"].casefold() in known_names:
            continue
        # sysfs exposes PCI identifiers instead of NVIDIA names; avoid listing
        # the same adapter twice when nvidia-smi already supplied its memory.
        if device["vendor"] == "NVIDIA" and any(gpu["vendor"] == "NVIDIA" for gpu in gpus):
            continue
        gpus.append(device)
        known_names.add(device["name"].casefold())
    if not gpus:
        notes.append("No GPU was detected; this does not prove that no GPU is installed.")
    if any(device["source"] == "windows_cim" for device in gpus):
        notes.append("Windows AdapterRAM values are informational and are not used as reliable VRAM sizes.")
    if any(device["vendor"] == "AMD" for device in gpus):
        notes.append("AMD support depends on the installed Ollama version and GPU; detection does not force CPU mode.")

    project_path = Path(__file__).resolve().parent
    try:
        usage = shutil.disk_usage(project_path)
        disk = {"path": str(project_path), "total_mib": usage.total // _MIB, "free_mib": usage.free // _MIB}
    except OSError:
        disk = {"path": str(project_path), "total_mib": None, "free_mib": None}
        notes.append("Free disk space could not be detected.")
    return {
        "platform": system,
        "architecture": architecture,
        "python_bits": struct.calcsize("P") * 8,
        "ram": {"total_mib": total, "available_mib": available},
        "disk": disk,
        "gpus": gpus,
        "detection_notes": notes,
    }


def select_profile(hardware: dict[str, Any]) -> dict[str, Any]:
    """Select resource settings without silently changing the analysed inputs.

    All profiles retain the same model, context and analysis timeout.  A weaker
    computer may be unable to complete a video in the existing job budget;
    that remains an explicit processing failure, never a clean verdict.
    """
    ram = hardware.get("ram") or {}
    total = _number(ram.get("total_mib"))
    available = _number(ram.get("available_mib"))
    disk_free = _number((hardware.get("disk") or {}).get("free_mib"))
    gpus = hardware.get("gpus") or []
    reliable_gpu_sizes = [
        size for gpu in gpus if gpu.get("memory_reliable")
        and (size := _number(gpu.get("total_mib"))) is not None
    ]
    enough_vram = bool(reliable_gpu_sizes and max(reliable_gpu_sizes) >= 4096)
    apple_metal = any(gpu.get("backend") == "metal" for gpu in gpus)
    enough_ram = (total is not None and total >= 12 * 1024
                  and (available is None or available >= 4 * 1024))
    warnings: list[str] = []
    recommendations: list[str] = []
    if hardware.get("python_bits") != 64:
        warnings.append("Нужен 64-битный Python.")
    if total is None:
        warnings.append("Объём RAM не определён; достаточность памяти для модели неизвестна.")
    elif total < 8 * 1024:
        warnings.append("RAM меньше 8 ГиБ: модель может не поместиться в память или превысить лимит обработки.")
    elif total < 12 * 1024:
        warnings.append("RAM меньше 12 ГиБ: запас памяти для модели, API и ОС небольшой.")
    if available is not None and available < 4 * 1024:
        recommendations.append("Свободно меньше 4 ГиБ RAM. Перед анализом закройте лишние программы.")
    if disk_free is None:
        recommendations.append("Проверьте свободное место перед скачиванием весов: объём не определён.")
    elif disk_free < 6 * 1024:
        recommendations.append("Свободно меньше 6 ГиБ на диске проекта. Нужен запас для зависимостей и временных файлов; Ollama может хранить веса на другом диске.")
    if any(gpu.get("vendor") == "AMD" for gpu in gpus):
        recommendations.append("Использование AMD GPU нужно подтвердить по работающему backend Ollama после реального запроса.")

    if enough_ram and (enough_vram or apple_metal):
        name = "standard"
        cache_ram_mib = 512
    else:
        name = "conservative"
        cache_ram_mib = 128
        warnings.append("Памяти мало или её объём неизвестен. Обработка в пределах текущего лимита времени не гарантируется.")
    if apple_metal:
        backend_hint = "metal"
    elif any(gpu.get("backend") == "cuda" for gpu in gpus):
        backend_hint = "cuda"
    elif gpus:
        backend_hint = "ollama_auto"
    else:
        backend_hint = "cpu_or_undetected_gpu"
    return {
        "name": name,
        "backend_hint": backend_hint,
        "cache_ram_mib": cache_ram_mib,
        "concurrency": 1,
        "num_ctx": 6144,
        "request_timeout_sec": 180,
        "warnings": warnings,
        "recommendations": recommendations,
    }


if __name__ == "__main__":
    detected = detect_hardware()
    print(json.dumps({"hardware": detected, "profile": select_profile(detected)}, ensure_ascii=False, indent=2))
