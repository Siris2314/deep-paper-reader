from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = PROJECT_ROOT / "agent_memory" / "hardware_profile.json"
ENV_BACKUP_PATH = PROJECT_ROOT / "agent_memory" / "hardware_setup_env_backup.env"


@dataclass(frozen=True)
class GpuInfo:
    vendor: str
    name: str
    vram_gb: float | None
    shared_memory: bool = False


@dataclass(frozen=True)
class HardwareInfo:
    system: str
    machine: str
    cpu_count: int
    system_ram_gb: float
    gpus: list[GpuInfo]
    ollama_installed: bool
    ollama_reachable: bool
    ollama_version: str | None
    installed_models: list[str]


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    rationale: str
    context_length: int
    max_loaded_models: int
    num_parallel: int
    flash_attention: bool
    kv_cache_type: str
    keep_alive: str
    concept_model: str
    concept_fallbacks: list[str]
    math_model: str
    math_fallbacks: list[str]
    judge_model: str
    judge_fallbacks: list[str]

    @property
    def required_models(self) -> list[str]:
        return list(
            dict.fromkeys(
                [
                    self.concept_model,
                    self.math_model,
                    self.judge_model,
                    *self.concept_fallbacks,
                    *self.math_fallbacks,
                    *self.judge_fallbacks,
                ]
            )
        )


def _run(command: list[str], timeout: float = 5.0) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _system_ram_gb() -> float:
    if platform.system() == "Windows":

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
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.total_physical / 1024**3, 1)
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / 1024**3, 1)
    except (AttributeError, OSError, ValueError):
        return 0.0


def _nvidia_gpus() -> list[GpuInfo]:
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[GpuInfo] = []
    for row in csv.reader(output.splitlines()):
        if len(row) < 2:
            continue
        try:
            vram = round(float(row[1].strip()) / 1024, 1)
        except ValueError:
            vram = None
        gpus.append(GpuInfo(vendor="nvidia", name=row[0].strip(), vram_gb=vram))
    return gpus


def _largest_numeric_value(value: Any, path: str = "") -> float | None:
    values: list[float] = []
    if isinstance(value, dict):
        for key, item in value.items():
            result = _largest_numeric_value(item, f"{path} {key}".lower())
            if (
                result is not None
                and "vram" in f"{path} {key}".lower()
                and "total" in f"{path} {key}".lower()
            ):
                values.append(result)
    elif isinstance(value, list):
        for item in value:
            result = _largest_numeric_value(item, path)
            if result is not None:
                values.append(result)
    elif isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, str):
        match = re.search(r"([0-9.]+)", value)
        if match:
            return float(match.group(1))
    return max(values) if values else None


def _amd_gpus() -> list[GpuInfo]:
    output = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if not output:
        return []
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return []
    gpus: list[GpuInfo] = []
    for key, item in payload.items() if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("Card series") or item.get("Card model") or key)
        raw_memory = _largest_numeric_value(item)
        vram = round(raw_memory / 1024**3, 1) if raw_memory and raw_memory > 1024**3 else None
        gpus.append(GpuInfo(vendor="amd", name=name, vram_gb=vram))
    return gpus


def _apple_gpu(system_ram_gb: float) -> list[GpuInfo]:
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        return []
    output = _run(["system_profiler", "SPHardwareDataType", "-json"])
    try:
        payload = json.loads(output)
        hardware = payload.get("SPHardwareDataType", [{}])[0]
        name = str(hardware.get("chip_type") or hardware.get("machine_name") or "Apple Silicon")
    except (json.JSONDecodeError, IndexError, TypeError):
        name = "Apple Silicon"
    # Unified memory is shared with the OS. Reserve 30 percent when choosing a profile.
    return [
        GpuInfo(
            vendor="apple", name=name, vram_gb=round(system_ram_gb * 0.7, 1), shared_memory=True
        )
    ]


def _windows_display_gpus() -> list[GpuInfo]:
    if platform.system() != "Windows":
        return []
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress",
    ]
    output = _run(command)
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return []
    rows = payload if isinstance(payload, list) else [payload]
    gpus: list[GpuInfo] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("Name") or "Unknown GPU")
        vendor = (
            "amd"
            if re.search(r"amd|radeon", name, re.I)
            else "intel"
            if "intel" in name.lower()
            else "other"
        )
        raw = row.get("AdapterRAM")
        try:
            vram = round(int(raw) / 1024**3, 1) if raw else None
        except (TypeError, ValueError):
            vram = None
        gpus.append(GpuInfo(vendor=vendor, name=name, vram_gb=vram))
    return gpus


def _ollama_status(base_url: str) -> tuple[bool, bool, str | None, list[str]]:
    installed = shutil.which("ollama") is not None
    version = _run(["ollama", "--version"]) if installed else ""
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/tags",
        headers={"Accept": "application/json", "User-Agent": "deep-paper-agent-hardware/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=1.5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = sorted(
            str(item.get("model") or item.get("name"))
            for item in payload.get("models", [])
            if isinstance(item, dict) and (item.get("model") or item.get("name"))
        )
        reachable = True
    except Exception:
        models = []
        reachable = False
    if installed and not models:
        rows = _run(["ollama", "list"]).splitlines()
        models = sorted(
            row.split()[0]
            for row in rows[1:]
            if row.strip() and row.split() and ":" in row.split()[0]
        )
    return installed, reachable, version or None, models


def detect_hardware(base_url: str = "http://localhost:11434") -> HardwareInfo:
    system_ram = _system_ram_gb()
    gpus = _nvidia_gpus()
    if not gpus:
        gpus = _amd_gpus()
    if not gpus:
        gpus = _apple_gpu(system_ram)
    if not gpus:
        gpus = _windows_display_gpus()
    installed, reachable, version, models = _ollama_status(base_url)
    return HardwareInfo(
        system=platform.system().lower(),
        machine=platform.machine(),
        cpu_count=os.cpu_count() or 1,
        system_ram_gb=system_ram,
        gpus=gpus,
        ollama_installed=installed,
        ollama_reachable=reachable,
        ollama_version=version,
        installed_models=models,
    )


def _profile(name: str) -> HardwareProfile:
    if name == "high-vram":
        return HardwareProfile(
            name=name,
            rationale="At least 24 GB accelerator memory is available; use Ollama's 32K tier.",
            context_length=32768,
            max_loaded_models=2,
            num_parallel=1,
            flash_attention=True,
            kv_cache_type="q8_0",
            keep_alive="15m",
            concept_model="qwen2.5:7b",
            concept_fallbacks=["qwen2.5:3b"],
            math_model="qwen3.5:4b",
            math_fallbacks=["qwen2.5:7b", "qwen2.5:3b"],
            judge_model="qwen2.5:7b",
            judge_fallbacks=["qwen2.5:3b"],
        )
    if name == "balanced":
        return HardwareProfile(
            name=name,
            rationale="12-23 GB accelerator memory can keep the two primary reader models resident at 8K.",
            context_length=8192,
            max_loaded_models=2,
            num_parallel=1,
            flash_attention=True,
            kv_cache_type="q8_0",
            keep_alive="15m",
            concept_model="qwen2.5:7b",
            concept_fallbacks=["qwen2.5:3b"],
            math_model="qwen3.5:4b",
            math_fallbacks=["qwen2.5:7b", "qwen2.5:3b"],
            judge_model="qwen2.5:7b",
            judge_fallbacks=["qwen2.5:3b"],
        )
    if name == "low-vram":
        return HardwareProfile(
            name=name,
            rationale="Under 12 GB accelerator memory favors one resident model at a time and a 4K context.",
            context_length=4096,
            max_loaded_models=1,
            num_parallel=1,
            flash_attention=True,
            kv_cache_type="q8_0",
            keep_alive="5m",
            concept_model="qwen2.5:3b",
            concept_fallbacks=[],
            math_model="qwen3.5:4b",
            math_fallbacks=["qwen2.5:3b"],
            judge_model="qwen2.5:3b",
            judge_fallbacks=[],
        )
    return HardwareProfile(
        name="cpu",
        rationale="No supported accelerator memory was detected; minimize model and context memory.",
        context_length=4096,
        max_loaded_models=1,
        num_parallel=1,
        flash_attention=False,
        kv_cache_type="f16",
        keep_alive="5m",
        concept_model="qwen2.5:3b",
        concept_fallbacks=[],
        math_model="qwen3.5:4b",
        math_fallbacks=["qwen2.5:3b"],
        judge_model="qwen2.5:3b",
        judge_fallbacks=[],
    )


def select_profile(hardware: HardwareInfo, requested: str = "auto") -> HardwareProfile:
    if requested != "auto":
        return _profile(requested)
    capacities = [gpu.vram_gb for gpu in hardware.gpus if gpu.vram_gb]
    capacity = max(capacities) if capacities else 0.0
    if capacity >= 24:
        selected = "high-vram"
    elif capacity >= 12:
        selected = "balanced"
    elif capacity > 0:
        selected = "low-vram"
    else:
        selected = "cpu"
    profile = _profile(selected)
    if hardware.system == "windows" and any(gpu.vendor == "amd" for gpu in hardware.gpus):
        profile = HardwareProfile(**{**asdict(profile), "max_loaded_models": 1})
    return profile


def profile_environment(profile: HardwareProfile) -> dict[str, str]:
    return {
        "MODEL_PROVIDER": "ollama",
        "MODEL_NAME": profile.concept_model,
        "OLLAMA_MODEL": profile.concept_model,
        "OLLAMA_NUM_CTX": str(profile.context_length),
        "PAPER_READER_AGENT_MODEL": profile.concept_model,
        "PAPER_READER_AGENT_FALLBACK_MODELS": ",".join(profile.concept_fallbacks),
        "PAPER_READER_AGENT_KEEP_ALIVE": profile.keep_alive,
        "PAPER_READER_MATH_MODEL": profile.math_model,
        "PAPER_READER_MATH_FALLBACK_MODELS": ",".join(profile.math_fallbacks),
        "PAPER_READER_JUDGE_MODEL": profile.judge_model,
        "PAPER_READER_JUDGE_FALLBACK_MODELS": ",".join(profile.judge_fallbacks),
    }


def server_environment(profile: HardwareProfile) -> dict[str, str]:
    return {
        "OLLAMA_FLASH_ATTENTION": "1" if profile.flash_attention else "0",
        "OLLAMA_KV_CACHE_TYPE": profile.kv_cache_type,
        "OLLAMA_MAX_LOADED_MODELS": str(profile.max_loaded_models),
        "OLLAMA_NUM_PARALLEL": str(profile.num_parallel),
        "OLLAMA_CONTEXT_LENGTH": str(profile.context_length),
        "OLLAMA_KEEP_ALIVE": profile.keep_alive,
    }


def _update_env_file(path: Path, values: dict[str, str]) -> None:
    source = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = source.splitlines()
    replaced: set[str] = set()
    output: list[str] = []
    for line in lines:
        match = re.match(r"^([A-Z][A-Z0-9_]*)=", line)
        if match and match.group(1) in values:
            key = match.group(1)
            output.append(f"{key}={values[key]}")
            replaced.add(key)
        else:
            output.append(line)
    missing = [key for key in values if key not in replaced]
    if missing:
        output.extend(["", "# Managed by paper-agent hardware setup"])
        output.extend(f"{key}={values[key]}" for key in missing)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def apply_project_profile(hardware: HardwareInfo, profile: HardwareProfile) -> Path:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        example = PROJECT_ROOT / ".env.example"
        if example.exists():
            shutil.copy2(example, env_path)
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if env_path.exists() and not ENV_BACKUP_PATH.exists():
        shutil.copy2(env_path, ENV_BACKUP_PATH)
    _update_env_file(env_path, profile_environment(profile))
    installed = set(hardware.installed_models)
    payload = {
        "version": 1,
        "configuredAt": datetime.now(timezone.utc).isoformat(),
        "hardware": asdict(hardware),
        "profile": asdict(profile),
        "serverEnvironment": server_environment(profile),
        "missingModels": [model for model in profile.required_models if model not in installed],
    }
    PROFILE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return PROFILE_PATH


def apply_windows_server_profile(profile: HardwareProfile) -> None:
    if platform.system() != "Windows":
        raise RuntimeError(
            "Automatic Ollama server environment setup is currently supported on Windows only."
        )
    script = PROJECT_ROOT / "scripts" / "configure_ollama_windows.ps1"
    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-Action",
        "Apply",
        "-KvCacheType",
        profile.kv_cache_type,
        "-FlashAttention",
        "1" if profile.flash_attention else "0",
        "-ContextLength",
        str(profile.context_length),
        "-MaxLoadedModels",
        str(profile.max_loaded_models),
        "-NumParallel",
        str(profile.num_parallel),
        "-KeepAlive",
        profile.keep_alive,
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Ollama Windows profile script failed with exit code {result.returncode}."
        )


def server_setup_instructions(hardware: HardwareInfo, profile: HardwareProfile) -> str:
    values = server_environment(profile)
    if hardware.system == "windows":
        return (
            "Run: paper-agent hardware --apply --apply-server, then fully quit and reopen Ollama."
        )
    if hardware.system == "darwin":
        commands = "\n".join(f"launchctl setenv {key} {value}" for key, value in values.items())
        return f"Set the Ollama app environment, then restart the app:\n{commands}"
    lines = "\n".join(f'Environment="{key}={value}"' for key, value in values.items())
    return (
        "Create a systemd override with `systemctl edit ollama`, add the lines below under [Service], "
        f"then run `systemctl daemon-reload` and `systemctl restart ollama`:\n[Service]\n{lines}"
    )


def hardware_report(hardware: HardwareInfo, profile: HardwareProfile) -> dict[str, Any]:
    installed = set(hardware.installed_models)
    return {
        "hardware": asdict(hardware),
        "profile": asdict(profile),
        "serverEnvironment": server_environment(profile),
        "missingModels": [model for model in profile.required_models if model not in installed],
        "serverInstructions": server_setup_instructions(hardware, profile),
    }


def print_report(report: dict[str, Any]) -> None:
    hardware = report["hardware"]
    profile = report["profile"]
    print("Deep Paper Reader hardware check")
    print(f"OS: {hardware['system']} {hardware['machine']}")
    print(f"CPU threads: {hardware['cpu_count']} | System RAM: {hardware['system_ram_gb']} GB")
    if hardware["gpus"]:
        for gpu in hardware["gpus"]:
            memory = f"{gpu['vram_gb']} GB" if gpu.get("vram_gb") else "VRAM unknown"
            suffix = " shared" if gpu.get("shared_memory") else ""
            print(f"GPU: {gpu['name']} ({gpu['vendor']}, {memory}{suffix})")
    else:
        print("GPU: no supported accelerator memory detected")
    print(
        f"Ollama: {'reachable' if hardware['ollama_reachable'] else 'not reachable'}"
        f" | {hardware.get('ollama_version') or 'version unavailable'}"
    )
    print(f"Selected profile: {profile['name']} | context {profile['context_length']} tokens")
    print(profile["rationale"])
    print(
        f"Models: math={profile['math_model']} | concept={profile['concept_model']} | "
        f"judge={profile['judge_model']}"
    )
    if report["missingModels"]:
        print("Missing models:")
        for model in report["missingModels"]:
            print(f"  ollama pull {model}")
    print("\nOllama server setup:")
    print(report["serverInstructions"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect hardware and configure Deep Paper Reader safely."
    )
    parser.add_argument(
        "--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    parser.add_argument(
        "--profile", choices=["auto", "cpu", "low-vram", "balanced", "high-vram"], default="auto"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Update managed model settings in .env and save the profile.",
    )
    parser.add_argument(
        "--apply-server",
        action="store_true",
        help="Apply the selected Ollama server profile on Windows.",
    )
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hardware = detect_hardware(args.base_url)
    profile = select_profile(hardware, args.profile)
    report = hardware_report(hardware, profile)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    if args.apply:
        path = apply_project_profile(hardware, profile)
        if not args.json:
            print(f"\nProject profile saved to {path}")
    if args.apply_server:
        apply_windows_server_profile(profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
