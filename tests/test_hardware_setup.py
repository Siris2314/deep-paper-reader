import pytest

from paper_agent import hardware_setup
from paper_agent.hardware_setup import GpuInfo, HardwareInfo


def hardware_with_vram(vram_gb: float | None, *, system: str = "windows") -> HardwareInfo:
    gpus = [GpuInfo("nvidia", "Test GPU", vram_gb)] if vram_gb is not None else []
    return HardwareInfo(
        system=system,
        machine="x86_64",
        cpu_count=16,
        system_ram_gb=32.0,
        gpus=gpus,
        ollama_installed=True,
        ollama_reachable=True,
        ollama_version="ollama version 1.0",
        installed_models=["qwen3.5:4b", "qwen2.5:7b", "qwen2.5:3b"],
    )


@pytest.mark.parametrize(
    ("vram", "expected", "context"),
    [
        (None, "cpu", 4096),
        (8.0, "low-vram", 4096),
        (16.0, "balanced", 8192),
        (24.0, "high-vram", 32768),
    ],
)
def test_profile_selection_tracks_available_accelerator_memory(vram, expected, context):
    profile = hardware_setup.select_profile(hardware_with_vram(vram))

    assert profile.name == expected
    assert profile.context_length == context


def test_project_profile_preserves_secrets_and_writes_managed_settings(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "TAVILY_API_KEY=keep-this-secret\nOLLAMA_NUM_CTX=2048\nPAPER_READER_JUDGE_MODEL=old-model\n",
        encoding="utf-8",
    )
    profile_path = tmp_path / "agent_memory" / "hardware_profile.json"
    backup_path = tmp_path / "agent_memory" / "hardware_setup_env_backup.env"
    monkeypatch.setattr(hardware_setup, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hardware_setup, "PROFILE_PATH", profile_path)
    monkeypatch.setattr(hardware_setup, "ENV_BACKUP_PATH", backup_path)

    hardware = hardware_with_vram(16.0)
    profile = hardware_setup.select_profile(hardware)
    written = hardware_setup.apply_project_profile(hardware, profile)
    updated = env_path.read_text(encoding="utf-8")

    assert written == profile_path
    assert "TAVILY_API_KEY=keep-this-secret" in updated
    assert "OLLAMA_NUM_CTX=8192" in updated
    assert "PAPER_READER_JUDGE_MODEL=qwen2.5:7b" in updated
    assert backup_path.read_text(encoding="utf-8").startswith("TAVILY_API_KEY=keep-this-secret")


def test_missing_models_are_reported_as_pull_targets():
    hardware = HardwareInfo(
        **{
            **hardware_with_vram(16.0).__dict__,
            "installed_models": ["qwen3.5:4b"],
        }
    )
    profile = hardware_setup.select_profile(hardware)
    report = hardware_setup.hardware_report(hardware, profile)

    assert "qwen2.5:7b" in report["missingModels"]
    assert (
        "ollama pull" in report["serverInstructions"]
        or "paper-agent hardware" in report["serverInstructions"]
    )


def test_ollama_cli_models_are_used_when_server_is_offline(monkeypatch):
    monkeypatch.setattr(hardware_setup.shutil, "which", lambda name: "ollama.exe")
    monkeypatch.setattr(
        hardware_setup,
        "_run",
        lambda command, timeout=5.0: (
            "ollama version is 1.0"
            if "--version" in command
            else "NAME ID SIZE\nqwen3.5:4b abc 3 GB\nqwen2.5:3b def 2 GB"
        ),
    )

    class Offline:
        def __enter__(self):
            raise OSError("offline")

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(hardware_setup.urllib.request, "urlopen", lambda *args, **kwargs: Offline())
    installed, reachable, version, models = hardware_setup._ollama_status("http://localhost:11434")

    assert installed is True
    assert reachable is False
    assert version == "ollama version is 1.0"
    assert models == ["qwen2.5:3b", "qwen3.5:4b"]
