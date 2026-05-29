from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildSummary:
    command: str
    output_dir: str
    message: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Build",
                f"Comando: {self.command}",
                f"Saida: {self.output_dir}",
                f"Mensagem: {self.message}",
            ]
        )


def build_executable(base_dir: Path, *, name: str = "lotofacil-analytics") -> BuildSummary:
    if importlib.util.find_spec("PyInstaller") is None:
        raise RuntimeError(
            "PyInstaller nao esta instalado. Instale com: python -m pip install pyinstaller"
        )

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name",
        name,
        "main.py",
    ]
    subprocess.run(command, cwd=str(base_dir), check=True)
    return BuildSummary(
        command=" ".join(command),
        output_dir=str(base_dir / "dist"),
        message="Executavel gerado com sucesso.",
    )
