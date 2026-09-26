from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, TextIO


def _natural_sort_key(name: str):
    return [
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in re.split(r"(\d+)", name)
    ]


def find_images(
    input_dir: Path,
    extensions: Sequence[str] = (".jpg", ".jpeg"),
) -> List[Path]:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    wanted = {ext.lower() for ext in extensions}
    images = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in wanted
    ]
    return sorted(images, key=lambda p: _natural_sort_key(p.name))


def _format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


class ProgressBar:
    def __init__(
        self,
        total: int,
        label: str = "Progress",
        width: int = 28,
        stream: Optional[TextIO] = None,
    ) -> None:
        self.total = max(int(total), 1)
        self.label = label
        self.width = max(int(width), 1)
        self.stream = stream if stream is not None else sys.stderr
        self.n = 0
        self.description = ""
        self._start = time.perf_counter()
        self._last_line_len = 0
        self._is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._render()

    def set_description(self, description: str) -> None:
        self.description = description or ""
        self._render()

    def update(self, n: int = 1, description: Optional[str] = None) -> None:
        self.n = min(self.n + n, self.total)
        if description is not None:
            self.description = description
        self._render()

    def write(self, message: str) -> None:
        if self._is_tty:
            self._clear_line()
        self.stream.write(message + "\n")
        self.stream.flush()
        if self._is_tty:
            self._render()

    def close(self) -> None:
        if self._is_tty:
            self._render()
            self.stream.write("\n")
        else:
            self.stream.write(self._compose_line() + "\n")
        self.stream.flush()

    # -- internal helpers -------------------------------------------------

    def _compose_line(self) -> str:
        frac = self.n / self.total
        filled = int(round(self.width * frac))
        bar = "#" * filled + "-" * (self.width - filled)
        pct = int(round(frac * 100))
        elapsed = time.perf_counter() - self._start
        line = f"{self.label} [{bar}] {self.n}/{self.total} {pct:3d}% {_format_duration(elapsed)}"
        if self.description:
            line += f" | {self.description}"
        return line

    def _clear_line(self) -> None:
        self.stream.write("\r" + " " * self._last_line_len + "\r")

    def _render(self) -> None:
        if not self._is_tty:
            return
        line = self._compose_line()
        self._clear_line()
        self.stream.write(line)
        self.stream.flush()
        self._last_line_len = len(line)
