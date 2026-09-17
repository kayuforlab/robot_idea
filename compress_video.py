"""指定した動画（URLまたはローカルファイル）を圧縮する。

実行方法:
    uv run compress_video.py
"""

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import imageio_ffmpeg
import requests

# 圧縮したい動画をここに入れる。http(s)://... のURLでも、
# ローカルファイルパス（WSLの /mnt/c/... でもWindows形式の C:/... でもよい）でも指定できる。
VIDEO_URL = "C:/Users/ms8/Videos/画面録画/movie.mp4"

# 圧縮の強さ（CRF値）。18=高画質/大きい 〜 32=低画質/小さい。23前後が標準的な目安。
CRF = 28

# 出力解像度の高さ(px)。Noneなら元解像度のまま、720にすると720pにダウンスケールする。
SCALE_HEIGHT = None

_HEADERS = {"User-Agent": "Mozilla/5.0"}
_WINDOWS_PATH_RE = re.compile(r"^([A-Za-z]):[///](.*)$")


def is_url(s: str) -> bool:
    return urlparse(s).scheme in ("http", "https")


def to_local_path(s: str) -> Path:
    """Windows形式のパス（C:/... や C://...）をWSLの/mnt/c/...に変換する。"""
    m = _WINDOWS_PATH_RE.match(s)
    if m:
        drive, rest = m.group(1).lower(), m.group(2).replace("//", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(s)


def download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=30, headers=_HEADERS) as res:
        res.raise_for_status()
        with dest.open("wb") as f:
            for chunk in res.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)


def compress(src: Path, dest: Path, crf: int, scale_height: int | None) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "warning",
        "-i", str(src),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "128k",
    ]
    if scale_height is not None:
        cmd += ["-vf", f"scale=-2:{scale_height}"]
    cmd.append(str(dest))
    subprocess.run(cmd, check=True)


def main() -> None:
    if is_url(VIDEO_URL):
        name = Path(urlparse(VIDEO_URL).path).name or "video.mp4"
        src_path = Path(__file__).parent / f"_downloaded_{name}"
        print(f"downloading: {VIDEO_URL}")
        download(VIDEO_URL, src_path)
        cleanup = True
        output_dir = Path(__file__).parent
    else:
        src_path = to_local_path(VIDEO_URL)
        if not src_path.is_file():
            sys.exit(f"file not found: {src_path}")
        name = src_path.name
        cleanup = False
        output_dir = src_path.parent

    # 元のファイルと同じディレクトリに保存する。
    output_path = output_dir / f"{Path(name).stem}_compressed.mp4"

    print(f"compressing (crf={CRF}, scale_height={SCALE_HEIGHT}) -> {output_path}")
    try:
        compress(src_path, output_path, CRF, SCALE_HEIGHT)
    finally:
        if cleanup:
            src_path.unlink(missing_ok=True)

    before = output_path.stat().st_size
    print(f"done: {output_path} ({before / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
