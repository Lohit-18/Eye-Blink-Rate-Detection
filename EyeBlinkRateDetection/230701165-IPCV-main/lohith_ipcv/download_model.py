"""
Download dlib's 68-landmark shape predictor if missing.
Official file: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""

from __future__ import annotations

import bz2
import shutil
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models"
BZ2_NAME = "shape_predictor_68_face_landmarks.dat.bz2"
DAT_NAME = "shape_predictor_68_face_landmarks.dat"
URL = f"http://dlib.net/files/{BZ2_NAME}"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dat_path = MODEL_DIR / DAT_NAME
    if dat_path.is_file():
        print(f"Model already present: {dat_path}")
        return

    bz2_path = MODEL_DIR / BZ2_NAME
    if not bz2_path.is_file():
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, bz2_path)

    print("Extracting .bz2 ...")
    with bz2.open(bz2_path, "rb") as src, open(dat_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    bz2_path.unlink(missing_ok=True)
    print(f"Ready: {dat_path}")


if __name__ == "__main__":
    main()
