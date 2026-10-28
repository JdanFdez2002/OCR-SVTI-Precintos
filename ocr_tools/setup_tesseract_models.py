"""Descarga explícita y verificada de modelos; nunca se ejecuta al leer fotos."""
from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    (ROOT / "models" / "DB_IC15_resnet18.onnx",
     "https://drive.google.com/uc?export=download&id=1vY_KsDZZZb_svd5RT6pjyI8BS1nPbBSX",
     "62fce3a058ac519952062fbd2cf45bae661130ce753d2a0fc918f4776ba3a02a"),
    (ROOT / "models" / "tessdata_best" / "eng.traineddata",
     "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/eng.traineddata",
     "8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba"),
)


def main():
    for path, url, checksum in MODELS:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == checksum:
            print(f"Disponible: {path.name}")
            continue
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != checksum:
            raise RuntimeError(f"Checksum inválido: {path.name}; no se instaló el archivo")
        path.write_bytes(data)
        print(f"Instalado: {path.name}")


if __name__ == "__main__":
    main()
