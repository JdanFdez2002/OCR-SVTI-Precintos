Crear entorno virtual con Python 3.10:
- py -3.10 -m venv venv

Activar en entorno virtual:
- .\venv\Scripts\activate

Puedes instalar todo utilizando este comando:
- python -m pip install --upgrade pip setuptools wheel; pip install -r requirements.txt


o en su defecto uno por uno:
Actualizar herramientas de instalación:
- pip install -r requirements.txt

Verificar con:
- pip list | findstr "numpy opencv PySide6 openpyxl requests"

El reconocimiento OCR está desactivado temporalmente y no instala PaddleOCR.
