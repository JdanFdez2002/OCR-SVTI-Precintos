Crear entorno virtual con Python 3.10:
- py -3.10 -m venv venv

Activar en entorno virtual:
- .\venv\Scripts\activate

Puedes instalar todo utilizando este comando:
- python -m pip install --upgrade pip setuptools wheel; pip install paddlepaddle==2.6.2 paddleocr==2.7.3 pillow numpy==1.26.4 opencv-python==4.6.0.66 python-dotenv openpyxl
- pip install Pyside6


o en su defecto uno por uno:
Actualizar herramientas de instalación:
- pip install paddlepaddle==2.6.2
- pip install paddleocr==2.7.3
- pip install pillow
- pip install numpy==1.26.4
- pip install opencv-python==4.6.0.66
- pip install dotenv
- pip install python-dotenv
- pip install openpyxl
- pip install Pyside6


Verificar con:
- pip list | findstr "numpy opencv paddle"

numpy                    1.26.4
opencv-python            4.6.0.66
paddleocr                2.7.3
paddlepaddle             2.6.2
