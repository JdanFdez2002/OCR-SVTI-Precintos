# Instalación y ejecución

Usar Python 3.10 y el entorno virtual del proyecto:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe ocr_tools/setup_tesseract_models.py
```

Instalar Tesseract por separado. Debe estar en PATH o en su ubicación habitual
para Windows. También se puede definir SEAI_TESSERACT_CMD con la ruta a
tesseract.exe. El idioma predeterminado es eng. Los modelos y sus opciones
están documentados en [models/README.md](models/README.md).

Aplicación de cámara, PTZ, captura manual y manifiesto:

```powershell
.\venv\Scripts\python.exe main.py
```

Evaluador de carpetas, imágenes, resultados y exportación Excel/JSON:

```powershell
.\venv\Scripts\python.exe motor.py
```

Ambas aplicaciones usan el mismo lector y OcrWorker. No se realiza OCR continuo
sobre video. Las credenciales se configuran desde la interfaz o con CAM_IP,
CAM_USER, CAM_PASS y CAM_CHANNEL. Las capturas se guardan automáticamente en
capturas_precintos, que se crea al necesitarla.

Opciones OCR por variables de entorno:

- SEAI_OCR_MAX_SECONDS: presupuesto por imagen, 9 segundos por defecto.
- SEAI_OCR_MAX_CALLS: máximo de llamadas, 16 por defecto.
- SEAI_OCR_CALL_TIMEOUT: límite por llamada, 4 segundos por defecto.
- SEAI_OCR_MIN_CONFIDENCE: umbral de advertencia, 0.45 por defecto.
- SEAI_TESSERACT_LANG: idioma Tesseract, eng por defecto.
- SEAI_TESSDATA_DIR: carpeta alternativa con archivos .traineddata.
- SEAI_TEXT_EMPHASIS: 0 para desactivar la atenuación de fondo.

Las lecturas de baja confianza se conservan como inciertas, sin completar ni
sustituir caracteres. El presupuesto no garantiza precisión ni que se hayan
completado todas las variantes de reconocimiento.
