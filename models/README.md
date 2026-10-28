# Modelos locales del lector Tesseract

Instalación explícita desde la raíz del proyecto:

```powershell
.\venv\Scripts\python.exe ocr_tools/setup_tesseract_models.py
```

El instalador descarga y verifica SHA-256 de dos archivos:

- `DB_IC15_resnet18.onnx`: detector genérico descrito en la
  [documentación oficial de OpenCV](https://docs.opencv.org/4.x/d4/d43/tutorial_dnn_text_spotting.html).
  Sólo localiza trazos de texto; no interpreta caracteres ni conoce precintos.
- `tessdata_best/eng.traineddata`: pesos ingleses del repositorio oficial
  [tesseract-ocr/tessdata_best](https://github.com/tesseract-ocr/tessdata_best).
  El único reconocimiento de caracteres lo realiza Tesseract.

Los binarios están excluidos de Git. No se descarga nada durante el OCR ni se
envían fotografías a servicios externos. El detector y Tesseract se ejecutan
localmente en el hilo de trabajo existente.

El preprocesamiento conserva un lienzo completo y todas las propuestas en sus
coordenadas originales. No hay recortes individuales, selección central, límite
de cuatro regiones ni reglas de identificadores esperados. La variante original
continúa disponible como alternativa si la lectura no se estabiliza.

Opciones:

- `SEAI_TEXT_EMPHASIS=0`: desactiva la atenuación del fondo.
- `SEAI_TEXT_DETECTOR_MODEL`: ruta alternativa al modelo DB de OpenCV.
- `SEAI_TESSDATA_DIR`: carpeta alternativa de idiomas Tesseract.
- `SEAI_TESSERACT_LANG`: idioma, `eng` por defecto. Para otros idiomas hay que
  instalar sus archivos `.traineddata` en la carpeta elegida.

Para inglés se prefieren automáticamente los pesos locales `tessdata_best`.
Una carpeta explícita de idiomas o `TESSDATA_PREFIX` tiene prioridad. Sin pesos
locales se utilizan los idiomas de la instalación de Tesseract.
