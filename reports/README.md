# Evidencia de la investigación OCR

**Estado actual:** [Recuperación del OCR universal](RECUPERACION_OCR_20261027.md).
El recorrido original compartido es el predeterminado y la inferencia en vivo
está desactivada. Los informes anteriores describen estados históricos.

La auditoría, las correcciones de seguridad, el enfoque nativo, la integración y
sus límites se describen en [SEAAISCAN_VALIDACION_20261027.md](SEAAISCAN_VALIDACION_20261027.md).
La comparación simulada de seguridad está en
[safety_final_20261027.json](safety_final_20261027.json).
Los manifiestos del banco disponible conservan 128 archivos/64 hashes únicos sin
inventar referencias. Ver el informe antes de interpretar cifras como exactitud.

Los JSON en esta carpeta son resultados locales reproducibles, no una certificación
de precisión. No se incluyen fotografías ni secretos. Los experimentos sintéticos
se identifican expresamente y no sustituyen las 104 fotografías del banco real.

- `ocr_baseline.json`: auditoría antes de modificar el motor.
- `ocr_after.json`: comparación pareada de comportamiento y preprocesamiento.
- Los resultados fotográficos sólo son válidos con manifiesto, hashes, referencias
  revisadas y la política de recorte registrada.

La línea base es el archivo `modulos/ocr_engine.py` del commit `e35236e`.
Se carga con `git show`; no se altera el checkout ni el índice. El hash SHA-256 de
la fuente y los modelos se registra en los resultados.
