# Recuperación del OCR universal de SeaAIScan

## Estado entregado

**Ambas aplicaciones usan por defecto el recorrido original con evidencia
estructurada, sobre la fotografía completa y sin inferencia automática en vivo.**
La comparación nativa reprodujo exactamente la salida original en **64/64 imágenes
únicas**. Esto acredita recuperación de comportamiento, no 64 identificadores
correctos: el banco sigue sin referencias verificadas.

El lector nuevo con presupuesto quedó disponible como experimento optativo. Aunque
reduce latencia y pasa los controles gráficos de códigos desconocidos, perdió dos
salidas textuales del original. No sustituye al recorrido estable por defecto.

## Causas comprobadas

1. El worker había sustituido `mejor_ocr` por `ocr_analysis.analyze` en ambas
   aplicaciones. El análisis ejecutaba **21 inferencias** y había eliminado la
   salida anticipada. Primero aplicaba todos los filtros y sólo después rotaba.
2. El texto presentado provenía de `candidates[0]`, una caja individual ordenada
   por confianza. Una lectura corta como `A 0020` podía superar en puntuación a
   `A 0020427`, aunque la observación completa existiera. Las marcas y otras líneas
   se perdían en la presentación. Los vetos de fragmentos cortos y la política
   semántica de confirmación se habían mezclado con el reconocimiento visual.
3. El detector se ejecutaba periódicamente durante el video y podía activar OCR
   adicional. Esas tareas competían por CPU con RTSP, conversiones y Qt.
4. La retícula ampliada aún cortaba parte de algunos códigos. Se enviaba ese
   recorte al OCR principal, mientras la ventana de evaluación recibía la foto
   completa: las entradas eran diferentes.
5. Se copiaba y convertía el último fotograma incluso cuando su secuencia no había
   cambiado. Guardar el JPG y PNG de captura se hacía en el hilo de la interfaz.

Los modelos Paddle, sus diccionarios y sus pesos no se cambiaron. No se introdujo
un catálogo de códigos ni una corrección de caracteres por marca o valor esperado.

## Cambios e interfaces

- [ocr_reader.py](../modulos/ocr_reader.py): entrada común exclusivamente visual.
  `recognize_original` conserva el recorrido, texto, confianza, score y salida
  anticipada originales; anexa todas las cajas y observaciones recibidas.
  `recognize_fast` conserva observaciones completas y candidatos alternativos,
  con decisiones geométricas y presupuestos explícitos.
- [ocr_engine.py](../modulos/ocr_engine.py): añade el callback opcional `read` para
  observar la ejecución original sin duplicarla ni modificar sus filtros/píxeles.
  Las llamadas históricas siguen funcionando sin ese argumento.
- [ocr_worker.py](../modulos/ocr_worker.py): mismo lector en ambas ventanas;
  conserva sólo un trabajo manual y una muestra de video pendientes. Descarta
  video experimental cuando está desactivado, además de la protección en la UI.
  Guarda PNG y calcula el hash fuera del hilo Qt.
- [main_window.py](../ui/main_window.py): fotografía completa como entrada; no
  recorta los extremos por la retícula. Reutiliza el stream existente cuando ya
  está en el canal de alta resolución, evitando otra conexión RTSP. Reconocimiento
  y comparación tienen mensajes separados. Muestra el texto completo observado.
- [camera_service.py](../modulos/camera_service.py): permite consultar sólo muestras
  nuevas. Un duplicado reciente devuelve metadatos sin copiar píxeles; un frame
  obsoleto sigue invalidándose. No cambian las funciones HTTP, ISAPI ni de enfoque.
- [motor.py](../motor.py): conserva el mismo texto que la ventana principal;
  presenta el estado real del resultado, conserva las observaciones en la fila y
  exporta un JSON de evidencia junto al Excel elegido por el operador.
- [identifier_policy.py](../modulos/identifier_policy.py): conserva la política
  de llegada; el registro de rechazo puede incluir el texto visual completo.
  No se invoca desde el predictor y no cambia la lectura.
- [config.py](../config.py): modos y presupuestos explícitos; funciones de
  localización experimental conservadas y desactivadas por defecto.
- Pruebas: [test_ocr_recovery.py](../tests/test_ocr_recovery.py),
  [test_ui_safety.py](../tests/test_ui_safety.py),
  [test_camera_safety.py](../tests/test_camera_safety.py) y la prueba de cola
  experimental actualizada en [test_regions_and_fragments.py](../tests/test_regions_and_fragments.py).
- Herramientas: [evaluación](../ocr_tools/evaluate_recovery.py),
  [controles nativos](../ocr_tools/recovery_controls.py),
  [benchmark Qt/RTSP](../ocr_tools/benchmark_video_recovery.py) y
  [sondeo de sólo lectura](../ocr_tools/check_rtsp_readonly.py).

[ptz_worker.py](../modulos/ptz_worker.py) permanece idéntico por SHA-256 a la
validación de seguridad anterior. Se mantienen barreras de parada, validación
HTTP/ISAPI, invalidación por generación, cierre de hilos, autofocus nativo
condicionado a capacidades y prioridad J/L. Las pruebas no enviaron órdenes de
movimiento, zoom ni enfoque. El archivo de exclusiones Git tampoco cambió.

## Reconocimiento y comparación separados

El lector recibe imagen BGR, configuración, identificador técnico de evidencia y
cancelación. **No recibe manifiesto, lista de códigos, historial ni referencia.**
La ventana de evaluación ni siquiera pasa el nombre del archivo al predictor.

El resultado distingue el texto de la observación seleccionada, líneas originales,
cajas, confianza, candidatos y estado. El texto extenso se conserva, aunque incluya
marcas, símbolos u otras palabras. No se sustituye por el candidato más corto.
Las observaciones alternativas permanecen en el JSON aunque no sean la seleccionada.

La comparación se ejecuta después. Mantiene igualdad completa y normalización
conservadora; ninguna subcadena autoriza una llegada. El lector visual no declara
un identificador operativo confirmado por tener texto. **Las fotos completas
quedan para revisión de llegada**: la política conservadora continúa exigiendo
identificador visual único y contexto de ROI autorizado. El texto se muestra y
exporta independientemente de esa decisión y de que exista un manifiesto.

## Medición sobre fotografías reales

Se verificaron los hashes de las **128 rutas / 64 imágenes únicas** del inventario
existente. No se modificaron etiquetas, particiones, imágenes ni hashes. Se
evaluaron todos los hashes únicos; las copias no cuentan como muestras adicionales.
Las referencias no se entregaron a ninguno de los motores.

A y C se ejecutaron de nuevo, alternando el orden con el candidato acotado. B
proviene de la traza real conservada para exactamente los mismos hashes. Sus
tiempos son **históricos**, con otro intervalo de carga del equipo; no son una
comparación contemporánea emparejada. No se volvió a gastar 21 inferencias por
imagen para reproducir datos que ya estaban conservados.

| Motor | Mediana | p95 | Inferencias medias | Salidas no vacías / 64 |
|---|---:|---:|---:|---:|
| A: original, ejecución nueva | 2,70 s | 11,29 s | 8,64 | 63 |
| B: experimental lento, histórico | 14,32 s | 33,70 s | 21 | 63 |
| C: original recuperado y trazable, predeterminado | **2,54 s** | **11,86 s** | **8,64** | **63** |
| Candidato acotado, optativo | 2,17 s | 4,94 s | 4,31 | 61 |

C reproduce **el mismo texto que A en 64/64 casos**, incluido el vacío. No se
presenta la diferencia pequeña A/C como aceleración del algoritmo original.
Respecto del B histórico, se recuperó una latencia mediana aproximadamente 5,6
veces menor y se eliminaron las 21 llamadas obligatorias.

El candidato acotado perdió dos salidas que el original mostraba como `MA`. Sin
referencias verificadas no sabemos si eran ruido o texto útil. Ese resultado no
permite sustituir al recorrido estable por defecto.

Ejemplos emparejados por fotografía/hash, sin correcciones de caracteres:

| Fotografía | Original y recuperado | Experimental lento | Candidato acotado |
|---|---|---|---|
| `A0020427 Cerca(zoom alto Cam1).jpeg` | `@ evonik A 0020427 *` | `A 0020` | `A 0020427` y `EVOnIK` en líneas separadas |
| `ML-CL0314224 Cerca (zoom_Maximo Cam1)1.jpeg` | `MAERSK ML=CL0314224` | `L-CL0314224` | `MAERSK` y `ML=CL0314224` en líneas separadas |

En el segundo ejemplo se conserva el `=` realmente reconocido: no se reemplaza
por `-` para hacer coincidir un código conocido. Recuperar información completa
del modelo no garantiza que todos sus caracteres sean correctos.

[Resumen](recovery_20261027/summary.json), [CSV](recovery_20261027/results.csv),
[traza con cajas y observaciones](recovery_20261027/trace.jsonl) y
[protocolo con hashes](recovery_20261027/protocol.json).
Se archivaron el [evaluador ejecutado](recovery_20261027/evaluator_executed.py) y
el [lector ejecutado](recovery_20261027/reader_executed.py). Después de congelar
la comparación se añadieron presentación/exportación y el campo de score legado;
no se cambiaron las inferencias, rotaciones ni presupuestos comparados. Los
recursos históricos de B se representan como no disponibles, no como CPU/RAM cero.

No hay referencias verificadas del banco real: exactitud por identificador, CER,
omisiones, adiciones, sustituciones y falsas aceptaciones **no se pueden calcular**
con honestidad. La igualdad con A es una medida de recuperación, no de exactitud.
El banco disponible se concentra en dos tipos; no certifica códigos industriales
nuevos, condiciones de captura nuevas ni generalización a otras sesiones.

La medición de recursos comparte un proceso y los mismos predictores: CPU total
de A 724,55 s, C 704,0 s y candidato 381,02 s durante sus 64 ejecuciones. El RSS
muestreado alcanzó aproximadamente 2,31 GB en el proceso combinado. No se puede
atribuir ese máximo por separado a cada motor ni concluir una fuga a partir de él.

## Rotaciones y lector acotado

El plan optativo prueba BGR original, +90° **horario**, -90° antihorario y sólo
después calcula filtros en la orientación más prometedora. En la evidencia BGR
real congelada, +90° produjo alguna línea con confianza ≥0,88 en 50/64 imágenes;
-90° en 46/64 y sin rotación en 26/64. Es evidencia de confianza, no exactitud de
identificador. Se conservan ambas orientaciones; no hay una regla por marca o
posición. [Detalle](recovery_20261027/orientation_evidence.json).

El candidato requiere consistencia espacial entre observaciones para salir antes.
Una lectura completa y confiable de la misma región no es desplazada sólo porque
una truncada tenga confianza mayor. No se premia la longitud global ni se agregan
caracteres. Los rivales se guardan y la falta de consenso queda explícita.

Por defecto ese modo tiene seis llamadas y cinco segundos. El tiempo se comprueba
entre llamadas: no se interrumpe inseguramente una inferencia nativa en curso, que
puede sobrepasar el límite de pared. No hay salida condicionada a un manifiesto.
El modo original conserva fielmente su salida anticipada y su máximo histórico
de 21 intentos para fotos difíciles; no ejecuta 21 obligatoriamente.

## Controles con códigos desconocidos

El modelo real leyó seis cadenas ajenas a los ejemplos, con letras, números,
guiones, espacios y distintas longitudes, renderizadas a 0°, -90° y +90°.
Original y candidato obtuvieron **18/18 textos exactos** y dos negativos vacíos.
Las referencias se usaron para dibujar y evaluar después, nunca como entrada al
lector. [Resultados](recovery_controls_20261027/summary.json) y
[observaciones completas](recovery_controls_20261027/trace.jsonl).

Estos son controles técnicos de software/rotación con glifos OpenCV: **no son
fotografías de precintos industriales desconocidos ni demuestran generalización
real**. No se entrenó ni calibró el modelo con ellos.

## Video, CPU y memoria

La [prueba RTSP de sólo lectura](recovery_rtsp_20261027.json) recibió 167 muestras
recientes en ocho segundos, a 2688×1520, con edad máxima observada de 63 ms. No
envió HTTP, PTZ ni enfoque, y no guardó fotografías de la cámara.

Se midió además la interfaz **Qt offscreen con RTSP real**, reactivando la carga
anterior y después usando los valores recuperados. Cada fase duró unos diez
segundos. El worker PTZ no se inició; ninguna prueba movió la cámara.

| Medida con RTSP real | Carga anterior reactivada | Valores recuperados |
|---|---:|---:|
| p95 entre actualizaciones Qt | 66,2 ms | **55,3 ms** |
| Intervalos mayores de 66 ms | 9 | **3** |
| Actualizaciones Qt | 165 | **208** |
| CPU del proceso, respecto de un núcleo | 122,5 % | **49,3 %** |
| Trabajos de video enviados | 33 | **0** |
| Llamadas al detector | 5 | **0** |
| Llamadas OCR de video | 0 | **0** |
| RSS al medir | 742,24 MB | 742,83 MB |

En esta escena el OCR automático no llegó a activarse; la diferencia medida
corresponde al detector y a conversiones/copias. No se atribuye al OCR un coste
que no ocurrió. [Datos RTSP/Qt](recovery_video_rtsp_20261027.json).

Un [ensayo adicional con foto repetida y llegadas simuladas a 15 FPS](recovery_video_v2_20261027.json)
sí ejecutó cuatro llamadas OCR en la carga anterior: p95 de intervalos 316,0 →
48,4 ms y CPU 157,8 → 12,7 % de un núcleo. Es una carga técnica, separada del
ensayo RTSP real.

La primera instrumentación del benchmark guardaba arrays en `MagicMock` y
distorsionaba RAM/carga; quedó marcada como sustituida. Se corrigió usando
funciones contadoras sin historial. El [renderizado aislado durante 80 frames](recovery_paint_before_20261027.json)
se mantuvo aproximadamente en 90 MB. No se introdujo una supuesta corrección de
fuga en la aplicación sin reproducción. En el ensayo corregido, RSS quedó estable.

Las fases fueron secuenciales, cortas y con Qt offscreen. No certifican el monitor
Windows, red bajo carga prolongada, varias cámaras ni todos los escenarios de
captura manual mientras se muestra video. El guardado de evidencia y una captura
adicional de alta resolución no están incluidos en los tiempos OCR de la tabla.

## Configuración y reproducción

Valores predeterminados:

```text
SEAI_OCR_ENGINE=original
SEAI_LIVE_DETECTION=0
SEAI_LIVE_OCR=0
```

La localización y el OCR en vivo siguen implementados para investigación. Para
probarlos se requiere opt-in explícito; no son necesarios para capturar, enfocar
o usar zoom. El lector acotado se activa explícitamente con variables del proceso:

```powershell
$env:SEAI_OCR_ENGINE = 'fast'
$env:SEAI_OCR_MAX_CALLS = '6'
$env:SEAI_OCR_MAX_SECONDS = '5'
```

`original` conserva la alternativa estable; `experimental` permite investigar el
recorrido lento anterior. Ambas ventanas leen la misma configuración. Las claves
pueden editarse también en el módulo de configuración; no se cambiaron credenciales.

Desde la raíz, eligiendo salidas nuevas para conservar informes anteriores:

```powershell
& '.\venv\Scripts\python.exe' -m unittest discover -s tests -v
& '.\venv\Scripts\python.exe' -m ocr_tools.evaluate_recovery --output reports/nueva_recuperacion
& '.\venv\Scripts\python.exe' -m ocr_tools.recovery_controls --output reports/nuevos_controles
& '.\venv\Scripts\python.exe' -m ocr_tools.benchmark_video_recovery --output reports/nuevo_video.json --duration 10
& '.\venv\Scripts\python.exe' -m ocr_tools.benchmark_video_recovery --output reports/nuevo_rtsp_video.json --duration 10 --live-rtsp
```

El último comando lee la cámara configurada, sin iniciar PTZ. No usar múltiples
benchmarks nativos simultáneos. Las referencias del banco se deben verificar
independientemente antes de afirmar exactitud, CER o mejoras generales.

## Límites pendientes

Persisten errores visuales del modelo, fotos difíciles y tiempos largos en el
fallback original. El candidato rápido sigue optativo. Falta un banco revisado
con más tipos de precinto, capturas independientes y negativos etiquetados. Se
mantiene pendiente validar el funcionamiento óptico del autofocus y la prioridad
manual con hardware; aquí se conservaron y probaron sus contratos simulados.
No se prometió precisión perfecta ni compatibilidad con seis cámaras.

## Cierre de la intervención

Por solicitud del usuario se detuvieron las investigaciones y no se ejecutaron
nuevos benchmarks. La prueba de OCR en un proceso separado no produjo un resultado
validado: se cerraron sus procesos y se retiró la integración multiproceso.
El motor original sigue siendo la opción predeterminada.

Comprobación final: ambos módulos de entrada se importaron y ambas ventanas se
construyeron, mostraron y cerraron sin errores, con cámara e inferencias inhibidas
para esta comprobación. Se verificó la sintaxis de 30 archivos Python.
La verificación anterior de 90 pruebas queda conservada en su informe fechado.

Las mediciones ya realizadas muestran que una lectura manual difícil todavía
puede introducir tirones mientras el video está activo. No se aplicaron los
experimentos de hilos ni de reducción de imagen, que no demostraron una mejora.
El usuario verificará personalmente lecturas nuevas, fluidez durante la captura,
controles de cámara y autofocus antes de decidir próximas modificaciones.
