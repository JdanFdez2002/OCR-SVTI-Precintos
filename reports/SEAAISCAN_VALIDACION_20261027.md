# SeaAIScan: auditoría, correcciones y validación

Trabajo local del 27 de octubre de 2026. No se conectó ni movió hardware durante
las pruebas automatizadas. Los cambios existentes en el motor, las herramientas,
las pruebas y la ventana de evaluación se conservaron; no se modificó el índice
de Git ni se ejecutaron operaciones destructivas.

## A. Fallos reproducidos y corregidos

La línea base se carga mediante `git show e35236e:<archivo>`, sin modificar el
checkout. [La comparación reproducida](safety_final_20261027.json) registra
los hashes de las fuentes originales y utiliza exclusivamente servicios simulados.

| Caso observado | Antes | Después |
|---|---|---|
| HTTP 403 en movimiento | Devuelve éxito | Devuelve fallo |
| HTTP 200 con `ResponseStatus/statusCode=4` | Devuelve éxito | Devuelve fallo |
| Falla la parada de pan/tilt/zoom | No intenta parar enfoque | Intenta ambos recursos, conserva el fallo |
| Movimiento, parada, movimiento antes de despachar | Sobrescribe la parada | Despacha la parada primero |
| Reconectar con enfoque pendiente | Puede enviarlo al servicio nuevo | Descarta el movimiento anterior, para y retira el servicio viejo |
| Fotograma conservado tras perder conexión | Se devuelve como video | Se invalida y vence por edad, se deshabilita captura |
| Lectura `XB 0097312Y`, esperado `B 0097312` | Marca Llegado por subcadena | Rechaza |
| Lectura posterior parcial de un Llegado | Retrocede a Dudoso | Conserva Llegado; registra rechazo |

Las paradas son barreras independientes de la ranura que conserva el último
movimiento. Se reintentan hasta tres veces; si fallan se descarta movimiento
pendiente y se requiere una parada satisfactoria antes de volver a mover.
Una aplicación no puede garantizar que una cámara físicamente desconectada reciba
una parada. El fallo queda visible; no se comunica como éxito.

El lector RTSP abre FFmpeg con límites de apertura y lectura de 2 s, libera cada
captura en `finally` e invalida el fotograma al fallar. Una muestra de más de 1,5 s
no habilita operaciones. Esto mide llegada de fotogramas decodificados: no prueba
que una cámara que siga transmitiendo imágenes idénticas esté mostrando una escena
actual. El respeto de los timeouts debe comprobarse con la cámara y red reales.

Los servicios retirados se cierran fuera de la cola de movimientos. La ventana
permanece viva hasta que terminan OCR, captura y cierre del lector, evitando
destruir un `QThread` activo. No se fuerza la terminación de inferencias nativas.

## B. Enfoque nativo Hikvision

Se verificó el contrato en **Intelligent Security API (General Application)
Developer Guide**, páginas impresas 174, 508 y 596 (PDF 214, 548 y 636):

- `GET /ISAPI/Image/channels/<ID>/focusConfiguration/capabilities`.
- `GET` y `PUT /ISAPI/Image/channels/<ID>/focusConfiguration`.
- Nodo `focusStyle`: `AUTO`, `MANUAL`, `SEMIAUTOMATIC`.

Fuente consultada: [copia del manual del fabricante](https://raw.githubusercontent.com/loozhengyuan/hikvision-sdk/master/resources/isapi.pdf).
El [portal oficial Hikvision](https://tpp.hikvision.com/download/ISAPI_OTAP)
permite seleccionar documentación por modelo. La copia general confirma el
contrato; no acredita compatibilidad con el equipo concreto del usuario.

Antes de escribir un modo, el servicio exige que el dispositivo anuncie **AUTO y
MANUAL**, lee la configuración y conserva sus otros campos y namespace. No prueba
endpoints alternativos ni envía una supuesta orden de autofocus auxiliar. Si la
capacidad no existe, se mantiene el mando manual original y se informa el estado.

Al conectar se solicita AUTO. J/L invalida autofocus pendiente y solicita MANUAL
antes del movimiento de enfoque; al soltar se paran los motores y se conserva el
control manual. I/K permite recuperar AUTO una sola vez después de una parada de
zoom confirmada. Repeticiones del teclado y paradas duplicadas no generan ráfagas.
Cambiar de servicio, perder video o cerrar invalida automatismos anteriores.
No se agregaron botones ni selectores de modo de enfoque.

Una petición HTTP ya enviada no puede cancelarse con seguridad: el mando manual
espera esa petición acotada y después tiene prioridad. El control óptico real,
sus tiempos, el posible autofocus interno al hacer zoom y el mantenimiento del
enfoque al soltar deben validarse físicamente. Aquí AUTO significa el modo nativo
anunciado por la cámara, no una orden de enfoque único ni una medida OpenCV.

## C. Regiones dinámicas y rendimiento

Se reutiliza el detector de texto de PaddleOCR sin cargar una segunda red. Trabaja
con lado máximo de 960 px; la interfaz presenta rectángulos amarillos discontinuos
rotulados **Texto candidato**. La retícula verde manual permanece disponible.
Detectar texto no certifica que sea un precinto.

La prueba con el detector **real**, sobre texto sintético, reprodujo un defecto
de la instalación 2.7.3: `ocr(rec=False)` evalúa `if not dt_boxes` sobre un ndarray
y lanza `ValueError` cuando detecta texto. El adaptador llama directamente a
`engine.text_detector`, el mismo detector instalado, y convierte explícitamente
sus cajas. Se agregó regresión y se repitió la prueba nativa. Esta corrección del
adaptador de detección no modifica las inferencias OCR de la comparación de fotos.

[Vista sintética de los recuadros renderizada y revisada](dynamic_regions_synthetic.png).
La imagen no proviene de una cámara; se cargó una fuente del sistema explícitamente
para el renderizador Qt offscreen.

Configuración inicial en [config.py](../config.py): detección cada 2 s,
seguimiento con muestras cada 250 ms y OCR de video como máximo cada 4 s. El
intervalo de detección se configura con `SEAI_DETECTION_INTERVAL_S`, entre 0,25 y
30 s. Son valores iniciales de investigación; consultar las medidas finales antes
de usarlos como presupuesto de procesamiento.

Entre detecciones se usa flujo óptico Lucas–Kanade, comprobación ida/vuelta,
un mínimo de cuatro puntos y rechazo de movimientos incoherentes. La asociación
usa superposición geométrica. Se exigen dos detecciones y al menos tres pasos
estables antes de pedir OCR de una región. La falta de puntos, la salida de la
imagen, una detección vacía, el cambio de dimensiones o la caducidad invalidan
el seguimiento. Los paquetes también llevan generación, época del stream y hora
de captura para que la UI descarte resultados anteriores.

OCR y detección comparten **un solo hilo/predictor**. La captura solicitada por el
operador tiene prioridad sobre el trabajo de video pendiente. Sólo se conserva
una muestra de video pendiente; las reemplazadas y antiguas se cuentan y descartan.
Una llamada nativa que ya comenzó debe terminar antes de atender otra tarea.

El OCR dinámico se ejecuta como `full_photo`, con dos llamadas como máximo, y sólo
muestra una sugerencia. **Nunca confirma una llegada.** No se eligió un candidato
dinámico como sustituto operativo de la retícula sin validación semántica.

Las pruebas de seguimiento usan una textura artificial trasladada y pérdida de
objetivo explícita. No son un ensayo con video Hikvision. No se han validado seis
cámaras concurrentes ni oclusiones reales, desenfoque, cambios de zoom o vibración.

## D. OCR e identificadores completos

La aplicación y la ventana de evaluación usan [ocr_analysis.py](../modulos/ocr_analysis.py)
a través del worker existente. `mejor_ocr` se conserva como API histórica para
compatibilidad y comparación; no autoriza llegadas.

La instalación PaddleOCR 2.7.3 filtra en `TextSystem.__call__` las cajas cuya
confianza queda por debajo de `drop_score=0.5`. El análisis cambia temporalmente
ese filtro a cero dentro del bloqueo de inferencia y lo restaura incluso si falla
el modelo. Esto conserva evidencia; **no baja la confianza exigida para aceptar**.
No recupera caracteres que el detector nunca localizó.

Se conservan texto original, coordenadas originales incluso al rotar, confianza,
variante, observación y fragmentos. La sintaxis admite espacios y guiones. No hay
diccionarios de precintos/marcas, sustituciones O/0 o I/1 ni prefijos inferidos.

Las hipótesis de agrupación usan orientación (hasta 8°), altura compatible,
alineación y separación relativas a la altura de las cajas. Un fragmento corto
no se descarta. Palabras alfabéticas de más de dos caracteres no se concatenan
por proximidad porque pueden ser marcas. Una agrupación geométrica sola queda
como **posiblemente incompleto**: se requiere además que una observación lea la
línea completa. Los rivales se conservan, incluso cuando difieren en un carácter.
No se premia un texto por tener más caracteres o incluir otras etiquetas.

El OCR manual analiza hasta 21 variantes y no sale después de dos lecturas
coincidentes: una variante posterior puede revelar un extremo omitido. La captura
usa margen proporcional a la altura del ROI, guarda la foto completa y el recorte
PNG exacto utilizado para inferir. Texto a menos de tres píxeles del borde impide
confirmación. Una letra fuera del recorte ampliado sigue requiriendo otra captura.

Estados: `reconocido`, `posiblemente_incompleto`, `ambiguo`, `sin_texto`,
`texto_no_identificado`, `imagen_invalida`, `error_tecnico` y `cancelado`.
**Sin texto no demuestra ausencia de precinto.** Sin anotación o detector semántico
no se pueden certificar categorías «sin precinto» y «texto irrelevante»; el banco
permite registrarlas como referencias revisadas. Los identificadores sólo
alfabéticos pasan a revisión porque pueden ser marcas. Esto limita cobertura y
no implica que esos identificadores sean inválidos.

## E. Política operativa y trazabilidad

[identifier_policy.py](../modulos/identifier_policy.py) compara **identificadores
completos** por igualdad. Sólo convierte minúsculas ASCII a mayúsculas, recorta
espacio exterior y colapsa espacios ASCII repetidos. Conserva espacios interiores,
guiones, ceros y cada carácter alfanumérico. No elimina puntuación desconocida para
hacer coincidir un valor. No acepta coincidencias de subcadena ni similitud.
Por ejemplo, `B 0097312` nunca equivale a `0097312`, `XB 0097312Y`, `B OO97312` ni
`B0097312`; el último requiere revisar el formato del manifiesto.

Para llegar a **Llegado** se requiere ROI manual declarado, estado reconocido,
candidato único, confianza mínima 0,88, observaciones suficientes y una única
fila del manifiesto con coincidencia completa. Variantes del mismo modelo son
evidencia correlacionada, no verificaciones independientes; el umbral no está
calibrado para producción. Coincidencia con manifiesto tampoco prueba integridad
visual si el propio manifiesto contiene un identificador truncado.

Las capturas guardan foto completa, PNG de entrada, hash, recorte, JSON con todas
las observaciones y una decisión en `capturas_precintos/decisiones.jsonl`. Se
escribe la evidencia y la decisión **antes** de modificar el estado. Un fallo de
escritura impide confirmar. Una lectura posterior incierta no cambia Llegado a
Dudoso ni a Pendiente. Esos archivos siguen bajo la exclusión Git existente.

## F. Protocolo fotográfico y resultados

Se inventariaron **128 archivos y 64 hashes distintos** en la carpeta facilitada.
Existen 64 copias idénticas. No corresponde a 104 imágenes únicas verificables.
Se conservaron todos los archivos, incluidos los duplicados, sin excluir casos
difíciles ni inferir etiquetas desde nombres o carpetas.

El [inventario](dataset_inventory_20261027.json) conserva rutas y SHA-256. El
[manifiesto de evaluación](dataset_evaluation_20261027.json) mantiene referencias
sin verificar y agrupa conservadoramente el banco como una sesión relacionada.
Todo se asignó a evaluación: el desarrollo utilizó pruebas sintéticas. No hay
partición independiente de nuevas sesiones/códigos para probar generalización.
Las referencias deben revisarse antes de calcular precisión, CER o falsas
aceptaciones. No se deben repartir copias o capturas relacionadas entre particiones.

La comparación ejecuta el motor original y el análisis nuevo con los mismos
modelos instalados, en orden alternado y sin recorte adicional (`full_photo`).
El original puede salir temprano; el nuevo inspecciona hasta 21 variantes. La
diferencia de latencia incluye esa diferencia de trabajo y no se interpreta como
aceleración del OCR. La interfaz utiliza la retícula ampliada y ese recorrido no
es equivalente a una fotografía completa declarada como ROI manual.

El informe final registra todas las filas, candidatos, cajas, confianza, variante,
tiempos, llamadas, errores técnicos, entorno y hashes de modelos/fuentes. Incluye
resúmenes por archivo y por hash único para no inflar el tamaño muestral. Errores
de inserción, eliminación, sustitución y de extremos, coincidencia exacta y falsas
aceptaciones quedan **null** si la referencia no está verificada. Rechazar todo no
cuenta como precisión perfecta.

Las inferencias de copias exactas se reutilizan por SHA-256: se verifican todos los
archivos y se conserva una fila por ruta, pero la latencia cuenta una medición real
por motor/hash. `measurement_origin`, `source_measurement_path` y
`reused_measurement` distinguen una inferencia nueva de su reutilización. La
[traza parcial inicial](photographic_20261027/trace.jsonl) se conservó al continuar
la evaluación en [photographic_complete_20261027](photographic_complete_20261027/).
No se modificó el comportamiento OCR durante la comparación. CPU/RSS de la
continuación no incluyen el intervalo previo; el benchmark separado registra sus
propios recursos. La deduplicación evita contar copias como medidas independientes.

### Medidas completadas

[Resumen OCR](photographic_complete_20261027/summary.json),
[resultados CSV](photographic_complete_20261027/results.csv) y
[benchmark de regiones](regions_benchmark_20261027.json).
La traza completa contiene **256 filas: 128 rutas × 2 motores**, correspondientes
a **128 mediciones reales: 64 hashes × 2 motores**. Se reutilizaron 69 mediciones
conservadas de la ejecución inicial y se completaron 59 nuevas. No hubo errores
técnicos ni entradas inválidas en la comparación. OpenCV emitió avisos de dos bytes
JPEG extra antes de un marcador; los archivos se decodificaron y no se excluyeron.

| Recorrido | Muestras reales | Mediana | p95 | Llamadas OCR medias |
|---|---:|---:|---:|---:|
| OCR original, foto completa | 64 | 3,75 s | 11,84 s | 8,64 |
| Análisis nuevo, foto completa | 64 | 14,32 s | 33,70 s | 21 |
| Detector real de texto, lado máximo 960 | 128 | 234 ms | 637 ms | No reconoce texto |
| Seguimiento de traslación **sintética** | 30 | 10,1 ms | 14,1 ms | Sin OCR |

El recorrido nuevo fue aproximadamente **3,8 veces más lento en mediana** al
inspeccionar todas las variantes. No se presenta como una optimización de tiempo.
En 62/64 fotos únicas devolvió `posiblemente_incompleto`, en una `sin_texto` y en
una `texto_no_identificado`; ninguna foto completa autoriza llegada. El original
produjo texto en 63/64, con aplanamiento de marcas y otras líneas. Esas salidas no
son referencias de identificadores correctos.

Hay **cero etiquetas verificadas**. Exactitud por identificador, CER, inserciones,
eliminaciones, sustituciones, errores iniciales/finales y tasas de falsas
aceptaciones siguen sin poder calcularse. Los ceros en contadores con denominador
cero no acreditan ausencia de errores. Esta ejecución demuestra trazabilidad y
costes; **no demuestra mejora de exactitud**.

El detector completó 128 mediciones sin excepciones y no produjo propuestas en
32 archivos; una propuesta ausente no demuestra ausencia de precinto. Su tiempo
máximo observado fue 1,166 s. El intervalo inicial de 2 s deja margen sobre el p95
de 0,637 s en esta máquina para una sola cámara; queda pendiente comprobarlo con
video real y carga concurrente. No se redujo el intervalo por extrapolación.

El benchmark del detector, ejecutado después de terminar las inferencias de la
comparación, registró 46,32 s de pared, 55,0 s de CPU, 118,8 % de CPU respecto de
un núcleo y 505,5 MiB de RSS máximo **muestreado**. El intervalo de continuación
de la comparación completa registró RSS inicial de 410.611.712 bytes y final de
2.294.833.152 bytes, compartido por ambos motores y sus trazas. Ese crecimiento
requiere perfilado antes de dimensionar seis cámaras; estas medidas no identifican
por sí solas una fuga ni atribuyen memoria a un motor concreto.

[La verificación final](verification_final_20261027.json) registra **73 pruebas
aprobadas**, los hashes de las fuentes y la comprobación de sintaxis. Los
diagnósticos del editor no reportaron errores y `git diff --check` pasó.

### Inspección ilustrativa de una fotografía real

Después de congelar el análisis se inspeccionó visualmente
`A0020427/A0020427 Cerca(zoom alto Cam1).jpeg`. La foto mide 2688×1520 y muestra
la letra A separada del bloque numérico en un texto orientado verticalmente.
Paddle detectó `A 0020427` en la entrada directa con confianza 0,9644 y caja
`[[1166,747],[1239,753],[1214,1066],[1141,1060]]`.

La retícula original recorta `(941,608)-(1747,912)`: excluye parte de esa línea.
El margen inicial amplía a `(835,502)-(1853,1018)`, pero tampoco garantiza incluir
todo el identificador en esa composición. Por tanto, el recorte central es una
causa **geométricamente comprobada de pérdida de contexto en esta foto**, sin que
ello pruebe que explique todas las omisiones de A observadas en producción.

En la evaluación de la foto completa, el análisis conserva `A 0020427` entre los
candidatos, pero algunas variantes producen `A 0020` con confianza mayor. La
sugerencia puede seguir siendo incompleta; se conserva el estado
`posiblemente_incompleto` y no se autoriza llegada. Esto es una limitación real
de selección/cobertura, no una mejora de exactitud demostrada. No se ajustaron
umbrales con esta foto. El operador debe mantener la línea entera dentro del ROI;
la foto completa queda disponible para revisión. La localización dinámica
todavía no se habilita para confirmar capturas fuera de la retícula.

## Reproducción

Desde la raíz del repositorio, con el entorno Python 3.10 configurado:

```powershell
& '.\venv\Scripts\python.exe' -m unittest discover -s tests -v
& '.\venv\Scripts\python.exe' -m ocr_tools.audit_safety --output reports/nueva_auditoria_seguridad.json
& '.\venv\Scripts\python.exe' -m ocr_tools.evaluate_ocr inventory --images 'C:\Users\Jdan\Downloads\Limpios' --output reports/nuevo_inventario.json --expected 128
& '.\venv\Scripts\python.exe' -m ocr_tools.evaluate_ocr evaluate --manifest reports/dataset_evaluation_20261027.json --output reports/nueva_comparacion --engines baseline experimental --split evaluation --repeats 1 --max-calls 21 --input-mode full_photo --deduplicate
& '.\venv\Scripts\python.exe' -m ocr_tools.benchmark_regions --manifest reports/dataset_evaluation_20261027.json --output reports/nuevo_benchmark_regiones.json
```

Las salidas se crean de forma exclusiva para conservar evidencia: elegir nombres
nuevos al repetir. No es necesario ejecutar la aplicación ni usar
credenciales para estas pruebas. Ejecutar benchmarks sin otras inferencias pesadas
concurrentes. Los tests del control usan servicios simulados; el benchmark usa el
modelo real en fotos y una secuencia de seguimiento artificial identificada.

No hubo herramientas Pylance MCP disponibles; la validación usa diagnósticos del
editor, análisis de sintaxis y pruebas ejecutadas con el intérprete del entorno.

## Validación física pendiente

Se necesita modelo y firmware exactos, respuestas de capacidades de la cámara,
confirmación óptica AUTO/MANUAL y zoom, latencia J/L, interrupción de red mientras
se mueve y cierre/reconexión RTSP. También se necesitan referencias revisadas y
negativos identificados, video real anotado y una prueba separada con seis streams.
Estos puntos no se sustituyeron por resultados simulados.
