# RadeonVideoAI

Suite de escritorio para Windows que reescala vídeo y mejora su calidad con IA
(elimina ruido, reconstruye detalle y nitidez). Desarrollada y probada en
hardware AMD de gama media (**AMD Ryzen 5 2600 + Radeon RX 9060 XT, 16 GB**),
pero la detección de hardware es automática y no depende de una marca: usa
DirectML (DirectX 12), que acelera por GPU en **NVIDIA, AMD e Intel** por
igual sin ninguna configuración manual — ver [Detección automática de
hardware](#detección-automática-de-hardware).

Cubre tres flujos independientes — reescalado/restauración por IA, recreación
generativa y aumento de FPS por interpolación de fotogramas — con interfaz
gráfica, vista previa lado a lado y exportación con codificación acelerada
por hardware, usando modelos de IA reales y con pesos entrenados
públicamente disponibles.

<img width="1600" height="864" alt="image" src="https://github.com/user-attachments/assets/8cfa52ab-04ff-47b1-a7e9-02e956bf406f" />

## Cómo funciona la IA

El motor usa las arquitecturas y pesos oficiales de **Real-ESRGAN**
(xinntao/Real-ESRGAN, licencia BSD-3-Clause), que son redes generativas
realmente entrenadas para super-resolución y restauración — no una red con
pesos aleatorios:

| Modelo en la interfaz | Arquitectura | Licencia | Uso recomendado |
|---|---|---|---|
| Universal Video Restoration | SRVGGNetCompact (`realesr-general-x4v3`) | BSD-3-Clause | Rápido, uso general. El deslizador de desruido interpola de verdad entre los pesos "nítido" y "con desruido" (la misma técnica que usa el CLI oficial de Real-ESRGAN). |
| Fine Details & Texture SR | RRDBNet 23 bloques (`RealESRGAN_x4plus`) | BSD-3-Clause | Máxima calidad de detalle en 4x, más lento. |
| Native 2x High Fidelity | RRDBNet 23 bloques (`RealESRGAN_x2plus`) | BSD-3-Clause | Mejor fidelidad que reescalar 4x→2x cuando solo necesitas duplicar la resolución. |
| Realistic Photo/Video Restoration | RRDBNet 23 bloques (`BSRGAN`, cszn/KAIR) | Apache-2.0 | Entrenado con un modelo de degradación más realista; suele lucir mejor en vídeo con ruido/artefactos de compresión reales. |
| Clean Animation & CG | RRDBNet 6 bloques (`RealESRGAN_x4plus_anime_6B`) | BSD-3-Clause | Animación / CG. |
| Community Ultra Sharp | RRDBNet 23 bloques (`4x-UltraSharp`, Kim2091) | **CC BY-NC-SA — solo uso no comercial** | Muy popular en la comunidad por su nitidez agresiva. Usa un tercer formato de checkpoint ("old-arch" ESRGAN, `model.0`/`model.1.sub...`) que el cargador remapea automáticamente a la arquitectura RRDBNet estándar. |

Los pesos (5–67 MB cada uno) se descargan automáticamente la primera vez que
se usa cada modelo, desde las release oficiales de GitHub/Hugging Face, y se
cachean en `models/weights/`.

**Nota honesta sobre "todos los modelos de Topaz":** Topaz Video AI (Proteus,
Artemis, Iris, Rhea, Theia, Gaia, etc.) usa redes propietarias y cerradas que
la empresa nunca publica; no es legal ni técnicamente posible incluirlas aquí.
Lo que sí se puede seguir sumando son más modelos **abiertos** de calidad
reconocida — si conoces alguno con pesos públicos, es fácil de añadir (ver
abajo). Tampoco existen "modelos propios de Claude/Anthropic" para
super-resolución de vídeo: Claude es un asistente de lenguaje, no un
proveedor de modelos de visión, así que no hay nada de eso que agregar.

### Añadir más modelos manualmente (avanzado)

El cargador soporta automáticamente los tres formatos de checkpoint ESRGAN
que circulan en la comunidad (BasicSR/Real-ESRGAN moderno, el "old-arch" de
BSRGAN con nombres `RRDB_trunk`/`upconv1`, y el "old-arch" secuencial de
`model.0`/`model.1.sub...` usado por UltraSharp) — se detectan y remapean
solos. Para añadir un nuevo checkpoint RRDBNet: descarga el `.pth` a
`models/weights/`, añade una entrada en `MODEL_REGISTRY` en `core/models.py`
apuntando a ese archivo (mismos `arch_kwargs` que `x4plus`, ajustando
`num_block` si corresponde), y aparecerá disponible en el desplegable.

### Detección automática de hardware

La app detecta el hardware al arrancar (`core/amd_backend.py`) y no requiere
configuración manual — ni elegir "modo NVIDIA" o "modo AMD" en ningún lado:

- **GPU**: se identifica el fabricante (NVIDIA, AMD o Intel) vía WMI, y el
  cómputo de la red neuronal corre por **ONNX Runtime con el proveedor
  DirectML** (`onnxruntime-directml`). A diferencia de CUDA o ROCm, DirectML
  no es específico de un fabricante — es una capa de cómputo de DirectX 12
  nativa de Windows — así que **el mismo código acelera por GPU en NVIDIA,
  AMD e Intel Arc por igual**, con cualquier driver moderno con soporte
  DirectX 12, sin instalar CUDA ni ROCm.
- **VRAM**: se lee el tamaño real desde el registro del driver (WMI trunca a
  32 bits y da valores incorrectos en tarjetas de 4GB+); en NVIDIA se usa
  además `nvidia-smi` como confirmación. Si no se puede determinar con
  certeza, se asume un valor conservador en vez de adivinar una tarjeta
  específica — el sistema de tiling degrada el tamaño de parche
  automáticamente si hace falta, así que subestimar VRAM es más seguro que
  sobrestimarla.
- **CPU**: si no hay ninguna GPU con DirectX 12 disponible, la app usa
  automáticamente todos los núcleos físicos de la CPU detectada como
  respaldo (más lento, pero funcional, sea Intel o AMD).

PyTorch solo se usa en CPU, una vez por modelo, para cargar el checkpoint y
exportarlo a un grafo ONNX cacheado (`models/weights/onnx_cache/`). Todo el
procesamiento por fotograma ocurre dentro de la sesión de ONNX Runtime. Se
descartó `torch-directml` porque Microsoft lo tiene en mantenimiento y ya no
publica builds compatibles con versiones actuales de PyTorch/Python; ONNX
Runtime + DirectML es la ruta vigente y soportada en 2026 para inferencia
GPU en Windows.

**Límite honesto — codificación de salida:** la aceleración por hardware del
*encoder* de salida (`hevc_amf`/`h264_amf`, sección siguiente) sí es
específica de AMD (AMF). En NVIDIA/Intel esto cae automáticamente a
`libx265`/`libx264` por software — la IA sigue corriendo por GPU igual, pero
la etapa final de exportar el video no usa el encoder de hardware de esas
tarjetas (NVENC/QuickSync no están implementados todavía).

### Pipeline de vídeo

- Lectura/escritura de fotogramas en crudo por tuberías binarias de FFmpeg
  (sin archivos intermedios en disco).
- División en parches (tiling) con **fusión por coseno** (cosine feathering)
  para reescalar imágenes grandes sin costuras ni artefactos en los bordes,
  con degradación automática del tamaño de parche si la memoria se satura.
- Codificación de salida acelerada por hardware AMD (`hevc_amf` / `h264_amf`)
  cuando hay una GPU AMD con AMF disponible, con fallback automático a
  `libx265`/`libx264` por software en cualquier otro caso (incluyendo
  NVIDIA/Intel, donde la IA sigue corriendo por GPU vía DirectML igual).
- El audio original se preserva intacto (remux sin recodificar).

## Pestaña "Generar" (IA Generativa)

Además de "Reescalar con IA" (Real-ESRGAN, arriba), la app tiene una segunda
pestaña completamente independiente — con su propio selector de archivos,
motor (`core/generative_engine.py`) y botones de Iniciar/Pausar/Cancelar —
que usa **Stable Diffusion 1.5 img2img** (`stable-diffusion-v1-5/stable-
diffusion-v1-5`, licencia CreativeML Open RAIL-M) vía `optimum` + ONNX
Runtime DirectML para *reinterpretar* cada fotograma agregando detalle nuevo,
en vez de restaurar fielmente lo que ya hay. Es una técnica distinta a
propósito, por eso vive en su propia pestaña en lugar de ser una opción más
del reescalado:

- **No cambia la resolución** — el vídeo generado sale del mismo tamaño que
  el original; no tiene relación con el factor de escala de la otra pestaña.
- **Es mucho más lento**: cada parche de 512×512 tarda unos segundos (no
  tiempo real), a diferencia del reescalado por IA restaurativa.
- **Puede producir parpadeo (flicker) entre fotogramas.** Cada fotograma se
  regenera de forma independiente, sin condicionamiento temporal entre
  frames — es una limitación conocida de aplicar difusión de imagen fija a
  vídeo, no un error. Para vídeo con temporalidad importante (caras,
  texturas finas) esto es más notorio que en el reescalado restaurativo.
- Los parámetros expuestos son **fuerza/creatividad** (cuánto se aleja del
  original) y **pasos de generación** (calidad vs. velocidad).

**Recomendación para restauración de películas antiguas:** usa "Reescalar
con IA" como herramienta principal — es determinista, no introduce
parpadeo y tiene un modelo (BSRGAN) entrenado específicamente para ruido y
artefactos de compresión reales. Trata "Generar" como experimental.

## Pestaña "Interpolar (FPS)" — resumible

Una tercera pestaña, también completamente independiente (`core/
interpolation_engine.py` + `core/interpolation_models.py`), que **aumenta
los FPS** del video generando fotogramas intermedios reales por flujo óptico
— el mismo tipo de técnica que usan herramientas como Flowframes/SVP —, sin
tocar la resolución ni tener relación con las otras dos pestañas:

- **Modelo**: RIFE 4.9 (hzwer/Practical-RIFE, licencia MIT), pre-exportado a
  un único grafo ONNX que acepta un `timestep` arbitrario, lo que permite
  generar directamente x2 (1 fotograma intermedio en t=0.5) o x4 (3
  fotogramas en t=0.25/0.5/0.75) sin encadenar pasadas. Corre por **ONNX
  Runtime DirectML** — no Vulkan/ncnn como el rife-ncnn-vulkan que usa
  Flowframes — para reutilizar exactamente el mismo camino de GPU
  multi-fabricante ya validado en el resto de la app, sin sumar un stack de
  inferencia nativo aparte.
- **Resumible de verdad**: el progreso se guarda en un archivo `progress.json`
  junto a una carpeta temporal (`.radeonvideoai_interp_<hash>` al lado del
  destino). Si el proceso se corta por lo que sea (cierre, corte de luz,
  cancelar a propósito), al volver a iniciar el mismo trabajo (mismo origen +
  destino + multiplicador) continúa desde el último punto guardado en vez de
  arrancar de cero.
- **Prioriza espacio en disco**: en vez de ir dejando fotogramas sueltos sin
  comprimir, cada checkpoint se guarda como un segmento de video ya
  codificado; estos se van combinando periódicamente en un único archivo
  acumulado y se borran apenas se confirman, así el uso de disco temporal
  queda acotado al tamaño del video ya producido, no al de todos los
  fotogramas interpolados sueltos.
- El audio original se preserva sin tocar (la interpolación no cambia la
  duración real del video, solo agrega fotogramas dentro del mismo tiempo).

## Requisitos

- Windows 10/11 64-bit.
- Cualquier GPU con soporte DirectX 12 (NVIDIA, AMD o Intel Arc) — probado
  principalmente en AMD Ryzen 5 2600 + Radeon RX 9060 XT, pero la detección
  de hardware es automática (ver arriba). Sin GPU compatible, funciona igual
  por CPU multi-núcleo, más lento.
- Driver de GPU reciente con soporte DirectX 12/DirectML.
- Python 3.10+ si se ejecuta desde código fuente (no hace falta si usas el
  ejecutable portable compilado).
- Conexión a internet la primera vez que se usa cada modelo de IA (descarga
  de pesos).

## Instalación (modo desarrollo)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

`ffmpeg.exe` y `ffprobe.exe` ya se incluyen en la raíz del proyecto.

## Compilar versión portable (.exe)

```bat
python build_portable.py
```

Genera `dist/RadeonVideoAI/RadeonVideoAI.exe`, listo para copiar a cualquier
PC con Windows y GPU AMD. Usa `Lanzar_App.bat` para iniciar la app (detecta
automáticamente si hay un ejecutable compilado o si debe correr desde
Python).

## Uso

La app tiene tres pestañas totalmente independientes — "Reescalar con IA",
"Generar" e "Interpolar (FPS)" (ver arriba) — cada una con su propio
selector de vídeo y botones de Iniciar/Pausar/Cancelar. Arrastrar un archivo
a la ventana lo carga en la pestaña que esté activa en ese momento.

1. Arrastra un vídeo a la ventana (o usa "Examinar...").
2. Elige factor de reescalado (1x restauración / 2x / 4x) y modelo de IA —
   prueba más de uno con un clip corto para comparar calidad antes de
   procesar el vídeo completo.
3. Ajusta los deslizadores de desruido y nitidez.
4. Ajusta el tamaño de parche según tu VRAM (768–1024 px aprovecha mejor una
   GPU de 16 GB; baja a 384–256 px si ves errores de memoria).
5. Pulsa "INICIAR PROCESAMIENTO". El visor central funciona como el
   comparador de Topaz: alterna entre vista **dividida**, **solo original**
   o **solo mejorada**, haz zoom con la rueda del mouse y arrastra para
   recorrer la imagen a máximo detalle — todo se actualiza en vivo mientras
   procesa. La barra inferior muestra FPS, ETA y uso de memoria en tiempo
   real.

### Sobre la velocidad

La super-resolución por IA real es un proceso pesado en cualquier GPU que no
sea de gama muy alta — Topaz Video AI recomienda tarjetas potentes
precisamente por esto, y en hardware de gama media ellos tampoco procesan a
velocidad de tiempo real. Si notas que va lento: usa el modelo "Universal
(rápido)", sube el tamaño de parche a 768–1024 px, o baja la escala a 2x en
vez de 4x. Si el resultado no convence en calidad, compara los distintos
modelos con un clip corto — cada uno tiene fortalezas distintas según el
tipo de contenido (ver tabla arriba).

## Pruebas

```bat
python -m unittest discover -s tests
```

`test_03_ai_model_forward` y `test_e2e.py` requieren red la primera vez
(descargan los pesos oficiales); si no hay conexión, el primero se omite
automáticamente.

## Licencias de los modelos

Los pesos de Real-ESRGAN se distribuyen bajo licencia BSD-3-Clause por sus
autores (xinntao). Revisa el repositorio oficial
(https://github.com/xinntao/Real-ESRGAN) para más detalles antes de un uso
comercial.

Stable Diffusion 1.5 (pestaña "Generar") se distribuye bajo licencia
CreativeML Open RAIL-M — permisiva para uso personal y la mayoría de usos
comerciales, con restricciones de uso aceptable (ver
https://huggingface.co/spaces/CompVis/stable-diffusion-license).

RIFE (pestaña "Interpolar") se distribuye bajo licencia MIT por sus autores
(hzwer/Practical-RIFE); el export a ONNX usado aquí proviene de
https://huggingface.co/edgetools/rife.
