# AGROINTEP – Inspección de EPP en **video en vivo** (Visión Computacional)

Institución Universitaria de Roldanillo – UNINTEP · Inteligencia Artificial · 8vo semestre
Equipo: Juan Pablo Florez Soto, Juan Manuel Franco Rojas, Milton Vanegas Delgado
Docente: Oscar Fernando Pulgarín Molina

## Qué hace
Antes de entrar al laboratorio o planta agroindustrial, el estudiante es grabado por una cámara. El
sistema verifica **cofia, guantes y bata** en tiempo real y decide:

- **INGRESO APROBADO**: cumple con todo.
- **INGRESO DENEGADO - FALTA: …**: le falta algún elemento (indica cuál).
- **INCIERTO - REVISIÓN MANUAL**: no hay confianza suficiente; el docente decide (`a` autorizar / `r` rechazar).

Aprobar exige más confianza (0.80) que denegar (0.70): el error grave es dejar entrar a quien no cumple.
Cada decisión queda en `registro_inspecciones.csv` (origen: sistema o docente).

```
Cámara/RTSP → ¿hay estudiante? → clasificar fotograma → promedio temporal → decisión
            → pantalla en vivo → validación del docente → registro CSV
```

## Evaluación por elemento (recomendada): `--backend multi`
Un solo modelo (MobileNetV2) con **una salida por elemento** (cofia, guantes, bata): estima la
probabilidad de que cada uno esté presente. La pantalla muestra `COFIA: OK / FALTA / ?`,
`GUANTES: …`, `BATA: …` y el banner indica exactamente qué falta, incluso si faltan varios a la vez.

- Aprobado solo si **los tres** están OK (P ≥ 0.80). Si alguno falta (P ≤ 0.30) → denegado y se nombra.
- Si ninguno falta pero alguno no es seguro → INCIERTO (revisión manual).

```bash
uv run python -m src.multietiqueta                  # entrena; reporta recall de FALTA por elemento y falsos aprobados
uv run python -m src.inspeccion_video --backend multi
```
Los **datos públicos** se usan por elemento (`dataset/publico/<elemento>/<si|no>/`): una imagen con
`hairnet` solo aporta la etiqueta de cofia; lo demás no cuenta en la pérdida.
```bash
uv run python -m src.convertir_yolo_elementos --dataset descargas/<dataset-yolo> --elementos cofia,guantes
```
### Datos públicos: descarga y uso
| Dataset | Aporta | Comando de descarga |
|---|---|---|
| Higiene alimentaria (Zenodo 16329852) | cofia y guantes, con ausencia (`no_hairnet`, `no_glove`) | `uv run python -m src.descargar_datasets zenodo` |
| SH17 (Kaggle, CC BY-NC-SA) | guantes y bata (`medical-suit`, solo presencia) | `uv run python -m src.descargar_datasets kaggle --id mugheesahmad/sh17-dataset-for-ppe-detection` |
| Food Industry PPE (Roboflow, CC BY 4.0) | guantes (`gloves` / `no-gloves`) | `uv run python -m src.descargar_datasets roboflow --api-key … --workspace proof-87nsb --proyecto food-industry-ppe --version N` |

Luego, por cada carpeta descargada en `descargas/`:
```bash
uv run python -m src.convertir_yolo_elementos --dataset descargas/<carpeta> --elementos cofia,guantes --max-por-clase 800
uv run python -m src.multietiqueta                  # entrena con propios + públicos
uv run python -m src.multietiqueta --sin-publicos   # comparación: si los públicos no mejoran el test, descártelos
```
Los públicos van **solo a entrenamiento**; el test es siempre de su cámara. Si un elemento solo tiene
imágenes "presente" (p. ej. bata en SH17), el conversor avisa: no lo use, porque sesgaría el modelo.
Edite `mapeo_clases.json` si el dataset nombra distinto las clases. Cite cada dataset y su licencia.

Las imágenes propias (`dataset/raw/<estado>/`) etiquetan los tres elementos a la vez. Para que el
modelo no confunda elementos, grabe también **combinaciones** de faltantes (`falta_cofia_guantes`,
…; ver `docs/guion_captura.md`). `--backend cnn` (modelo por estado) sigue disponible como
comparación.

## Clases del modelo por estado (`--backend cnn` / `clasico`)
Cada fotograma se clasifica según el **estado de cumplimiento** de la persona, para que el video
indique directamente qué falta. Convención de nombres (carpetas en `dataset/raw/`):

| Clase | Resultado en pantalla |
|---|---|
| `completo` | INGRESO APROBADO |
| `falta_<epp>` (`falta_cofia`, `falta_bata`, `falta_guantes`) | INGRESO DENEGADO - FALTA: COFIA / BATA / GUANTES |
| `sin_epp` | INGRESO DENEGADO - FALTAN TODOS LOS EPP |

EPP verificados: **cofia, bata y guantes** (5 clases: `completo`, `falta_cofia`, `falta_bata`,
`falta_guantes`, `sin_epp`). Ajusta a los protocolos reales. Si falta más de un elemento a la vez
(salvo todos), el modelo no tiene clase propia y puede errar: bajará la confianza o marcará INCIERTO.
Confianza promedio < 0.70 → **INCIERTO – REVISIÓN MANUAL**.

## Ruta por fases (el video en vivo funciona desde la Fase 1)

| Fase | Entregable | Comando |
|---|---|---|
| 0 | Dataset con la **misma cámara** del acceso | `uv run python -m src.capturar_dataset --persona p01 --clase completo` |
| 1 · Corte 1 (30%) | EDA + limpieza + descriptores OpenCV | `uv run python -m src.eda_limpieza` |
| 1 | Comparativa ≥3 modelos + PCA + K-Means (guarda el mejor) | `uv run python -m src.benchmark_fase1` |
| 1 | MLP en Keras | `uv run python -m src.mlp_keras` |
| **1** | **Prototipo en vivo (modelo clásico)** | `uv run python -m src.inspeccion_video --backend clasico` |
| 2 · Sem. 8 | Transfer Learning MobileNetV2 | `uv run python -m src.transfer_learning` |
| **2** | **En vivo con la CNN** | `uv run python -m src.inspeccion_video --backend cnn` |
| **2** | **Modelo por elemento + en vivo** | `uv run python -m src.multietiqueta` · `uv run python -m src.inspeccion_video --backend multi` |
| 2 · Sem. 10–12 | API + Docker + MLflow + ética | ver abajo |
| **2** | **En vivo contra la API desplegada** | `uv run python -m src.inspeccion_video --backend api --url http://localhost:8000/predict` |

`--fuente` acepta cámara USB (`0`), archivo de video o transmisión (`rtsp://…`).
MLflow: `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`

## Instalación (con uv)
```bash
uv sync                      # crea .venv con Python 3.11 e instala todo (OpenCV 4.x)
uv run python -m src.inspeccion_video --backend multi   # los comandos se ejecutan con `uv run`
```
Sin uv: `pip install -r requirements.txt` (OpenCV 4.x; OpenCV 5 quitó CascadeClassifier).

## Dataset (ver `docs/guion_captura.md`)
Se construye con **su propia cámara**: 6+ personas × 5 clases × 60 fotogramas (~2,5 h en total).
```bash
uv run python -m src.capturar_dataset --persona p01 --clase completo     # ESPACIO inicia, para sola a los 60
uv run python -m src.capturar_dataset --persona p01 --clase falta_cofia  # ... y así con cada clase y persona
uv run python -m src.eda_limpieza                                        # control de calidad y resumen
```
Opcional, para sumar variedad: `uv run python -m src.convertir_yolo --dataset <carpeta YOLO> --requeridos cofia`
convierte datasets públicos de cajas a clases (ver `mapeo_clases.json`).

**Evaluación honesta:** train/validación/test se separan **por persona** (los fotogramas seguidos de un video son
casi idénticos). Los datos públicos van solo a entrenamiento; el test usa únicamente personas de su cámara.

## Despliegue (Fase 2)
```bash
uv run uvicorn api.main:app --reload                      # http://localhost:8000/docs
docker build -t agrointep-epp . && docker run -p 8000:8000 agrointep-epp
curl -F "archivo=@foto.jpg" http://localhost:8000/predict
```
`/predict` devuelve clase, confianza, probabilidades y `requiere_revision`. `/health` y `/stats` para monitoreo.
La API no guarda imágenes, solo metadatos.

## Controles del modo en vivo
`a` autorizar · `r` rechazar · `q` salir. Se registra en `registro_inspecciones.csv`.
Solo se analiza cuando hay un estudiante frente a la cámara (detector de rostro de OpenCV, usado
**únicamente** para detectar presencia: no identifica a nadie ni guarda nada). `--sin-presencia` lo desactiva.

## Roles (3 integrantes)
| Rol | Responsabilidad | Archivos |
|---|---|---|
| Data Engineer & MLOps | Captura, conversión, limpieza, EDA, MLflow, GitHub | `capturar_dataset.py`, `convertir_yolo.py`, `eda_limpieza.py`, `features.py`, `common.py` |
| Modelador & ML Lead | Benchmark, MLP, Transfer Learning, métricas | `benchmark_fase1.py`, `mlp_keras.py`, `transfer_learning.py` |
| Software & API Deployer | Video en vivo, FastAPI, Docker | `inspeccion_video.py`, `api/`, `Dockerfile` |

## Métricas a reportar
F1 macro, matriz de confusión y **recall por clase** (error crítico: dar por `completo` a quien le falta un EPP).
En vivo: FPS y latencia por predicción. Fase 2: latencia de la API.

## Limitaciones
Una persona por encuadre; el modelo se entrena para la cámara y encuadre del acceso (cambiar la cámara exige
recapturar/reajustar); no verifica que el EPP esté bien puesto ni su estado; depende de la luz y de la
variedad del dataset. No reemplaza la supervisión del responsable.
