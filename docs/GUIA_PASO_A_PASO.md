# Guía paso a paso: de cero al sistema funcionando

Control de ingreso AGROINTEP: cámara → verifica **cofia, guantes y bata** → aprueba o deniega (y dice qué falta).

**Tiempo total estimado:** 1 día de captura y entrenamiento + 1 día de pruebas y entrega.
**Quién hace qué:** Data Engineer (pasos 3–6), Modelador (pasos 7–10), Software/API (pasos 11–13).

> Cada paso termina con **"Salió bien si…"**. No pase al siguiente hasta que se cumpla.

---

## Paso 1. Preparar el computador con uv (15 min)

El proyecto ya trae `pyproject.toml`, `uv.lock` y `.python-version` (Python 3.11, que es el que soporta
TensorFlow). `uv` descarga Python y crea el entorno solo: no hace falta instalar Python aparte.

**1.1 Instalar uv** (PowerShell, no necesita administrador):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Cierre y abra de nuevo la terminal, y compruebe: `uv --version`.

**1.2 Instalar el proyecto** (dentro de la carpeta `agrointep_en_vivo`):
```powershell
uv sync
```
Crea `.venv` con Python 3.11 e instala todo (OpenCV 4.x, TensorFlow, scikit-learn, MLflow, FastAPI…).
La primera vez descarga unos 1–2 GB.

**1.3 Cómo se ejecutan los comandos:** en toda la guía ya van con `uv run` (usa el entorno del proyecto
sin activarlo). Ejemplos:
```powershell
uv run python -m src.capturar_dataset --persona p01 --clase completo
uv run python -m src.eda_limpieza
uv run python -m src.inspeccion_video --backend multi
uv run uvicorn api.main:app --reload
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
(Alternativa: active el entorno una vez con `.venv\Scripts\activate` y use los comandos sin `uv run`.)

**Si falla la instalación**
- "uv no se reconoce": cierre y reabra la terminal; si sigue, `pip install uv`.
- Error de permisos o de red institucional: copie el mensaje completo.
- Para agregar una librería: `uv add nombre`.

**Salió bien si:** `uv run python -c "import cv2, tensorflow, sklearn; print(cv2.__version__)"` imprime una
versión 4.x sin errores.

Estructura de carpetas (ya viene en el zip):
```
src/  api/  docs/  models/ (se llena sola)  dataset/ (se llena sola)  Dockerfile  mapeo_clases.json  pyproject.toml
```

## Paso 2. Probar la cámara y fijar el encuadre (15 min)

1. Ponga la cámara **donde irá en producción** (altura, ángulo, fondo). No la mueva después.
2. Marque en el piso un punto a ~1 m y otro a ~2 m. Un solo estudiante en el encuadre.
3. Pruebe: `uv run python -m src.capturar_dataset --persona prueba --clase prueba` (ESPACIO graba, `q` sale; borre `dataset/raw/prueba` después).

**Salió bien si:** ve la imagen en vivo y se ven la cabeza, el torso y **las manos** cuando la persona las pone al frente.

## Paso 3. Consentimiento y roles (20 min)

- Cada participante firma el consentimiento de `docs/guion_captura.md` (Ley 1581 de 2012).
- Use códigos `p01`, `p02`… La tabla código ↔ nombre se guarda **fuera** del repositorio.
- Roles en cada sesión: camarógrafo (ejecuta), modelo (la persona), coordinador (anota y revisa).

## Paso 4. Capturar el dataset propio (≈ 10 min por persona)

Es el paso que más pesa. Mínimo **6 personas** (ideal 10), distintas en talla, ropa y cabello.

Por cada persona, 8 tandas de 30 s (ESPACIO inicia; para sola a los 60 fotogramas):
```bash
uv run python -m src.capturar_dataset --persona p01 --clase completo
uv run python -m src.capturar_dataset --persona p01 --clase falta_cofia
uv run python -m src.capturar_dataset --persona p01 --clase falta_guantes
uv run python -m src.capturar_dataset --persona p01 --clase falta_bata
uv run python -m src.capturar_dataset --persona p01 --clase sin_epp
uv run python -m src.capturar_dataset --persona p01 --clase falta_cofia_guantes
uv run python -m src.capturar_dataset --persona p01 --clase falta_cofia_bata
uv run python -m src.capturar_dataset --persona p01 --clase falta_guantes_bata
```
Cómo se nombran: la carpeta lista **lo que falta**. `falta_cofia_guantes` = lleva solo la bata.

En cada tanda la persona hace la secuencia del guion (de frente, gira ±30°, se acerca y se aleja) y
**muestra las manos** al frente la mitad del tiempo. Capture con 2 condiciones de luz y guantes de
distinto color. Repita para p02, p03… (cada persona siempre con su mismo código).

> **Si hay poco tiempo:** empiece con `completo`, las tres `falta_<e>` y `sin_epp`, y agregue las
> combinaciones con las personas que alcancen. Sin combinaciones el modelo por elemento se confunde.

**Salió bien si:** `dataset/raw/` tiene 8 carpetas con ~60 × (n.º de personas) imágenes cada una.

## Paso 5. Control de calidad del dataset (10 min)

```bash
uv run python -m src.eda_limpieza
```
Revise el resumen y `reports/eda.png`:
- ≥ 6 personas propias y ≥ 3 por clase; desbalance < 3:1.
- El brillo no debe depender de la clase (si `completo` siempre es más claro, el modelo aprenderá la luz).
- Borre a mano fotogramas inútiles (cámara tapada, persona fuera de cuadro) y vuelva a correrlo.

El plan del curso pide máximo 3–4 clases en el **primer corte**. Para respetarlo, copie solo 4 carpetas
(`completo`, `falta_cofia`, `falta_guantes`, `sin_epp`) a `dataset/raw_corte1/` y corra
`uv run python -m src.eda_limpieza --raw dataset/raw_corte1`. Para el modelo final vuelva a correrlo sin `--raw`.

**Salió bien si:** crea `data/features.csv` y `dataset/limpio/` sin avisos graves.

## Paso 6. Datos públicos (opcional, recomendado para cofia y guantes)

Solo sirven para **entrenar**; el test siempre es de su cámara. Detalle y licencias en el README.
```bash
uv run python -m src.descargar_datasets zenodo
type descargas\zenodo16329852\data.yaml          # Windows (en Linux/Mac use cat): vea los nombres de clase
uv run python -m src.convertir_yolo_elementos --dataset descargas/zenodo16329852 --elementos cofia,guantes --max-por-clase 800
```
- Si el conversor dice "AVISO: solo hay imágenes con el elemento presente", **no use** ese elemento.
- La **bata** no tiene dataset público con ausencia: sale solo de sus fotos.
- Si el conversor no encuentra clases, edite `mapeo_clases.json` con los nombres de su `data.yaml`.

**Salió bien si:** aparecen imágenes en `dataset/publico/<elemento>/<si|no>/`.

## Paso 7. Fase 1 (primer corte): comparativa de modelos y MLP (≈ 15 min)

```bash
uv run python -m src.benchmark_fase1      # Regresión Logística, Random Forest, SVM + PCA + K-Means
uv run python -m src.mlp_keras
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://localhost:5000
```
Entregue: tabla de `reports/benchmark_fase1.csv`, matrices de confusión, `reports/pca.png`, MLflow.
Mire el **recall por clase**: el error grave es dar por `completo` a quien le falta algo.

**Salió bien si:** se crea `models/clasico.joblib` y el F1 en test es razonable (< 1.0 en datos reales; si
da 1.0 exacto, sospeche fuga de datos: revise que cada persona tenga un código único).

## Paso 8. Prototipo en vivo con el modelo clásico (corte 1)

```bash
uv run python -m src.inspeccion_video --backend clasico
```
Teclas: `a` autorizar · `r` rechazar · `q` salir. Las decisiones quedan en `registro_inspecciones.csv`.

**Salió bien si:** al ponerse frente a la cámara con y sin EPP el banner cambia a INGRESO APROBADO /
DENEGADO y cuando se va dice "ESPERANDO ESTUDIANTE". Anote FPS y latencia (van en pantalla).

## Paso 9. Fase 2: CNN y modelo por elemento (≈ 30–60 min, con internet)

```bash
uv run python -m src.transfer_learning        # MobileNetV2 por estado → models/epp_mobilenet.keras
uv run python -m src.multietiqueta --img 288  # un resultado por elemento → models/epp_multi.keras
uv run python -m src.multietiqueta --sin-publicos --img 288   # comparación sin datos públicos
```
La primera vez descarga los pesos de ImageNet (necesita internet). Sin GPU puede tardar; baje a
`--img 224` si es muy lento.

Mire en la salida de `multietiqueta`:
- **recall_falta** por elemento (meta ≥ 0.95) y **falsos_aprobados_tasa** (meta cercana a 0).
- Si con `--sin-publicos` el test no empeora, no use los públicos.
- Si un elemento (casi siempre guantes) tiene recall bajo: capture más tandas con las manos visibles,
  suba `--img` y revise las imágenes mal clasificadas.

**Salió bien si:** existen `models/epp_multi.keras` y `models/elementos.json`, y `falsos_aprobados_tasa` es baja.

## Paso 10. Calibrar los umbrales

Aprobar exige más confianza (0.80) que denegar (0.70). Con el resultado del paso 9:
- Si hay falsos aprobados: suba `--umbral-aprobar` a 0.90–0.95 (más casos irán al docente).
- Si hay demasiados INCIERTO con personas que sí cumplen: baje un poco, sin permitir falsos aprobados.

```bash
uv run python -m src.inspeccion_video --backend multi --umbral-aprobar 0.90
```

## Paso 11. Prueba en vivo con voluntarios nuevos (≈ 1 h)

Use **personas que no participaron en la captura**. Pruebe estas 8 situaciones con cada una y anote
si el resultado fue correcto:

| # | Situación | Resultado esperado |
|---|---|---|
| 1 | Todo puesto | APROBADO |
| 2–4 | Falta solo cofia / guantes / bata | DENEGADO, nombra el elemento |
| 5 | Nada puesto | DENEGADO |
| 6–8 | Faltan dos elementos | DENEGADO, nombra los dos |

Repita con luz distinta, otra distancia y manos en bolsillos. Cuente **cuántas veces aprobó a quien no
cumplía** (error grave) y cuántas mandó a revisión. Esa tabla es su evidencia para la sustentación.

## Paso 12. API REST y Docker (Semana 10–12)

Local:
```bash
uv run uvicorn api.main:app --reload             # http://localhost:8000/docs
curl -F "archivo=@foto.jpg" http://localhost:8000/predict
```
Respuesta: `ingreso` (aprobado | denegado | incierto), `resultado`, `faltantes`, `elementos`,
`probabilidades`, `requiere_revision`, `latencia_ms`. `/health` y `/stats` sirven de monitoreo.

Docker (antes confirme que `models/` tiene `epp_multi.keras` y `elementos.json`):
```bash
docker build -t agrointep-epp .
docker run -p 8000:8000 agrointep-epp
uv run python -m src.inspeccion_video --backend api --url http://localhost:8000/predict
```
Para usar el modelo clásico en el contenedor: `docker run -e EPP_BACKEND=clasico -p 8000:8000 agrointep-epp`.

**Salió bien si:** `http://localhost:8000/health` responde `ok` y el video en vivo funciona contra la API.

## Paso 13. Entrega final (Semana 12, 35 %)

Lista de verificación:
- [ ] Solución end-to-end: cámara → API en contenedor Docker → decisión.
- [ ] MLflow con los experimentos de Fase 1 y Fase 2 (capturas de la interfaz).
- [ ] Tabla de métricas: recall de "falta" por elemento, falsos aprobados, latencia, FPS.
- [ ] Prueba con voluntarios nuevos (paso 11).
- [ ] Análisis de impacto ético: consentimiento, datos codificados y no publicados, sesgos (tallas, tonos
      de piel, cabello), qué pasa si el sistema falla (decisión final del docente), derecho a eliminación.
- [ ] Limitaciones: una persona por encuadre, depende de la cámara y la luz, no verifica que el EPP esté
      bien puesto ni en buen estado, no controla físicamente la puerta.
- [ ] Datasets públicos citados con su licencia.
- [ ] El informe describe lo que realmente se implementó: **clasificación**, no un detector YOLO/SSD
      (el informe actual habla de detección con mAP; alinéelo o justifique el cambio).

---

## Solución de problemas

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| "No se pudo cargar el detector de rostro" | OpenCV 5 instalado | `uv add "opencv-python>=4.9,<5"` o use `--sin-presencia` |
| "No se pudo abrir la fuente de video" | Cámara ocupada o índice incorrecto | Cierre otras apps; pruebe `--fuente 1` |
| `FileNotFoundError: clasico.joblib` | No se corrió el benchmark | Paso 7 |
| `FileNotFoundError: epp_multi.keras` | No se entrenó el modelo por elemento | Paso 9 |
| F1 = 1.0 en datos reales | Fuga: personas repetidas o código mal escrito | Un código por persona, siempre el mismo |
| "Hay N personas propias; se necesitan al menos 3" | Pocas personas | Capture más personas |
| Aprueba a quien no cumple | Umbral bajo o faltan datos de ese caso | Suba `--umbral-aprobar`; capture más de ese caso |
| Muchos INCIERTO | Manos poco visibles, mala luz | Mejore el encuadre; capture más variedad |
| TensorFlow muy lento | Sin GPU | `--img 224`; menos épocas |
| La API no responde en el video | URL o contenedor | Revise `/health`; el programa sigue y muestra ERROR |
