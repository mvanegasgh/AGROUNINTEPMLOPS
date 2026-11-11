# Guion de captura del dataset (≈ 2 horas en total)

## Meta
| | Mínimo | Recomendado |
|---|---|---|
| Personas | 6 | 8–10 |
| Clases | 4 | 5 (`completo`, `falta_cofia`, `falta_bata`, `falta_guantes`, `sin_epp`) |
| Fotogramas por persona y clase | 60 (30 s) | 60–120 |
| **Total** | ~1.080 | ~2.000+ |

Lo que más importa es el **número de personas distintas**, no el de fotogramas: el sistema separa
train y test **por persona**, así que el test se evalúa con gente que el modelo nunca vio.

## Preparación (una sola vez)
1. Fije la cámara **donde irá en producción** (altura, ángulo, fondo). No la mueva entre tandas.
   Marque en el piso un punto a ~1 m y a ~2 m. Un solo estudiante en el encuadre.
2. Reúna los EPP reales (cofia, bata, guantes, botas) en varias tallas, si es posible.
3. Cada participante firma el consentimiento (modelo abajo). Use **códigos** (`p01`, `p02`…) en `--persona`;
   la tabla código ↔ nombre se guarda aparte, fuera del repositorio.
4. Roles: *camarógrafo* (ejecuta el comando), *modelo* (la persona) y *coordinador* (anota y controla la lista).

## Qué grabar por persona (5 tandas de 30 s)
| Clase | Qué viste la persona |
|---|---|
| `completo` | Todos los EPP del protocolo (con guantes puestos) |
| `falta_cofia` | Todo **menos** la cofia (pelo suelto/recogido) |
| `falta_bata` | Todo **menos** la bata |
| `falta_guantes` | Todo **menos** los guantes (manos descubiertas, bien visibles) |
| `sin_epp` | Ropa normal, sin ningún EPP |

**Para el modelo por elemento (`src/multietiqueta.py`)** graben además las combinaciones de faltantes;
sin ellas el modelo puede confundir elementos (en una prueba, sin combinaciones, no detectó los
guantes ausentes cuando también faltaba la cofia). El nombre de la carpeta lista lo que falta:

| Clase adicional | Qué viste la persona |
|---|---|
| `falta_cofia_guantes` | Solo la bata |
| `falta_cofia_bata` | Solo los guantes |
| `falta_guantes_bata` | Solo la cofia |

```bash
python -m src.capturar_dataset --persona p01 --clase falta_cofia_guantes
```
Total: 8 tandas por persona (≈ 10 min). Si hay poco tiempo, priorice `completo`, las tres `falta_<e>` y
`sin_epp`, y agregue combinaciones con las personas que alcancen.

Dentro de **cada tanda de 30 s**, la persona hace esta secuencia (así hay variedad sin esfuerzo):
- 0–10 s: de frente, a ~1 m, brazos abajo.
- 10–20 s: gira ±30° a cada lado; manos al frente, luego a los lados.
- 20–30 s: se acerca y se aleja (0,7 a 1,5 m).

**Guantes:** son lo más difícil de ver (manos pequeñas, a veces fuera de cuadro). En todas las tandas
las manos deben quedar visibles al menos la mitad del tiempo: al frente a la altura del pecho, no en
los bolsillos. En `falta_guantes` y `completo` capture también guantes de distinto color (látex, nitrilo).
Si en el video las manos casi no se distinguen, el modelo no podrá decidir y marcará INCIERTO.

```bash
python -m src.capturar_dataset --persona p01 --clase completo       # ESPACIO inicia; para sola a los 60
python -m src.capturar_dataset --persona p01 --clase falta_cofia
python -m src.capturar_dataset --persona p01 --clase falta_bata
python -m src.capturar_dataset --persona p01 --clase falta_guantes
python -m src.capturar_dataset --persona p01 --clase sin_epp
```
Repita con p02, p03… (≈ 6-7 min por persona incluyendo cambios de EPP).

## Variedad (haga al menos 2 de estas)
- Dos momentos del día (luz de mañana y de tarde) o con la luz del pasillo encendida/apagada.
- Ropa de abajo distinta (jean, sudadera, colores claros y oscuros).
- Personas de distinta estatura, complexión y tipo de cabello.
- Algunas tandas con las manos ocupadas o en los bolsillos.

## Control de calidad
```bash
python -m src.eda_limpieza
```
Revise el resumen: **≥ 6 personas propias**, **≥ 3 personas por clase**, desbalance < 3:1, y mire `reports/eda.png`
(brillo por clase: si `completo` siempre es más claro que `sin_epp`, el modelo aprenderá la luz y no el EPP).
Borre a mano fotogramas inútiles (cámara tapada, persona fuera de cuadro).

## Si hay poco tiempo
Empiece con 3 clases (`completo`, `falta_cofia`, `sin_epp`) y 5 personas; agregue clases/personas después.
Para el corte de la Semana 6 es suficiente para demostrar el pipeline y la comparativa.

## Datos públicos (complemento, opcional)
Sirven para sumar variedad, **no** para reemplazar su cámara. Descargue un dataset YOLO y conviértalo:
```bash
python -m src.convertir_yolo --dataset descargas/ppe-zaintures --requeridos cofia
```
Se copian a `dataset/raw/<clase>/` con grupo `pub…`: **van solo a entrenamiento**; el test siempre usa personas
propias. Compruebe con el test si ayudan: si las métricas no mejoran (o empeoran), descártelos.
Revise la licencia de cada dataset y cítelo en el informe. Edite `mapeo_clases.json` si sus clases se llaman distinto.

## Modelo de consentimiento (adaptar con el docente/institución)
> Yo, ______________, autorizo a los estudiantes de Ingeniería Informática de UNINTEP a registrar imágenes mías
> mediante video exclusivamente para entrenar y evaluar un sistema académico de verificación de EPP en AGROINTEP.
> Entiendo que las imágenes se usarán de forma codificada, no se publicarán, se conservarán solo durante el proyecto
> y puedo solicitar su eliminación en cualquier momento (Ley 1581 de 2012).
> Firma: ____________ · Fecha: ________
