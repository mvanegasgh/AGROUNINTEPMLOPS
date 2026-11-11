"""Control de ingreso al laboratorio / planta AGROINTEP con VIDEO EN VIVO.

El estudiante se para frente a la camara. El sistema verifica cofia, guantes y bata:
  - cumple con todo            -> INGRESO APROBADO
  - le falta alguno            -> INGRESO DENEGADO - FALTA: <elemento(s)>
  - no esta seguro             -> INCIERTO: revision manual (el docente decide con a / r)

Con --backend multi (recomendado) cada elemento se evalua por separado y la pantalla muestra
el estado de COFIA, GUANTES y BATA (OK / FALTA / ?) con su probabilidad.

Uso:
  python -m src.inspeccion_video --backend multi
  python -m src.inspeccion_video --backend cnn --fuente 0          # modelo por estado (5 clases)
  python -m src.inspeccion_video --backend clasico
  python -m src.inspeccion_video --backend api --url http://localhost:8000/predict
  python -m src.inspeccion_video --backend multi --fuente rtsp://usuario:clave@ip/stream
  python -m src.inspeccion_video --backend multi --fuente prueba.mp4 --sin-ventana

Flujo: camara -> hay estudiante? -> clasificar fotograma -> promedio temporal -> decision estable
       -> pantalla + registro_inspecciones.csv (una fila por decision)
Teclas (correccion del docente): a = autorizar | r = rechazar | q = salir.

Privacidad: el detector de rostro de OpenCV solo detecta PRESENCIA; no identifica a nadie y
no se guarda ninguna imagen (el CSV solo contiene metadatos de la decision).
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2

from .predictores import UMBRAL_APROBAR, UMBRAL_DEFECTO, Suavizador, crear_predictor, decidir

COLOR = {"aprobado": (60, 170, 60), "denegado": (50, 50, 210), "incierto": (0, 150, 240),
         "espera": (110, 110, 110)}
COLOR_ELEM = {"ok": (90, 220, 90), "falta": (80, 80, 255), "incierto": (0, 180, 255)}
ETIQ_ELEM = {"ok": "OK", "falta": "FALTA", "incierto": "?"}
CAMPOS = ["fecha_hora", "ingreso", "origen", "estado_sistema", "clase", "confianza", "backend"]


def registrar(ruta, ingreso, origen, estado, clase, confianza, backend):
    """Una fila por decision. origen = sistema (automatico) | docente (correccion manual).
    clase: completo | falta_<elemento>[_<elemento>] | incierto_<elemento>"""
    nuevo = not ruta.exists()
    with ruta.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, CAMPOS)
        if nuevo:
            w.writeheader()
        w.writerow({"fecha_hora": datetime.now().isoformat(timespec="seconds"),
                    "ingreso": ingreso, "origen": origen, "estado_sistema": estado,
                    "clase": clase, "confianza": round(confianza, 3), "backend": backend})


def dibujar(frame, categoria, texto, probs, estados, fps, lat_ms, aviso, error):
    h, w = frame.shape[:2]
    color = COLOR[categoria]
    cv2.rectangle(frame, (0, 0), (w, 50), color, -1)
    ancho = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0]
    escala = min(0.9, 0.9 * (w - 24) / max(ancho, 1))  # que el texto siempre quepa
    cv2.putText(frame, texto, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), 2)
    if categoria != "espera":
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 8)
    y = 84
    if estados:  # panel por elemento: COFIA: OK (0.93) / GUANTES: FALTA (0.08) ...
        for e, s in estados.items():
            cv2.rectangle(frame, (12, y - 24), (290, y + 8), (0, 0, 0), -1)
            cv2.putText(frame, f"{e.upper()}: {ETIQ_ELEM[s]}  ({probs[e]:.2f})", (18, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ELEM[s], 2)
            y += 38
    else:
        for c, p in sorted((probs or {}).items(), key=lambda kv: -kv[1]):
            cv2.putText(frame, f"{c}: {p:.2f}", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2)
            y += 22
    pie = f"FPS {fps:.1f} | latencia {lat_ms:.0f} ms | a=autorizar r=rechazar q=salir"
    cv2.putText(frame, pie, (14, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if aviso:
        cv2.putText(frame, aviso, (14, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if error:
        cv2.putText(frame, error, (14, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["multi", "cnn", "clasico", "api"], default="clasico")
    ap.add_argument("--fuente", default="0", help="0 = camara USB | archivo de video | rtsp://...")
    ap.add_argument("--url", help="endpoint /predict (solo backend api)")
    ap.add_argument("--modelo", help="ruta alternativa al modelo (multi, cnn o clasico)")
    ap.add_argument("--umbral", type=float, default=UMBRAL_DEFECTO,
                    help="confianza minima para denegar; por debajo -> INCIERTO")
    ap.add_argument("--umbral-aprobar", type=float, default=UMBRAL_APROBAR,
                    help="confianza minima para APROBAR el ingreso (mas estricta)")
    ap.add_argument("--ventana", type=int, default=10, help="fotogramas del promedio temporal")
    ap.add_argument("--confirmar", type=int, default=8,
                    help="evaluaciones seguidas con el mismo resultado antes de decidir")
    ap.add_argument("--cada", type=int, default=2, help="clasificar 1 de cada N fotogramas")
    ap.add_argument("--gracia", type=int, default=15,
                    help="fotogramas sin rostro antes de dar por ausente al estudiante")
    ap.add_argument("--sin-presencia", action="store_true", help="analizar siempre")
    ap.add_argument("--sin-ventana", action="store_true", help="sin pantalla: imprime en consola")
    ap.add_argument("--registro", default="registro_inspecciones.csv")
    a = ap.parse_args()

    predictor = crear_predictor(a.backend, a.url, a.modelo)
    suav = Suavizador(a.ventana)
    cascada = None
    if not a.sin_presencia:
        cascada = cv2.CascadeClassifier(cv2.data.haarcascades +
                                        "haarcascade_frontalface_default.xml")
        if cascada.empty():
            raise SystemExit("No se pudo cargar el detector de rostro (use OpenCV 4.x) "
                             "o ejecute con --sin-presencia.")

    cap = cv2.VideoCapture(int(a.fuente) if a.fuente.isdigit() else a.fuente)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la fuente de video: {a.fuente}")
    ruta_reg = Path(a.registro)

    n = sin_cara = 0
    fps, lat_ms, t_fps, n_fps = 0.0, 0.0, time.time(), 0
    error, aviso, aviso_hasta = "", "", 0.0
    categoria, texto, probs, estados = "espera", "INICIANDO...", None, None
    racha, clave_prev, txt_reg = 0, None, None
    ultima = None  # (categoria, clase, confianza) mas reciente, para la correccion del docente
    ultimo_print = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        n_fps += 1
        if time.time() - t_fps >= 1.0:
            fps, n_fps, t_fps = n_fps / (time.time() - t_fps), 0, time.time()

        if cascada is not None:
            caras = cascada.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                             1.2, 5, minSize=(60, 60))
            sin_cara = 0 if len(caras) else sin_cara + 1
        presente = cascada is None or sin_cara < a.gracia

        if not presente:  # el estudiante se fue: se reinicia todo para el siguiente
            suav.limpiar()
            racha, clave_prev, txt_reg, ultima = 0, None, None, None
            categoria, texto, probs, estados = "espera", "ESPERANDO ESTUDIANTE...", None, None
        elif n % max(1, a.cada) == 0:
            t0 = time.time()
            try:
                suav.agregar(predictor.predecir(frame))
                error = ""
            except Exception as e:  # p. ej. la API no responde: seguir, no cerrar el programa
                error = f"ERROR: {str(e)[:60]}"
            lat_ms = (time.time() - t0) * 1000
            prom = suav.promedio()
            if prom is None:
                categoria, texto, probs, estados = "espera", "ANALIZANDO...", None, None
            else:
                cat, txt, estados, clase, conf = decidir(predictor.tipo, prom, a.umbral,
                                                         a.umbral_aprobar)
                probs, ultima = prom, (cat, clase, conf)
                racha = racha + 1 if (cat, txt) == clave_prev else 1
                clave_prev = (cat, txt)
                if racha >= a.confirmar:  # decision estable: no cambia por un fotograma ruidoso
                    categoria, texto = cat, txt
                    if cat in ("aprobado", "denegado") and txt != txt_reg:
                        registrar(ruta_reg, cat, "sistema", cat, clase, conf, a.backend)
                        txt_reg = txt
                else:
                    categoria, texto = "espera", "ANALIZANDO..."

        if a.sin_ventana:
            if (categoria, texto) != ultimo_print:
                detalle = "  " + " ".join(f"{e}={s}" for e, s in estados.items()) if estados else ""
                print(f"{datetime.now():%H:%M:%S} [{categoria}] {texto}{detalle} {error}")
                ultimo_print = (categoria, texto)
            continue

        if time.time() > aviso_hasta:
            aviso = ""
        dibujar(frame, categoria, texto, probs, estados, fps, lat_ms, aviso, error)
        cv2.imshow("AGROINTEP - Control de ingreso", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k in (ord("a"), ord("r")):
            if ultima is None:
                aviso, aviso_hasta = "SIN RESULTADO PARA REGISTRAR", time.time() + 2
            else:
                ingreso = "aprobado" if k == ord("a") else "denegado"
                registrar(ruta_reg, ingreso, "docente", ultima[0], ultima[1], ultima[2], a.backend)
                aviso, aviso_hasta = f"DOCENTE: INGRESO {ingreso.upper()}", time.time() + 2

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
