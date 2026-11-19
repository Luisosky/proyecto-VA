"""
Demostración EN VIVO — Defensa ante Ataque Adversario y Perturbaciones (40% de la Rúbrica)
Proyecto: Auditoría Híbrida de Modelos de Visión (Seguridad Ciudadana: Tamaño + Bozal)

Uso:
  python demo_en_vivo.py                         # Usa webcam 0 (o galería test si no hay cámara)
  python demo_en_vivo.py --fuente test           # Recorre imágenes reales de dataset_seguridad_canina/test
  python demo_en_vivo.py --fuente video.mp4      # Ejecuta sobre un archivo de video
  python demo_en_vivo.py --pesos ruta/best.pt    # Carga pesos entrenados de YOLOv8 (2 clases: dog, muzzle)

Controles de teclado en vivo:
  [1] Sin perturbación (Limpio)
  [2] Ataque: Baja luz nocturna (alpha=0.35)
  [3] Ataque: Crítico mixto (Baja luz + Ruido térmico ISO alto + Blur)
  [4] Ataque: Desenfoque óptico (destruye altas frecuencias del bozal)
  [5] Ataque: Sobreexposición / Deslumbramiento
  [d] Activar / Desactivar DEFENSA MATEMÁTICA (Bilateral + Gamma + CLAHE Lab)
  [m] Cambiar vista: PARALELO (Clásico vs YOLOv8) / SOLO CLÁSICO / SOLO YOLOV8
  [n] / [p] Siguiente / Anterior imagen (en modo galería 'test')
  [q] Salir
"""

import argparse
import glob
import os
import time
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

UMBRAL_MEDIANO = 0.12
UMBRAL_GRANDE = 0.30


def clasificar_tamano(area_px, w_img, h_img):
    ratio = float(area_px) / float(max(1, w_img * h_img))
    if ratio >= UMBRAL_GRANDE:
        return "Grande", ratio
    if ratio >= UMBRAL_MEDIANO:
        return "Mediano", ratio
    return "Pequeno", ratio


def alerta_seguridad(tamano, tiene_bozal):
    if tamano == "Grande" and not tiene_bozal:
        return "ALERTA", (0, 0, 255)
    if tamano == "Grande" and tiene_bozal:
        return "CUMPLE", (0, 200, 0)
    return "RIESGO_BAJO", (255, 165, 0)


def contencion(inner, outer):
    xA, yA = max(inner[0], outer[0]), max(inner[1], outer[1])
    xB = min(inner[0] + inner[2], outer[0] + outer[2])
    yB = min(inner[1] + inner[3], outer[1] + outer[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area = float(inner[2] * inner[3])
    return inter / area if area > 0 else 0.0


def aplicar_ataque(img, modo):
    if modo == 1:
        return img, "LIMPIO"
    if modo == 2:
        return np.clip(img.astype(np.float32) * 0.35, 0, 255).astype(np.uint8), "ATAQUE: BAJA LUZ (0.35x)"
    if modo == 3:
        dark = np.clip(img.astype(np.float32) * 0.40, 0, 255)
        noise = np.random.normal(0, 22, img.shape).astype(np.float32)
        out = np.clip(dark + noise, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(out, (5, 5), 0), "ATAQUE: CRITICO (LUZ+RUIDO+BLUR)"
    if modo == 4:
        return cv2.GaussianBlur(img, (15, 15), 0), "ATAQUE: DESENFOQUE (15x15)"
    if modo == 5:
        return np.clip(img.astype(np.float32) * 1.7 + 40, 0, 255).astype(np.uint8), "ATAQUE: DESLUMBRAMIENTO"
    return img, "LIMPIO"


def aplicar_defensa(img):
    denoised = cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)
    lut = np.array([((i / 255.0) ** (1.0 / 0.65)) * 255 for i in range(256)], dtype=np.uint8)
    gamma_img = cv2.LUT(denoised, lut)
    lab = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)


def analizar_bozal_clasico(img, box):
    h_i, w_i = img.shape[:2]
    x, y, w, h = box
    roi = img[max(0, y):min(h_i, y + int(h * 0.70)), max(0, x):min(w_i, x + w)]
    if roi.size == 0 or roi.shape[0] < 16 or roi.shape[1] < 16:
        return False, 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    edges = cv2.Canny(gray_eq, 80, 190)
    dens = float(np.count_nonzero(edges)) / float(edges.size)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=max(12, int(min(roi.shape[:2]) * 0.12)), maxLineGap=5)
    n_lin = len(lines) if lines is not None else 0
    score = min(1.0, 0.50 * min(1.0, dens / 0.25) + 0.50 * min(1.0, n_lin / 30.0))
    return score >= 0.70, score


def correr_clasico(img):
    t0 = time.perf_counter()
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 25, 35]), np.array([35, 220, 255]))
    mask = cv2.medianBlur(mask, 5)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    n_det = 0
    for c in contours:
        if cv2.contourArea(c) < h * w * 0.025:
            continue
        n_det += 1
        bx, by, bw, bh = cv2.boundingRect(c)
        tam, _ = clasificar_tamano(bw * bh, w, h)
        tiene_b, sc_b = analizar_bozal_clasico(img, (bx, by, bw, bh))
        alerta, col = alerta_seguridad(tam, tiene_b)
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), col, 2)
        cv2.putText(out, f"{tam} | {'BOZAL' if tiene_b else 'SIN'} ({sc_b:.2f}) | {alerta}",
                    (bx, max(20, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2, cv2.LINE_AA)
    ms = (time.perf_counter() - t0) * 1000.0
    cv2.putText(out, f"CLASICO | {ms:.1f} ms | Det:{n_det}", (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def correr_yolo(img, model, entrenado):
    t0 = time.perf_counter()
    out = img.copy()
    h, w = img.shape[:2]
    if model is None:
        cv2.putText(out, "Instala ultralytics para activar YOLOv8", (20, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return out
    res = model(img, conf=0.35, verbose=False)[0]
    dogs, muzzles = [], []
    for b in res.boxes:
        cid = int(b.cls[0].item())
        cf = float(b.conf[0].item())
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        box = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
        if (entrenado and cid == 0) or (not entrenado and cid == 16):
            dogs.append((box, cf))
        elif entrenado and cid == 1:
            muzzles.append((box, cf))
    for box, cf in dogs:
        x, y, bw, bh = box
        tam, _ = clasificar_tamano(bw * bh, w, h)
        tiene_b = any(contencion(mb, box) >= 0.20 for mb, _ in muzzles)
        alerta, col = alerta_seguridad(tam, tiene_b)
        cv2.rectangle(out, (x, y), (x + bw, y + bh), col, 2)
        cv2.putText(out, f"{tam} {cf:.2f} | {'BOZAL' if tiene_b else 'SIN'} | {alerta}",
                    (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2, cv2.LINE_AA)
    for (mx, my, mw, mh), cf in muzzles:
        cv2.rectangle(out, (mx, my), (mx + mw, my + mh), (255, 255, 0), 2)
    ms = (time.perf_counter() - t0) * 1000.0
    cv2.putText(out, f"YOLOv8 | {ms:.1f} ms | Perros:{len(dogs)} Bozales:{len(muzzles)}", (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description="Demo en vivo: Ataque adversario y defensa matemática")
    parser.add_argument("--fuente", default="test", help="'0' para webcam, ruta de video o 'test' para galería")
    parser.add_argument("--pesos", default="dataset_seguridad_canina/seguridad_canina_yolov8n/weights/best.pt")
    args = parser.parse_args()

    model, entrenado = None, False
    if HAS_YOLO:
        if os.path.exists(args.pesos):
            model, entrenado = YOLO(args.pesos), True
            print(f"[INFO] Cargado modelo entrenado (2 clases): {args.pesos}")
        else:
            model, entrenado = YOLO("yolov8n.pt"), False
            print("[INFO] Pesos entrenados no encontrados; usando yolov8n.pt base (solo clase dog).")

    modo_ataque = 1
    defensa_activa = False
    vista = 0  # 0: Paralelo, 1: Clásico, 2: YOLOv8

    use_gallery = (args.fuente.lower() == "test")
    gallery = sorted(glob.glob("dataset_seguridad_canina/test/images/*.jpg")) if use_gallery else []
    idx_img = 0
    cap = None if use_gallery else cv2.VideoCapture(0 if args.fuente == "0" else args.fuente)

    if use_gallery and not gallery:
        print("[ERROR] No se encontraron imágenes en dataset_seguridad_canina/test/images/")
        return

    print("Controles: [1-5] Ataques | [d] Defensa ON/OFF | [m] Vista | [n/p] Imagen | [q] Salir")
    win_name = "Demo En Vivo - Auditoria Hibrida Seguridad Canina"
    last_state = None
    canvas = None

    while True:
        cur_state = (idx_img, modo_ataque, defensa_activa, vista)
        if not use_gallery or cur_state != last_state:
            if use_gallery:
                frame = cv2.imread(gallery[idx_img])
            else:
                ok, frame = cap.read()
                if not ok:
                    break
            frame = cv2.resize(frame, (540, 540))
            frame_proc, nombre_ataque = aplicar_ataque(frame, modo_ataque)
            if defensa_activa:
                frame_proc = aplicar_defensa(frame_proc)

            if vista == 0:
                v1 = correr_clasico(frame_proc)
                v2 = correr_yolo(frame_proc, model, entrenado)
                canvas = np.hstack((v1, v2))
            elif vista == 1:
                canvas = correr_clasico(frame_proc)
            else:
                canvas = correr_yolo(frame_proc, model, entrenado)

            # Barra superior de estado físico del píxel (luminancia media V)
            v_mean = float(np.mean(cv2.cvtColor(frame_proc, cv2.COLOR_BGR2HSV)[:, :, 2]))
            banner = f"{nombre_ataque} | DEFENSA: {'ON (Bilateral+Gamma+CLAHE)' if defensa_activa else 'OFF'} | Brillo V_medio={v_mean:.1f}"
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 32), (20, 20, 20), -1)
            col_txt = (0, 255, 0) if defensa_activa else ((0, 200, 255) if modo_ataque == 1 else (0, 80, 255))
            cv2.putText(canvas, banner, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col_txt, 1, cv2.LINE_AA)
            cv2.imshow(win_name, canvas)
            last_state = cur_state

        key = cv2.waitKey(30 if not use_gallery else 60) & 0xFF
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("q") or key == 27:
            break
        elif key in [ord("1"), ord("2"), ord("3"), ord("4"), ord("5")]:
            modo_ataque = int(chr(key))
        elif key == ord("d"):
            defensa_activa = not defensa_activa
        elif key == ord("m"):
            vista = (vista + 1) % 3
        elif use_gallery and key == ord("n"):
            idx_img = (idx_img + 1) % len(gallery)
        elif use_gallery and key == ord("p"):
            idx_img = (idx_img - 1) % len(gallery)

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
