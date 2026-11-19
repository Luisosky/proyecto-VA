"""
Generador de Página HTML Interactiva para Clasificación de Perros (Tamaño + Bozal + Alerta de Seguridad)
Procesa el conjunto real dataset_seguridad_canina/test/ con Visión Clásica (OpenCV) y YOLOv8 en paralelo,
sin ningún dato simulado ni valores hardcodeados.
"""

import base64
import glob
import json
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
CLASES_TAMANO = ["Pequeno", "Mediano", "Grande"]


def img_to_b64(img_bgr, max_side=360):
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 84])
    return base64.b64encode(buf).decode("utf-8")


def clasificar_tamano(area_px, w, h):
    ratio = float(area_px) / float(max(1, w * h))
    if ratio >= UMBRAL_GRANDE:
        return "Grande", ratio
    if ratio >= UMBRAL_MEDIANO:
        return "Mediano", ratio
    return "Pequeno", ratio


def alerta_seguridad(tamano, tiene_bozal):
    if tamano == "Grande" and not tiene_bozal:
        return "ALERTA_SEGURIDAD", (0, 0, 255)
    if tamano == "Grande" and tiene_bozal:
        return "CUMPLE_NORMATIVA", (0, 200, 0)
    return "RIESGO_BAJO", (255, 165, 0)


def calcular_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = float(a[2] * a[3] + b[2] * b[3] - inter)
    return inter / union if union > 0 else 0.0


def contencion(inner, outer):
    xA, yA = max(inner[0], outer[0]), max(inner[1], outer[1])
    xB, yB = min(inner[0] + inner[2], outer[0] + outer[2]), min(inner[1] + inner[3], outer[1] + outer[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area = float(inner[2] * inner[3])
    return inter / area if area > 0 else 0.0


def cargar_gt(img_path, lbl_dir, w, h):
    lbl_path = os.path.join(lbl_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
    dogs, muzzles = [], []
    if os.path.exists(lbl_path):
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    cid = int(p[0])
                    cx, cy, bw, bh = map(float, p[1:5])
                    box = (int((cx - bw / 2) * w), int((cy - bh / 2) * h), int(bw * w), int(bh * h))
                    if cid == 0:
                        dogs.append(box)
                    elif cid == 1:
                        muzzles.append(box)
    return dogs, muzzles


def analizar_bozal_clasico(img_bgr, box):
    h_i, w_i = img_bgr.shape[:2]
    x, y, w, h = box
    roi = img_bgr[max(0, y):min(h_i, y + int(h * 0.70)), max(0, x):min(w_i, x + w)]
    if roi.size == 0 or roi.shape[0] < 16 or roi.shape[1] < 16:
        return False, 0.0, 0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    edges = cv2.Canny(gray_eq, 80, 190)
    dens = float(np.count_nonzero(edges)) / float(edges.size)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=max(12, int(min(roi.shape[:2]) * 0.12)), maxLineGap=5)
    n_lin = len(lines) if lines is not None else 0
    score = min(1.0, 0.50 * min(1.0, dens / 0.25) + 0.50 * min(1.0, n_lin / 30.0))
    return (score >= 0.70), score, n_lin


def evaluar_clasico(img_bgr):
    t0 = time.perf_counter()
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 25, 35]), np.array([35, 220, 255]))
    mask = cv2.medianBlur(mask, 5)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img_bgr.copy()
    best = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < h * w * 0.025:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        tam, ratio = clasificar_tamano(bw * bh, w, h)
        tiene_b, sc_b, n_lin = analizar_bozal_clasico(img_bgr, (bx, by, bw, bh))
        alerta, col = alerta_seguridad(tam, tiene_b)
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), col, 3)
        cv2.putText(out, f"{tam} | {'BOZAL' if tiene_b else 'SIN'} ({sc_b:.2f})",
                    (bx, max(22, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2, cv2.LINE_AA)
        if area > best_area:
            best_area = area
            best = {
                "box": (bx, by, bw, bh), "tamano": tam, "ratio": round(ratio * 100, 1),
                "tiene_bozal": tiene_b, "score_bozal": round(sc_b, 2),
                "lineas_hough": n_lin, "alerta": alerta
            }
    ms = (time.perf_counter() - t0) * 1000.0
    return out, best, ms


def evaluar_yolo(img_bgr, model, entrenado):
    t0 = time.perf_counter()
    out = img_bgr.copy()
    h, w = img_bgr.shape[:2]
    if model is None:
        return out, None, 0.0
    res = model(img_bgr, conf=0.35, verbose=False)[0]
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
    best = None
    best_area = 0.0
    for box, cf in dogs:
        x, y, bw, bh = box
        tam, ratio = clasificar_tamano(bw * bh, w, h)
        tiene_b = any(contencion(mb, box) >= 0.20 for mb, _ in muzzles)
        alerta, col = alerta_seguridad(tam, tiene_b)
        cv2.rectangle(out, (x, y), (x + bw, y + bh), col, 3)
        cv2.putText(out, f"{tam} ({cf:.2f}) | {'BOZAL' if tiene_b else 'SIN'}",
                    (x, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2, cv2.LINE_AA)
        if bw * bh > best_area:
            best_area = bw * bh
            best = {
                "box": box, "tamano": tam, "ratio": round(ratio * 100, 1),
                "conf": round(cf, 2), "tiene_bozal": tiene_b, "alerta": alerta
            }
    for (mx, my, mw, mh), cf in muzzles:
        cv2.rectangle(out, (mx, my), (mx + mw, my + mh), (255, 255, 0), 2)
    ms = (time.perf_counter() - t0) * 1000.0
    return out, best, ms


def dibujar_gt(img_bgr, gt_dogs, gt_muzzles):
    out = img_bgr.copy()
    h, w = img_bgr.shape[:2]
    for db in gt_dogs:
        x, y, bw, bh = db
        tam, _ = clasificar_tamano(bw * bh, w, h)
        tiene_b = any(contencion(mb, db) >= 0.20 for mb in gt_muzzles)
        alerta, col = alerta_seguridad(tam, tiene_b)
        cv2.rectangle(out, (x, y), (x + bw, y + bh), col, 3)
        cv2.putText(out, f"GT: {tam} | {'BOZAL' if tiene_b else 'SIN'}",
                    (x, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2, cv2.LINE_AA)
    for mx, my, mw, mh in gt_muzzles:
        cv2.rectangle(out, (mx, my), (mx + mw, my + mh), (255, 255, 0), 2)
    return out


def generar_pagina_clasificacion(
    dataset_dir="dataset_seguridad_canina",
    pesos="dataset_seguridad_canina/seguridad_canina_yolov8n/weights/best.pt",
    output_html="visor_clasificacion_perros.html",
    max_cards=36
):
    img_dir = os.path.join(dataset_dir, "test", "images")
    lbl_dir = os.path.join(dataset_dir, "test", "labels")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not imgs:
        print(f"[ERROR] No hay imágenes en {img_dir}")
        return False

    model, entrenado = None, False
    if HAS_YOLO:
        if os.path.exists(pesos):
            model, entrenado = YOLO(pesos), True
            motor_yolo = f"YOLOv8 Entrenado (2 clases: dog + muzzle)"
        else:
            model, entrenado = YOLO("yolov8n.pt"), False
            motor_yolo = "YOLOv8n Base COCO (entrena Celda 9 en Colab para activar clase muzzle)"
    else:
        motor_yolo = "Ultralytics no instalado"

    print(f"-> Clasificando {len(imgs)} imágenes reales de {img_dir}...")

    cm_tam_c = np.zeros((3, 3), dtype=int)
    cm_tam_y = np.zeros((3, 3), dtype=int)
    cm_boz_c = np.zeros((2, 2), dtype=int)  # [[TN, FP], [FN, TP]]
    cm_boz_y = np.zeros((2, 2), dtype=int)
    conteo_gt_tam = {"Pequeno": 0, "Mediano": 0, "Grande": 0}
    conteo_gt_alerta = {"ALERTA_SEGURIDAD": 0, "CUMPLE_NORMATIVA": 0, "RIESGO_BAJO": 0}
    tiempos_c, tiempos_y = [], []

    items = []
    for p in imgs:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_dogs, gt_muzzles = cargar_gt(p, lbl_dir, w, h)
        if not gt_dogs:
            continue

        # Perro principal del Ground Truth (mayor caja)
        gt_main = max(gt_dogs, key=lambda b: b[2] * b[3])
        gt_tam, gt_ratio = clasificar_tamano(gt_main[2] * gt_main[3], w, h)
        gt_bozal = any(contencion(mb, gt_main) >= 0.20 for mb in gt_muzzles)
        gt_alerta, _ = alerta_seguridad(gt_tam, gt_bozal)
        conteo_gt_tam[gt_tam] += 1
        conteo_gt_alerta[gt_alerta] += 1

        out_c, best_c, ms_c = evaluar_clasico(img)
        out_y, best_y, ms_y = evaluar_yolo(img, model, entrenado)
        out_gt = dibujar_gt(img, gt_dogs, gt_muzzles)
        tiempos_c.append(ms_c)
        tiempos_y.append(ms_y)

        if best_c is not None:
            cm_tam_c[CLASES_TAMANO.index(gt_tam), CLASES_TAMANO.index(best_c["tamano"])] += 1
            cm_boz_c[1 if gt_bozal else 0, 1 if best_c["tiene_bozal"] else 0] += 1
        if best_y is not None:
            cm_tam_y[CLASES_TAMANO.index(gt_tam), CLASES_TAMANO.index(best_y["tamano"])] += 1
            cm_boz_y[1 if gt_bozal else 0, 1 if best_y["tiene_bozal"] else 0] += 1

        items.append({
            "file": os.path.basename(p),
            "gt_tam": gt_tam,
            "gt_ratio": round(gt_ratio * 100, 1),
            "gt_bozal": gt_bozal,
            "gt_alerta": gt_alerta,
            "clasico": best_c,
            "yolo": best_y,
            "ms_c": round(ms_c, 1),
            "ms_y": round(ms_y, 1),
            "img_c": out_c,
            "img_y": out_y,
            "img_gt": out_gt,
        })

    # Seleccionar muestra balanceada (mitad con bozal, mitad sin bozal, cubriendo todos los tamaños)
    con_b = [x for x in items if x["gt_bozal"]]
    sin_b = [x for x in items if not x["gt_bozal"]]
    half = max_cards // 2
    sel = con_b[:half] + sin_b[:half]
    # Intercalar para vista variada
    gallery = []
    for i in range(max(len(con_b[:half]), len(sin_b[:half]))):
        if i < len(con_b[:half]):
            gallery.append(con_b[i])
        if i < len(sin_b[:half]):
            gallery.append(sin_b[i])

    # Convertir solo la galería seleccionada a base64 para mantener el HTML ágil (~2-3 MB)
    cards_json = []
    for it in gallery:
        cards_json.append({
            "file": it["file"],
            "gt_tam": it["gt_tam"],
            "gt_ratio": it["gt_ratio"],
            "gt_bozal": it["gt_bozal"],
            "gt_alerta": it["gt_alerta"],
            "clasico": it["clasico"],
            "yolo": it["yolo"],
            "ms_c": it["ms_c"],
            "ms_y": it["ms_y"],
            "b64_c": img_to_b64(it["img_c"]),
            "b64_y": img_to_b64(it["img_y"]),
            "b64_gt": img_to_b64(it["img_gt"]),
        })

    acc_tam_c = (np.trace(cm_tam_c) / max(1, cm_tam_c.sum())) * 100.0
    acc_tam_y = (np.trace(cm_tam_y) / max(1, cm_tam_y.sum())) * 100.0
    acc_boz_c = (np.trace(cm_boz_c) / max(1, cm_boz_c.sum())) * 100.0
    acc_boz_y = (np.trace(cm_boz_y) / max(1, cm_boz_y.sum())) * 100.0

    def render_cm_table(cm, labels, title):
        mx = max(1, int(cm.max()))
        rows = ""
        for i, rl in enumerate(labels):
            cells = f"<th style='text-align:left'>{rl}</th>"
            for j in range(len(labels)):
                v = int(cm[i, j])
                alpha = 0.12 + 0.65 * (v / mx)
                bg = f"rgba(56,189,248,{alpha:.2f})" if i == j else f"rgba(248,113,113,{alpha*0.7:.2f})"
                cells += f"<td style='background:{bg};font-weight:700;text-align:center'>{v}</td>"
            rows += f"<tr>{cells}</tr>"
        hdr = "".join(f"<th>{l}</th>" for l in labels)
        return f"<div class='cm-card'><h4>{title}</h4><table><thead><tr><th>Real \\ Pred</th>{hdr}</tr></thead><tbody>{rows}</tbody></table></div>"

    cm_html = (
        render_cm_table(cm_tam_c, CLASES_TAMANO, f"Tamaño 3×3 — Clásico ({acc_tam_c:.1f}%)") +
        render_cm_table(cm_tam_y, CLASES_TAMANO, f"Tamaño 3×3 — YOLOv8 ({acc_tam_y:.1f}%)") +
        render_cm_table(cm_boz_c, ["Sin Bozal", "Con Bozal"], f"Bozal 2×2 — Clásico ({acc_boz_c:.1f}%)") +
        render_cm_table(cm_boz_y, ["Sin Bozal", "Con Bozal"], f"Bozal 2×2 — YOLOv8 ({acc_boz_y:.1f}%)")
    )

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Panel de Clasificación de Perros — Tamaño, Bozal y Seguridad</title>
  <style>
    :root {{ --bg:#0b0f19; --surface:#151d2c; --surface2:#1b2538; --border:#27354d; --text:#cbd5e1; --bright:#fff; --accent:#38bdf8; --green:#22c55e; --red:#ef4444; --orange:#f59e0b; }}
    * {{ box-sizing:border-box; margin:0; padding:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
    body {{ background:var(--bg); color:var(--text); padding:22px; line-height:1.45; }}
    header {{ background:linear-gradient(135deg,#1e293b,#0f172a); border:1px solid var(--border); padding:22px 28px; border-radius:14px; margin-bottom:20px; }}
    header h1 {{ color:var(--bright); font-size:1.7rem; margin-bottom:6px; }}
    header p {{ color:#94a3b8; font-size:.9rem; }}
    .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(175px,1fr)); gap:14px; margin-bottom:20px; }}
    .kpi {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:14px 16px; }}
    .kpi .lbl {{ font-size:.78rem; color:#94a3b8; text-transform:uppercase; letter-spacing:.04em; }}
    .kpi .val {{ font-size:1.5rem; font-weight:800; color:var(--bright); margin-top:4px; }}
    .kpi .sub {{ font-size:.78rem; color:var(--accent); margin-top:2px; }}

    .controls {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:14px 18px; margin-bottom:20px; display:flex; flex-wrap:wrap; gap:14px; align-items:center; justify-content:space-between; }}
    .ctrl-group {{ display:flex; gap:6px; align-items:center; flex-wrap:wrap; }}
    .ctrl-group span {{ font-size:.82rem; font-weight:700; color:#94a3b8; margin-right:4px; }}
    .btn {{ background:var(--surface2); border:1px solid var(--border); color:var(--text); padding:6px 12px; border-radius:8px; cursor:pointer; font-size:.84rem; font-weight:600; transition:.15s; }}
    .btn:hover {{ border-color:var(--accent); color:var(--bright); }}
    .btn.active {{ background:var(--accent); color:#0b0f19; border-color:var(--accent); }}

    .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(295px,1fr)); gap:18px; }}
    .card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; overflow:hidden; display:flex; flex-direction:column; transition:transform .15s; }}
    .card:hover {{ transform:translateY(-3px); border-color:var(--accent); }}
    .card-top {{ padding:10px 14px; display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border); gap:8px; }}
    .card-top .fname {{ font-size:.78rem; color:#94a3b8; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:140px; }}
    .pill {{ padding:3px 9px; border-radius:20px; font-size:.74rem; font-weight:700; display:inline-block; }}
    .p-alert {{ background:rgba(239,68,68,.2); color:#fca5a5; border:1px solid rgba(239,68,68,.5); }}
    .p-ok {{ background:rgba(34,197,94,.2); color:#86efac; border:1px solid rgba(34,197,94,.5); }}
    .p-low {{ background:rgba(245,158,11,.2); color:#fcd34d; border:1px solid rgba(245,158,11,.5); }}
    .p-info {{ background:rgba(56,189,248,.16); color:#7dd3fc; border:1px solid rgba(56,189,248,.35); }}

    .img-box {{ background:#000; aspect-ratio:1/1; display:flex; align-items:center; justify-content:center; cursor:zoom-in; }}
    .img-box img {{ width:100%; height:100%; object-fit:contain; }}
    .card-body {{ padding:12px 14px; font-size:.82rem; display:flex; flex-direction:column; gap:7px; }}
    .row {{ display:flex; justify-content:space-between; align-items:center; border-bottom:1px dashed rgba(255,255,255,.07); padding-bottom:5px; }}
    .row:last-child {{ border-bottom:none; padding-bottom:0; }}

    .cm-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px; margin-bottom:22px; }}
    .cm-card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:14px; }}
    .cm-card h4 {{ color:var(--bright); margin-bottom:10px; font-size:.92rem; }}
    .cm-card table {{ width:100%; border-collapse:collapse; font-size:.82rem; }}
    .cm-card th, .cm-card td {{ padding:7px 8px; border:1px solid var(--border); }}
    .cm-card th {{ color:#94a3b8; font-weight:600; }}

    .modal {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.92); z-index:1000; align-items:center; justify-content:center; flex-direction:column; gap:12px; }}
    .modal.show {{ display:flex; }}
    .modal img {{ max-width:90vw; max-height:82vh; border-radius:8px; }}
    .modal p {{ color:#cbd5e1; font-size:.95rem; }}
  </style>
</head>
<body>
  <header>
    <h1>🐾 Panel Interactivo de Clasificación Canina (Tamaño + Bozal + Seguridad)</h1>
    <p>Evaluación real sobre <strong>{len(items)}</strong> perros del conjunto <code>dataset_seguridad_canina/test</code> | Motor DL: <strong>{motor_yolo}</strong></p>
  </header>

  <section class="kpis">
    <div class="kpi"><div class="lbl">Perros en Test</div><div class="val">{len(items)}</div><div class="sub">Pequeño: {conteo_gt_tam['Pequeno']} · Med: {conteo_gt_tam['Mediano']} · Gr: {conteo_gt_tam['Grande']}</div></div>
    <div class="kpi"><div class="lbl">🔴 Alertas Reales</div><div class="val" style="color:var(--red)">{conteo_gt_alerta['ALERTA_SEGURIDAD']}</div><div class="sub">Grande + Sin bozal</div></div>
    <div class="kpi"><div class="lbl">🟢 Cumplen Normativa</div><div class="val" style="color:var(--green)">{conteo_gt_alerta['CUMPLE_NORMATIVA']}</div><div class="sub">Grande + Con bozal</div></div>
    <div class="kpi"><div class="lbl">Clásico — Bozal</div><div class="val">{acc_boz_c:.1f}%</div><div class="sub">Canny + Hough ({np.mean(tiempos_c):.1f} ms)</div></div>
    <div class="kpi"><div class="lbl">Clásico — Tamaño</div><div class="val">{acc_tam_c:.1f}%</div><div class="sub">Matriz 3×3 en detectados</div></div>
    <div class="kpi"><div class="lbl">YOLOv8 — Tamaño</div><div class="val">{acc_tam_y:.1f}%</div><div class="sub">Latencia media: {np.mean(tiempos_y):.1f} ms</div></div>
  </section>

  <section class="cm-grid">{cm_html}</section>

  <section class="controls">
    <div class="ctrl-group">
      <span>Ver anotación:</span>
      <button class="btn active" onclick="setView(this,'c')">Clásico (OpenCV)</button>
      <button class="btn" onclick="setView(this,'y')">YOLOv8</button>
      <button class="btn" onclick="setView(this,'gt')">Ground Truth Real</button>
    </div>
    <div class="ctrl-group">
      <span>Tamaño (GT):</span>
      <button class="btn active" onclick="setFilter(this,'tam','ALL')">Todos</button>
      <button class="btn" onclick="setFilter(this,'tam','Pequeno')">Pequeño</button>
      <button class="btn" onclick="setFilter(this,'tam','Mediano')">Mediano</button>
      <button class="btn" onclick="setFilter(this,'tam','Grande')">Grande</button>
    </div>
    <div class="ctrl-group">
      <span>Bozal (GT):</span>
      <button class="btn active" onclick="setFilter(this,'boz','ALL')">Todos</button>
      <button class="btn" onclick="setFilter(this,'boz','SI')">Con Bozal</button>
      <button class="btn" onclick="setFilter(this,'boz','NO')">Sin Bozal</button>
    </div>
    <div class="ctrl-group">
      <span>Alerta (GT):</span>
      <button class="btn active" onclick="setFilter(this,'ale','ALL')">Todas</button>
      <button class="btn" onclick="setFilter(this,'ale','ALERTA_SEGURIDAD')">🔴 Alerta</button>
      <button class="btn" onclick="setFilter(this,'ale','CUMPLE_NORMATIVA')">🟢 Cumple</button>
      <button class="btn" onclick="setFilter(this,'ale','RIESGO_BAJO')">⚪ Riesgo Bajo</button>
    </div>
  </section>

  <p id="counter" style="margin-bottom:14px;color:#94a3b8;font-size:.88rem;"></p>
  <section id="gallery" class="grid"></section>

  <div id="modal" class="modal" onclick="this.classList.remove('show')">
    <img id="mImg"><p id="mCap"></p>
  </div>

  <script>
    const DATA = {json.dumps(cards_json, ensure_ascii=False)};
    let viewMode = 'c';
    const filters = {{ tam: 'ALL', boz: 'ALL', ale: 'ALL' }};

    function pillClass(alerta) {{
      if (alerta === 'ALERTA_SEGURIDAD') return 'p-alert';
      if (alerta === 'CUMPLE_NORMATIVA') return 'p-ok';
      return 'p-low';
    }}
    function setView(btn, m) {{
      btn.parentElement.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active'); viewMode = m; render();
    }}
    function setFilter(btn, key, val) {{
      btn.parentElement.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active'); filters[key] = val; render();
    }}
    function openModal(src, cap) {{
      document.getElementById('mImg').src = src;
      document.getElementById('mCap').textContent = cap;
      document.getElementById('modal').classList.add('show');
    }}

    function render() {{
      const filtered = DATA.filter(d => {{
        if (filters.tam !== 'ALL' && d.gt_tam !== filters.tam) return false;
        if (filters.boz === 'SI' && !d.gt_bozal) return false;
        if (filters.boz === 'NO' && d.gt_bozal) return false;
        if (filters.ale !== 'ALL' && d.gt_alerta !== filters.ale) return false;
        return true;
      }});
      document.getElementById('counter').textContent = `Mostrando ${{filtered.length}} de ${{DATA.length}} perros de la galería (evaluación completa en KPIs: {len(items)} perros)`;

      const g = document.getElementById('gallery');
      g.innerHTML = filtered.map((d, idx) => {{
        const src = 'data:image/jpeg;base64,' + (viewMode === 'c' ? d.b64_c : (viewMode === 'y' ? d.b64_y : d.b64_gt));
        const cTxt = d.clasico
          ? `${{d.clasico.tamano}} (${{d.clasico.ratio}}%) · ${{d.clasico.tiene_bozal ? 'BOZAL' : 'SIN'}} (${{d.clasico.score_bozal}})`
          : 'Sin detección';
        const yTxt = d.yolo
          ? `${{d.yolo.tamano}} (${{d.yolo.conf}}) · ${{d.yolo.tiene_bozal ? 'BOZAL' : 'SIN'}}`
          : 'Sin detección';
        const okC = d.clasico && d.clasico.tiene_bozal === d.gt_bozal ? '✅' : '❌';
        const okTamC = d.clasico && d.clasico.tamano === d.gt_tam ? '✅' : '❌';
        return `
          <article class="card">
            <div class="card-top">
              <span class="fname" title="${{d.file}}">${{d.file}}</span>
              <span class="pill ${{pillClass(d.gt_alerta)}}">${{d.gt_alerta.replace('_',' ')}}</span>
            </div>
            <div class="img-box" onclick="openModal('${{src}}', '${{d.file}} — Real: ${{d.gt_tam}} | ${{d.gt_bozal ? 'CON BOZAL' : 'SIN BOZAL'}}')">
              <img src="${{src}}" loading="lazy">
            </div>
            <div class="card-body">
              <div class="row"><strong>Real (GT):</strong><span class="pill p-info">${{d.gt_tam}} (${{d.gt_ratio}}%) · ${{d.gt_bozal ? 'CON BOZAL' : 'SIN BOZAL'}}</span></div>
              <div class="row"><span>Clásico (${{d.ms_c}} ms):</span><span>${{cTxt}} ${{okTamC}}${{okC}}</span></div>
              <div class="row"><span>YOLOv8 (${{d.ms_y}} ms):</span><span>${{yTxt}}</span></div>
            </div>
          </article>`;
      }}).join('');
    }}
    render();
    document.addEventListener('keydown', e => {{ if (e.key === 'Escape') document.getElementById('modal').classList.remove('show'); }});
  </script>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[EXITO] Panel de clasificación generado en: {os.path.abspath(output_html)} ({len(cards_json)} tarjetas en galería)")
    return True


if __name__ == "__main__":
    generar_pagina_clasificacion()
