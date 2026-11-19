"""
Script Generador de Visor Didáctico Interactivo HTML (100% Real, sin métricas simuladas)
Proyecto: Auditoría Híbrida de Modelos de Visión — Seguridad Ciudadana (Tamaño + Bozal)
"""

import os
import sys
import glob
import base64
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


def img_a_base64(img_bgr):
    _, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buffer).decode("utf-8")


def clasificar_tamano(area_px, w, h):
    r = float(area_px) / float(max(1, w * h))
    if r >= UMBRAL_GRANDE:
        return "Grande", r
    if r >= UMBRAL_MEDIANO:
        return "Mediano", r
    return "Pequeno", r


def alerta_seguridad(tamano, tiene_bozal):
    if tamano == "Grande" and not tiene_bozal:
        return "ALERTA_SEGURIDAD", (0, 0, 255)
    if tamano == "Grande" and tiene_bozal:
        return "CUMPLE_NORMATIVA", (0, 200, 0)
    return "RIESGO_BAJO", (255, 165, 0)


def aplicar_correccion_gamma(img, gamma=0.65):
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def aplicar_clahe_lab(img, clip_limit=3.0, grid_size=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size).apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)


def analizar_bozal_clasico(img_bgr, box):
    h_i, w_i = img_bgr.shape[:2]
    x, y, w, h = box
    roi = img_bgr[max(0, y):min(h_i, y + int(h * 0.70)), max(0, x):min(w_i, x + w)]
    if roi.size == 0 or roi.shape[0] < 16 or roi.shape[1] < 16:
        return False, 0.0, np.zeros((16, 16), dtype=np.uint8), 0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    edges = cv2.Canny(gray_eq, 80, 190)
    dens = float(np.count_nonzero(edges)) / float(edges.size)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=max(12, int(min(roi.shape[:2]) * 0.12)), maxLineGap=5)
    n_lin = len(lines) if lines is not None else 0
    score = min(1.0, 0.50 * min(1.0, dens / 0.25) + 0.50 * min(1.0, n_lin / 30.0))
    return (score >= 0.70), score, edges, n_lin


def detectar_clasico(img_bgr):
    t0 = time.perf_counter()
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_cruda = cv2.inRange(hsv, np.array([0, 25, 35]), np.array([35, 220, 255]))
    mask_med = cv2.medianBlur(mask_cruda, 5)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_morfo = cv2.morphologyEx(cv2.morphologyEx(mask_med, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(mask_morfo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img_bgr.copy()
    dets = []
    edges_best = cv2.Canny(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 70, 170)
    for c in contours:
        area = cv2.contourArea(c)
        if area < h * w * 0.025:
            continue
        perim = cv2.arcLength(c, True)
        circ = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0.0
        bx, by, bw, bh = cv2.boundingRect(c)
        tam, ratio = clasificar_tamano(bw * bh, w, h)
        tiene_b, sc_b, ed, n_lin = analizar_bozal_clasico(img_bgr, (bx, by, bw, bh))
        alerta, col = alerta_seguridad(tam, tiene_b)
        dets.append({"tamano": tam, "area": int(area), "ratio": round(ratio * 100, 1),
                     "ar": round(bw / max(1, bh), 2), "circ": round(circ, 2),
                     "tiene_bozal": tiene_b, "score_bozal": round(sc_b, 2),
                     "lineas_hough": n_lin, "alerta": alerta})
        edges_best = ed
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), col, 3)
        cv2.putText(out, f"{tam} | {'BOZAL' if tiene_b else 'SIN'} ({sc_b:.2f}) | {alerta}",
                    (bx, max(22, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2, cv2.LINE_AA)
    ms = (time.perf_counter() - t0) * 1000.0
    return out, hsv, mask_cruda, mask_med, mask_morfo, edges_best, dets, ms


def detectar_modelo_defensa(img_bgr, pesos="dataset_seguridad_canina/seguridad_canina_yolov8n/weights/best.pt"):
    """Ejecuta inferencia REAL (YOLOv8 si está instalado; de lo contrario, pipeline clásico real)."""
    if HAS_YOLO:
        model_path = pesos if os.path.exists(pesos) else "yolov8n.pt"
        model = YOLO(model_path)
        t0 = time.perf_counter()
        res = model(img_bgr, conf=0.35, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000.0
        n_det = len(res.boxes)
        conf_max = max((float(b.conf[0].item()) for b in res.boxes), default=0.0)
        return res.plot(), f"YOLOv8 ({os.path.basename(model_path)})", n_det, conf_max, ms
    out, _, _, _, _, _, dets, ms = detectar_clasico(img_bgr)
    return out, "Pipeline Clásico (ultralytics no instalado)", len(dets), (dets[0]["score_bozal"] if dets else 0.0), ms


def generar_reporte_html(imagen_path=None, output_html="visor_transformaciones.html"):
    if imagen_path is None or not os.path.exists(imagen_path):
        candidatos = [
            "dataset_seguridad_canina/test/images/muzzle_image10_jpg.jpg",
            *sorted(glob.glob("dataset_seguridad_canina/test/images/muzzle_*.jpg")),
            *sorted(glob.glob("dog.v1i.yolov8/test/images/*.jpg"))
        ]
        imagen_path = next((p for p in candidatos if os.path.exists(p)), None)

    if not imagen_path:
        print("[ERROR] No se encontró imagen de prueba.")
        return False

    print(f"-> Procesando transformaciones REALES sobre: {imagen_path}")
    img_bgr = cv2.imread(imagen_path)

    # 1. Pipeline Clásico
    img_cdet, hsv, mask_cruda, mask_med, mask_morfo, edges_roi, dets_c, ms_clasico = detectar_clasico(img_bgr)
    h_col = cv2.applyColorMap((hsv[:, :, 0] * (255.0 / 179.0)).astype(np.uint8), cv2.COLORMAP_HSV)
    s_col = cv2.applyColorMap(hsv[:, :, 1], cv2.COLORMAP_PLASMA)
    edges_bgr = cv2.cvtColor(edges_roi, cv2.COLOR_GRAY2BGR)

    # 2. Perturbaciones reales (con semilla reproducible)
    rng = np.random.default_rng(42)
    img_subexp = np.clip(img_bgr.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
    img_sobreexp = np.clip(img_bgr.astype(np.float32) * 1.7 + 40, 0, 255).astype(np.uint8)
    img_gauss = np.clip(img_bgr.astype(np.float32) + rng.normal(0, 25, img_bgr.shape), 0, 255).astype(np.uint8)
    img_sp = img_bgr.copy()
    r_sp = rng.random(img_bgr.shape[:2])
    img_sp[r_sp < 0.015] = 0
    img_sp[r_sp > 0.985] = 255
    img_blur = cv2.GaussianBlur(img_bgr, (15, 15), 0)
    img_critico = cv2.GaussianBlur(
        np.clip(img_subexp.astype(np.float32) + rng.normal(0, 22, img_bgr.shape), 0, 255).astype(np.uint8), (5, 5), 0
    )

    # 3. Defensa matemática real y medición real antes/después
    t_def0 = time.perf_counter()
    img_bilat = cv2.bilateralFilter(img_critico, d=7, sigmaColor=50, sigmaSpace=50)
    img_gamma = aplicar_correccion_gamma(img_bilat, gamma=0.65)
    img_clahe = aplicar_clahe_lab(img_gamma, clip_limit=3.0, grid_size=(8, 8))
    ms_defensa = (time.perf_counter() - t_def0) * 1000.0

    img_eval_ataque, motor_name, n_ataque, conf_ataque, _ = detectar_modelo_defensa(img_critico)
    img_eval_rescate, _, n_rescate, conf_rescate, ms_inf = detectar_modelo_defensa(img_clahe)

    v_orig = float(np.mean(hsv[:, :, 2]))
    v_crit = float(np.mean(cv2.cvtColor(img_critico, cv2.COLOR_BGR2HSV)[:, :, 2]))
    v_def = float(np.mean(cv2.cvtColor(img_clahe, cv2.COLOR_BGR2HSV)[:, :, 2]))

    d0 = dets_c[0] if dets_c else {"tamano": "Ninguno", "area": 0, "ratio": 0, "ar": 0, "circ": 0, "tiene_bozal": False, "score_bozal": 0, "lineas_hough": 0, "alerta": "SIN_DETECCION"}

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Visor Didáctico Real: Seguridad Canina (Tamaño + Bozal)</title>
  <style>
    :root {{ --bg:#0b0f19; --surface:#151d2c; --border:#26334a; --text:#c9d5e8; --bright:#fff; --accent:#38bdf8; --green:#4ade80; --red:#f87171; --orange:#fbbf24; --purple:#c084fc; }}
    * {{ box-sizing:border-box; margin:0; padding:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
    body {{ background:var(--bg); color:var(--text); padding:24px; line-height:1.5; }}
    header {{ background:linear-gradient(135deg,#1e293b,#0f172a); border:1px solid var(--border); padding:24px 30px; border-radius:14px; margin-bottom:24px; }}
    header h1 {{ color:var(--bright); font-size:1.75rem; margin-bottom:6px; }}
    .badge {{ display:inline-block; padding:4px 12px; border-radius:20px; font-size:.8rem; font-weight:600; background:rgba(56,189,248,.15); color:var(--accent); border:1px solid rgba(56,189,248,.3); }}
    .tabs {{ display:flex; gap:10px; margin-bottom:24px; flex-wrap:wrap; background:var(--surface); padding:8px; border-radius:12px; border:1px solid var(--border); }}
    .tab-btn {{ background:transparent; border:none; padding:10px 20px; color:var(--text); cursor:pointer; font-weight:600; border-radius:8px; }}
    .tab-btn.active {{ background:var(--accent); color:#0b0f19; }}
    .tab-content {{ display:none; }} .tab-content.active {{ display:block; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:20px; }}
    .card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; overflow:hidden; display:flex; flex-direction:column; }}
    .card-header {{ padding:12px 16px; font-weight:700; color:var(--bright); display:flex; justify-content:space-between; border-bottom:1px solid var(--border); }}
    .card-img-wrap {{ background:#000; aspect-ratio:1/1; display:flex; align-items:center; justify-content:center; }}
    .card-img-wrap img {{ width:100%; height:100%; object-fit:contain; cursor:zoom-in; }}
    .card-body {{ padding:14px 16px; font-size:.88rem; }}
    .math {{ background:rgba(0,0,0,.4); border-left:3px solid var(--accent); padding:6px 10px; font-family:Consolas,monospace; font-size:.82rem; margin:8px 0; color:#7dd3fc; }}
    .summary-box {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:20px; margin-bottom:22px; }}
    .modal {{ display:none; position:fixed; z-index:1000; inset:0; background:rgba(0,0,0,.9); align-items:center; justify-content:center; }}
    .modal.show {{ display:flex; }} .modal img {{ max-width:90%; max-height:90%; }}
  </style>
</head>
<body>
  <header>
    <h1>🛡️ Visor Real de Transformaciones — Seguridad Canina (Tamaño + Bozal)</h1>
    <p>Imagen procesada en tiempo real con OpenCV: <code>{os.path.basename(imagen_path)}</code> | Motor de evaluación: <strong>{motor_name}</strong></p>
  </header>

  <nav class="tabs">
    <button class="tab-btn active" onclick="switchTab(event,'t1')">1. Pipeline Clásico + Bozal</button>
    <button class="tab-btn" onclick="switchTab(event,'t2')">2. Perturbaciones Urbanas</button>
    <button class="tab-btn" onclick="switchTab(event,'t3')">3. Defensa Matemática (40%)</button>
  </nav>

  <section id="t1" class="tab-content active">
    <div class="summary-box">
      <strong>Métricas reales medidas en este frame:</strong> Tiempo clásico = <code>{ms_clasico:.2f} ms</code> |
      Tamaño = <strong>{d0['tamano']}</strong> ({d0['ratio']}% del frame) |
      Bozal (Canny+Hough) = <strong>{'DETECTADO' if d0['tiene_bozal'] else 'NO DETECTADO'}</strong> (score={d0['score_bozal']}, líneas Hough={d0['lineas_hough']}) |
      Alerta = <strong>{d0['alerta']}</strong>
    </div>
    <div class="grid">
      <div class="card"><div class="card-header"><span>1. Original RGB</span><span class="badge">V̄={v_orig:.1f}</span></div><div class="card-img-wrap"><img src="data:image/jpeg;base64,{img_a_base64(img_bgr)}" onclick="openM(this.src)"></div><div class="card-body">Entrada nativa RGB del dataset unificado.</div></div>
      <div class="card"><div class="card-header"><span>2. Canal H (Tono)</span><span class="badge">HSV</span></div><div class="card-img-wrap"><img src="data:image/jpeg;base64,{img_a_base64(h_col)}" onclick="openM(this.src)"></div><div class="card-body">Desacople cromático.<div class="math">H = arctan2(√3(G-B), 2R-G-B)</div></div></div>
      <div class="card"><div class="card-header"><span>3. Canal S (Saturación)</span><span class="badge">HSV</span></div><div class="card-img-wrap"><img src="data:image/jpeg;base64,{img_a_base64(s_col)}" onclick="openM(this.src)"></div><div class="card-body">Pureza de color para separar pelaje y fondo.</div></div>
      <div class="card"><div class="card-header"><span>4. Máscara Cruda</span><span class="badge">inRange</span></div><div class="card-img-wrap"><img src="data:image/jpeg;base64,{img_a_base64(cv2.cvtColor(mask_cruda, cv2.COLOR_GRAY2BGR))}" onclick="openM(this.src)"></div><div class="card-body">Umbralización HSV antes de limpieza.</div></div>
      <div class="card"><div class="card-header"><span>5. Mediana + Morfología</span><span class="badge">Apertura+Cierre</span></div><div class="card-img-wrap"><img src="data:image/jpeg;base64,{img_a_base64(cv2.cvtColor(mask_morfo, cv2.COLOR_GRAY2BGR))}" onclick="openM(this.src)"></div><div class="card-body">Supresión de ruido y sellado de silueta.</div></div>
      <div class="card"><div class="card-header"><span>6. Bordes Canny (ROI Bozal)</span><span class="badge">Alta Frecuencia</span></div><div class="card-img-wrap"><img src="data:image/jpeg;base64,{img_a_base64(edges_bgr)}" onclick="openM(this.src)"></div><div class="card-body">Detección de la trama/rejilla del bozal sobre la zona cefálica (Hough={d0['lineas_hough']} líneas).</div></div>
      <div class="card"><div class="card-header"><span>7. Resultado Clásico Real</span><span class="badge">{ms_clasico:.1f} ms</span></div><div class="card-img-wrap"><img src="data:image/jpeg;base64,{img_a_base64(img_cdet)}" onclick="openM(this.src)"></div><div class="card-body">Contorno, tamaño relativo y alerta de seguridad calculados en vivo.</div></div>
    </div>
  </section>

  <section id="t2" class="tab-content">
    <div class="grid">
      <div class="card"><div class="card-header"><span>1. Subexposición Nocturna</span></div><div class="card-img-wrap"><img src="data:image/jpeg;base64,{img_a_base64(img_subexp)}" onclick="openM(this.src)"></div><div class="card-body">I' = 0.35·I (alumbrado urbano deficiente).</div></div>
      <div class="card"><div class="card-header"><span>2. Deslumbramiento</span></div><div class="card-img-wrap"><img src="data:image/jpeg;base64,{img_a_base64(img_sobreexp)}" onclick="openM(this.src)"></div><div class="card-body">Saturación por farolas o sol directo.</div></div>
      <div class="card"><div class="card-header"><span>3. Ruido Térmico Gaussiano</span></div><div class="card-img-wrap"><img src="data:image/jpeg;base64,{img_a_base64(img_gauss)}" onclick="openM(this.src)"></div><div class="card-body">Sensor nocturno ISO alto (σ=25).</div></div>
      <div class="card"><div class="card-header"><span>4. Sal y Pimienta</span></div><div class="card-img-wrap"><img src="data:image/jpeg;base64,{img_a_base64(img_sp)}" onclick="openM(this.src)"></div><div class="card-body">Interferencia impulsiva (p=0.03).</div></div>
      <div class="card"><div class="card-header"><span>5. Desenfoque Óptico</span></div><div class="card-img-wrap"><img src="data:image/jpeg;base64,{img_a_base64(img_blur)}" onclick="openM(this.src)"></div><div class="card-body">Destruye las altas frecuencias de la rejilla del bozal.</div></div>
      <div class="card"><div class="card-header"><span>6. Entorno Crítico Mixto</span><span class="badge">V̄={v_crit:.1f}</span></div><div class="card-img-wrap"><img src="data:image/jpeg;base64,{img_a_base64(img_critico)}" onclick="openM(this.src)"></div><div class="card-body">Subexposición + ruido + desenfoque simultáneos.</div></div>
    </div>
  </section>

  <section id="t3" class="tab-content">
    <div class="summary-box">
      <strong>Medición real de la defensa ({motor_name}):</strong>
      Brillo medio V̄: Limpio=<code>{v_orig:.1f}</code> → Atacado=<code>{v_crit:.1f}</code> → Defendido=<code>{v_def:.1f}</code> |
      Detecciones: Atacado=<strong>{n_ataque}</strong> (score={conf_ataque:.2f}) → Defendido=<strong>{n_rescate}</strong> (score={conf_rescate:.2f}) |
      Costo del filtro: <code>{ms_defensa:.2f} ms</code>
    </div>
    <div class="grid">
      <div class="card"><div class="card-header"><span>1. Inferencia Bajo Ataque</span><span class="badge">Det={n_ataque}</span></div><div class="card-img-wrap"><img src="data:image/jpeg;base64,{img_a_base64(img_eval_ataque)}" onclick="openM(this.src)"></div><div class="card-body">Gradientes atenuados (||∇I'|| = α||∇I||).</div></div>
      <div class="card"><div class="card-header"><span>2. Filtro Bilateral</span></div><div class="card-img-wrap"><img src="data:image/jpeg;base64,{img_a_base64(img_bilat)}" onclick="openM(this.src)"></div><div class="card-body">Elimina ruido térmico preservando los bordes del bozal.</div></div>
      <div class="card"><div class="card-header"><span>3. Corrección Gamma (0.65)</span></div><div class="card-img-wrap"><img src="data:image/jpeg;base64,{img_a_base64(img_gamma)}" onclick="openM(this.src)"></div><div class="card-body">Expande las sombras no linealmente.</div></div>
      <div class="card"><div class="card-header"><span>4. CLAHE en Espacio Lab</span><span class="badge">V̄={v_def:.1f}</span></div><div class="card-img-wrap"><img src="data:image/jpeg;base64,{img_a_base64(img_clahe)}" onclick="openM(this.src)"></div><div class="card-body">Restaura contraste local solo en el canal L.</div></div>
      <div class="card"><div class="card-header"><span>5. Inferencia Tras Defensa</span><span class="badge">Det={n_rescate} ({ms_inf:.1f}ms)</span></div><div class="card-img-wrap"><img src="data:image/jpeg;base64,{img_a_base64(img_eval_rescate)}" onclick="openM(this.src)"></div><div class="card-body">Resultado real tras aplicar el pipeline defensivo.</div></div>
    </div>
  </section>

  <div id="m" class="modal" onclick="this.classList.remove('show')"><img id="mi"></div>
  <script>
    function switchTab(ev, id) {{
      document.querySelectorAll('.tab-content,.tab-btn').forEach(e => e.classList.remove('active'));
      document.getElementById(id).classList.add('active'); ev.target.classList.add('active');
    }}
    function openM(s) {{ document.getElementById('mi').src = s; document.getElementById('m').classList.add('show'); }}
  </script>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[EXITO] Visor real generado en: {os.path.abspath(output_html)}")
    return True


if __name__ == "__main__":
    generar_reporte_html(sys.argv[1] if len(sys.argv) > 1 else None)
