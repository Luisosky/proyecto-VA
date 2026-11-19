# 🛡️ Auditoría Híbrida de Modelos de Visión — Seguridad Ciudadana Canina (Tamaño + Bozal)

Proyecto final de **Visión Artificial** (Programa de Ingeniería de Sistemas y Computación) orientado al sector de **Seguridad Ciudadana**: detección de caninos en espacio público, estimación biométrica de tamaño y verificación de uso de **bozal**, enfrentando en paralelo **Procesamiento Clásico (OpenCV)** vs. **Aprendizaje Profundo (YOLOv8)** bajo condiciones estables y degradadas.

---

## 🎯 Enfoque Sociorregional (Seguridad Ciudadana)

En parques, zonas residenciales y vías peatonales, el control de caninos de porte grande o manejo especial exige verificar el porte de **bozal**. El sistema aplica la misma regla operativa sobre ambos enfoques:

| Tamaño Relativo del Canino | Uso de Bozal | Estado de Seguridad |
| :--- | :---: | :--- |
| **Grande** ($\ge 30\%$ del frame) | ❌ Sin bozal | 🔴 **ALERTA_SEGURIDAD** |
| **Grande** ($\ge 30\%$ del frame) | ✅ Con bozal | 🟢 **CUMPLE_NORMATIVA** |
| **Pequeño / Mediano** ($< 30\%$ del frame) | Cualquiera | ⚪ **RIESGO_BAJO** |

---

## 📁 Estructura del Repositorio

```text
proyecto-VA/
├── Proyecto.ipynb                        # Notebook principal (4 módulos, ROC real, negativos reales)
├── demo_en_vivo.py                       # Script en vivo (webcam/video/galería) con teclas de ataque y defensa
├── iniciar_demo_en_vivo.bat              # Lanzador rápido en Windows para la demo en vivo
├── generar_visor.py                      # Script Python (OpenCV/YOLOv8) que genera el visor HTML con datos reales
├── generar_visor.bat                     # Lanzador rápido en Windows para abrir el visor HTML
├── visor_transformaciones.html           # Dashboard autónomo de transformaciones paso a paso
├── requirements.txt                      # Dependencias Python reproducibles (.venv)
├── Proyecto Final Visión Artificial.pdf  # Guía oficial y rúbrica de evaluación
├── .gitignore                            # Exclusión de pesos (*.pt), runs/, cachés y datasets crudos
├── README.md                             # Documentación del proyecto
└── dataset_seguridad_canina/             # Dataset unificado, balanceado y deduplicado (773 imágenes únicas)
    ├── data.yaml                         # 2 clases limpias: ['dog', 'muzzle']
    ├── train/                            # 619 imágenes (304 sin bozal + 304 con bozal + 11 negativos)
    ├── valid/                            # 77 imágenes  (38 sin bozal  + 38 con bozal  + 1 negativo)
    └── test/                             # 77 imágenes  (38 sin bozal  + 38 con bozal  + 1 negativo)
```

---

## 📊 Dataset Unificado y Deduplicado (`dataset_seguridad_canina/`)

El dataset final se construyó combinando y depurando dos fuentes públicas (CC BY 4.0) para eliminar duplicados `.rf.`, fuga de datos (*data leakage*) y desbalance de clases:
1. **`dog.v1i.yolov8`** (Roboflow): 380 imágenes únicas de perros **sin bozal** en escenas cotidianas (`0: dog`).
2. **`dog muzzle.v2i.yolov8`** (Roboflow): 380 imágenes únicas con bozal etiquetado (`0: dog`, `1: muzzle`) + 13 imágenes negativas vacías (descartando las 18 cajas ruidosas de `No_muzzle`).

* **Total:** **773 imágenes únicas** (equilibrio exacto 50/50 entre perros sin bozal y con bozal en `train`, `valid` y `test`), listas para entrenar rápido en Google Colab con la aumentación dinámica nativa de YOLOv8.

---

## 🧩 Módulos Implementados en `Proyecto.ipynb`

1. **Módulo 1 — Procesamiento Clásico (20%):**
   * Segmentación HSV, filtro de mediana $5\times 5$, morfología (apertura + cierre), Área de Green y tamaño relativo unificado.
   * **Detector clásico de bozal:** bordes **Canny** + **Transformada de Hough Probabilística** sobre la región cefálica para detectar la rejilla/correas de alta frecuencia.
   * Storyboard paso a paso ($2\times 3$), benchmark real de latencia ($\mu \pm \sigma$, $P_{95}$, FPS), **FPR real** sobre negativos de fondo y Matriz de Confusión $3\times 3$ de tamaño.
2. **Módulo 2 — Aprendizaje Profundo (20%):**
   * Transfer Learning con **YOLOv8n** (`yolov8n.pt`) sobre 2 clases (`dog`, `muzzle`), parada temprana (`patience=15`), curvas reales desde `results.csv` (Box/Cls Loss, Precision, Recall, mAP@50, mAP@50-95) y asociación espacial perro $\leftrightarrow$ bozal por contención geométrica.
3. **Módulo 3 — Análisis de Robustez y Rigor Métrico (20%):**
   * Perturbaciones físicas de videovigilancia urbana (nocturna, deslumbramiento, ruido ISO alto, sal y pimienta, desenfoque óptico y entorno crítico mixto).
   * **4 Matrices de Confusión cruzadas reales** y **Curvas ROC 100% reales** para ambos enfoques (sin valores sintéticos), usando positivos y negativos reales de fondo + tabla comparativa generada desde las mediciones.
4. **Módulo 4 — Defensa ante Ataque Adversario (40%):**
   * Diagnóstico matemático en la teoría del píxel (colapso de $V/H$, atenuación de gradientes $\|\nabla I'\| = \alpha \|\nabla I\|$ en SiLU y pérdida de altas frecuencias de la rejilla del bozal).
   * Pipeline de rescate: **Filtro Bilateral** + **Corrección Gamma ($\gamma=0.65$)** + **CLAHE en CIE-Lab**, validado visual y cuantitativamente (F1 antes vs. después sobre el test degradado).

---

## 🚀 Cómo Ejecutar

### 1. Notebook en Google Colab (Recomendado para entrenar YOLOv8 en GPU)
1. Sube `dataset_seguridad_canina/` a Google Drive (`/content/drive/MyDrive/dataset_seguridad_canina`).
2. Abre [`Proyecto.ipynb`](Proyecto.ipynb) en **Google Colab** con GPU T4 y ejecuta las celdas en orden.

### 2. Demostración en Vivo para la Sustentación (`demo_en_vivo.py`)
```bash
pip install ultralytics opencv-python numpy
python demo_en_vivo.py --fuente test        # Recorre el test set (teclas n/p)
python demo_en_vivo.py --fuente 0           # Usa tu webcam en vivo
```
**Teclas en vivo:** `[1]` Limpio · `[2]` Baja luz · `[3]` Crítico (Luz+Ruido+Blur) · `[4]` Desenfoque · `[5]` Deslumbramiento · `[d]` Activar/Desactivar Defensa Matemática · `[m]` Clásico / YOLOv8 / Paralelo · `[q]` Salir.

### 3. Visor Didáctico HTML
* Doble clic en [`generar_visor.bat`](generar_visor.bat) o abre [`visor_transformaciones.html`](visor_transformaciones.html).
* Para regenerarlo con OpenCV/YOLOv8 sobre otra imagen:
  ```bash
  python generar_visor.py "ruta/a/tu_imagen.jpg"
  ```
