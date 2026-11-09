# 🌱 AgroScan AI: Precision Crop Pathology & Climate-Fused Agronomic Intelligence Suite

[![Live Web Application](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-00FFA3?style=for-the-badge&logo=streamlit)](https://agroscan-precision-ai.streamlit.app)
[![Model Architecture](https://img.shields.io/badge/Model-MobileNetV2%20Transfer%20Learning-blue?style=for-the-badge&logo=tensorflow)](https://github.com/skyl1-create/agroscan-ai)
[![Dataset](https://img.shields.io/badge/Dataset-PlantVillage%20(38%20Classes)-green?style=for-the-badge)](https://www.kaggle.com/datasets/emmarex/plantdisease)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

An end-to-end, deep learning-powered agricultural diagnosis, explainable computer vision (XAI), and micro-climate epidemiological intelligence platform built for **CSE276: Applied Deep Learning Project**.

---

## 🔗 Live Deployment & Resources
- **Permanent 24/7 Cloud URL:** [https://agroscan-precision-ai.streamlit.app](https://agroscan-precision-ai.streamlit.app)
- **Official Open-Source Repository:** [https://github.com/skyl1-create/agroscan-ai](https://github.com/skyl1-create/agroscan-ai)
- **Dataset Download Link:** [PlantVillage Benchmark Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) (54,305 images across 38 crop disease categories).

---

## 🌟 Key Highlights & Engineering Contributions

1. **Lightweight Deep Neural Network (MobileNetV2)**:
   - Optimized for mobile and low-latency edge deployment (~2.58 million parameters).
   - Trained on 38 crop disease categories spanning 14 host crops (Apple, Corn, Grape, Potato, Tomato, Pepper, Cherry, etc.).
   - **Validation Accuracy: 89.21%** (Top-1) | **98.29%** (Top-3).

2. **Multi-Modal Micro-Climate Epidemiological Fusion**:
   - Integrates live meteorological telemetry (Ambient Temperature, Relative Humidity, Wind Speed, Precipitation) via free Open-Meteo REST API.
   - Dynamically calculates a **Climatic Outbreak Risk Index (%)** based on species-specific pathogen sporulation requirements.
   - Computes an **Estimated Infection Onset Timeline** by fusing lesion tissue necrosis percentages with incubation biology.

3. **Explainable AI (XAI) & Lesion Segmentation**:
   - **Convolutional Saliency Heatmaps:** Maps backpropagated gradients to the original leaf to visually highlight disease foci.
   - **HSV Foliar Damage Segmentation:** Quantifies precise infected surface area percentages and clinical infection stage.

4. **Continual Learning & Model Self-Improvement**:
   - Integrated active feedback loop where agronomists and farmers can confirm or correct diagnoses.
   - Includes one-click incremental fine-tuning on newly acquired field specimens with conservative learning rates ($10^{-5}$) to prevent catastrophic forgetting.

5. **Practical Field Agronomy Suite**:
   - **Spray & Tank Mixing Calculator:** Computes exact water volumes, chemical active ingredient weights, and knapsack refills calibrated for plot size.
   - **Live Weather Advisory:** Evaluates real-time wind drift and rainfall wash-off risk to determine the optimal spray window.
   - **Clinical Prescription Download:** One-click generation of certified diagnostic `.txt` reports.

---

## 📁 Repository & Codebase Structure

```text
cse276project/
├── app.py                      # Production Streamlit Cloud & Local Web Application
├── requirements.txt            # Python project dependencies
├── README.md                   # Complete system documentation and running instructions
├── .gitignore                  # Git exclusions (datasets, virtual environments, binaries)
│
├── src/                        # Core Python Engineering Modules
│   ├── __init__.py
│   ├── config.py               # Paths, hyperparameters, class labels, directory setup
│   ├── model.py                # MobileNetV2 architecture with local pre-trained weights
│   ├── dataset.py              # tf.data preprocessing, data augmentation, batch pipeline
│   ├── train.py                # Training pipeline with early stopping & checkpoints
│   ├── evaluate.py             # Generates confusion matrix and classification metrics
│   ├── ai_engine.py            # Foliar quality gate, TTA, Saliency Heatmaps, HSV damage segmentation
│   ├── epidemiology.py         # Open-Meteo geocoding, weather fetching, epidemiological risk fusion
│   ├── continual_learning.py   # Feedback collection, registry storage, incremental fine-tuning
│   └── remedies.py             # Database query interface for 38-class treatment protocols
│
├── models/                     # Trained Deep Learning Artifacts
│   ├── crop_disease_mobilenetv2.keras   # Saved production weights
│   ├── class_indices.json               # Integer-to-class mapping dictionary
│   └── mobilenet_v2_weights_...no_top.h5 # Base ImageNet weights for offline loading
│
├── data/
│   ├── samples/                # Benchmark field test leaves for instant demonstration
│   └── feedback_store/         # Active learning verified images and metadata registry
│
└── reports/                    # Empirical Evaluation Figures
    ├── training_curves.png     # Loss and accuracy vs. training epochs
    ├── confusion_matrix.png    # 38x38 normalized confusion matrix
    └── classification_report.csv # Precision, recall, and F1-score breakdown
```

---

## 🚀 Setup & Execution Guide (Step-by-Step)

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your system.
- Git installed.

### 1. Clone the Repository
```bash
git clone https://github.com/skyl1-create/agroscan-ai.git
cd agroscan-ai
```

### 2. Create and Activate Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
*The web interface will automatically open in your default browser at `http://localhost:8501`.*

---

## 🔬 Model Training & Evaluation (Optional)

If you wish to train the model from scratch on custom data or re-evaluate:

1. **Download Dataset:** Download the [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) and extract folders into `data/dataset/`.
2. **Execute Training:**
   ```bash
   python src/train.py --epochs 15 --batch-size 32 --lr 0.0001
   ```
3. **Compute Metrics & Heatmap:**
   ```bash
   python src/evaluate.py
   ```
   *Generated evaluation reports will be saved directly into `reports/`.*

---

## 📊 Summary of Model Evaluation Benchmarks

| Metric | Measured Value |
| :--- | :--- |
| **Model Backbone** | MobileNetV2 (Pre-trained on ImageNet) |
| **Input Resolution** | $224 \times 224 \times 3$ RGB |
| **Total Parameter Count** | 2,586,854 (~2.58 Million) |
| **Top-1 Validation Accuracy** | **89.21%** |
| **Top-3 Clinical Accuracy** | **98.29%** |
| **Mean Precision** | **0.89** |
| **Mean Recall** | **0.89** |
| **Mean F1-Score** | **0.89** |
| **Average Inference Latency** | **35 ms** on standard CPU |

---

## 📜 Citation & Consulted Works
- **PlantVillage Dataset:** Hughes, D., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060.
- **MobileNetV2:** Sandler, M., et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018.
- **Open-Meteo Weather API:** Free Open-Source Weather & Geocoding API ([open-meteo.com](https://open-meteo.com/)).
