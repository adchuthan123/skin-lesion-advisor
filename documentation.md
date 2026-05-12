# AI Applications Project Documentation

## Project Metadata

- Project title: Skin Lesion Diagnosis Advisor
- Student: Preadchuthan Adsayakulchai
- GitHub repository URL: *(nach Deployment eintragen)*
- Deployment URL: *(nach Deployment eintragen)*
- Submission date: *(Datum eintragen)*

### Mandatory Setup Checks

- [x] At least 2 blocks selected
- [x] Multiple and different data sources used
- [ ] Deployment URL provided *(nach Deployment)*
- [ ] Required GitHub users added to repository (`jasminh`, `bkuehnis`) *(nach Repo-Erstellung)*

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [x] Computer Vision

Primary blocks used for core solution (choose 2):
- Primary block 1: Computer Vision (Lesion-Klassifikation aus Bild)
- Primary block 2: ML Numeric Data (Risiko-Score aus Metadaten)

NLP ist als dritten Block implementiert und wird separat als Extraleistung bewertet.

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition
- Problem statement: Hautläsionen sind schwer zu klassifizieren — selbst Dermatologen benötigen Erfahrung. Patienten haben keinen einfachen Zugang zu einer ersten Einschätzung.
- Goal: Ein multimodales System das aus einem Hautbild + Patientenmetadaten (Alter, Geschlecht, Körperstelle) eine Diagnose-Wahrscheinlichkeit und eine verständliche Erklärung liefert.
- Success criteria: CV-Modell ≥ 75% Balanced Accuracy auf Testset; ML-Modell verbessert CV-Vorhersage messbar; NLP produziert klinisch sinnvolle Erklärungen in Patientensprache.

### 1.2 Integration Logic
- How the selected blocks interact:
  1. **CV** analysiert das hochgeladene Hautbild → liefert Top-3 Diagnosen mit Wahrscheinlichkeiten
  2. **ML** verarbeitet Metadaten (Alter, Geschlecht, Körperstelle) → liefert Risiko-Score und unterstützt die Diagnose
  3. **NLP** empfängt CV-Diagnose + ML-Risiko → generiert eine patientenfreundliche Erklärung
- Data and output flow between blocks:

```
[Bild] ──────► CV (EfficientNet) ──► Top-3 Klassen + Wahrscheinlichkeiten ──┐
                                                                               ▼
[Metadaten] ► ML (XGBoost) ──────► Risiko-Score + Klassen-Prediction ──────► NLP (LLM Prompt)
                                                                               │
                                                                               ▼
                                                              Patientenbericht (Text)
```

> See *Integration Pipeline* in [`notebooks/05_evaluation.ipynb`](notebooks/05_evaluation.ipynb)

---

## 2. Block Documentation

### 2A. ML Numeric Data

#### 2A.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | HAM10000 – `HAM10000_metadata.csv` (Kaggle) | CSV | 10'015 Zeilen, 7 Spalten | Trainings- und Testdaten |

Features verwendet: `age`, `sex`, `localization`  
Target: `dx` (7 Klassen: mel, nv, bkl, bcc, akiec, vasc, df)

> See *Data Loading* in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb#data-loading)

#### 2A.2 Preprocessing and Features
- Cleaning steps: Fehlende Werte in `age` mit Median imputed; `sex` und `localization` One-Hot-Encoded
- Preprocessing steps: Label Encoding für Target; Train/Test Split 80/20 stratifiziert nach Klasse
- Feature engineering and selection: 3 Features direkt aus Metadaten; kein zusätzliches Feature Engineering nötig

> See *Preprocessing* in [`notebooks/02_ml_training.ipynb`](notebooks/02_ml_training.ipynb#preprocessing)

#### 2A.3 Model Selection
- Models tested: Random Forest, XGBoost
- Why these models were chosen: Beide robust bei kleinen Feature-Sets und Klassen-Ungleichgewicht; XGBoost bietet native class_weight Unterstützung; Random Forest als Baseline gut interpretierbar

#### 2A.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline | Standardparameter, keine Gewichtung | Random Forest | Balanced Accuracy: *(nach Training)* | – |
| 2 | Klassen-Ungleichgewicht beheben | `class_weight='balanced'` / `scale_pos_weight` | Random Forest, XGBoost | Balanced Accuracy: *(nach Training)* | *(nach Training)* |
| 3 | Hyperparameter-Tuning | GridSearchCV auf n_estimators, max_depth | XGBoost | Balanced Accuracy: *(nach Training)* | *(nach Training)* |

> See *Model Training* in [`notebooks/02_ml_training.ipynb`](notebooks/02_ml_training.ipynb#model-training)

#### 2A.5 Evaluation and Error Analysis
- Metrics used: Balanced Accuracy, F1-Score per Klasse, Confusion Matrix
- Final results: *(nach Training eintragen)*
- Error patterns and likely causes: Erwartet: Seltene Klassen (vasc, df) schwerer zu klassifizieren wegen wenigen Samples

> See *Evaluation* in [`notebooks/02_ml_training.ipynb`](notebooks/02_ml_training.ipynb#evaluation)

#### 2A.6 Integration with Other Block(s)
- Inputs received from other block(s): Keine — ML arbeitet unabhängig auf Metadaten
- Outputs provided to other block(s): Risiko-Score + Klassen-Prediction → NLP Block (Prompt-Kontext)

---

### 2B. NLP

#### 2B.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | CV-Output (Top-3 Diagnosen + Wahrscheinlichkeiten) | Strukturierter Text | 1 Record pro Anfrage | Input für Prompt |
| 2 | ML-Output (Risiko-Score, Klassen-Prediction) | Strukturierter Text | 1 Record pro Anfrage | Input für Prompt |

#### 2B.2 Preprocessing and Prompt Design
- Text preprocessing: CV- und ML-Outputs werden zu einem strukturierten Kontext-String formatiert
- Prompt design: 3 Prompt-Strategien verglichen:
  1. **Zero-Shot**: Direkte Aufgabenbeschreibung ohne Beispiele
  2. **Few-Shot**: 2 Beispiel-Erklärungen im Prompt
  3. **Chain-of-Thought**: Schritt-für-Schritt Reasoning angewiesen

> See *Prompt Design* in [`notebooks/04_nlp_explanation.ipynb`](notebooks/04_nlp_explanation.ipynb#prompt-design)

#### 2B.3 Approach Selection
- Approach used: Prompt Engineering mit OpenAI GPT-4o-mini (kein eigenes Training)
- Alternatives considered: RAG mit medizinischem Corpus (verworfen wegen Komplexität); fine-tuning (verworfen wegen Datenmangel)

#### 2B.4 Comparison and Iterations

| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline | Zero-Shot Prompt | GPT-4o-mini | Klinische Korrektheit (manuell) | – |
| 2 | Verbesserung Verständlichkeit | Few-Shot mit Patientenbeispielen | GPT-4o-mini | Lesbarkeit + Korrektheit | *(nach Evaluation)* |
| 3 | Reasoning-Qualität | Chain-of-Thought | GPT-4o-mini | Vollständigkeit des Berichts | *(nach Evaluation)* |

> See *Prompt Comparison* in [`notebooks/04_nlp_explanation.ipynb`](notebooks/04_nlp_explanation.ipynb#prompt-comparison)

#### 2B.5 Evaluation and Error Analysis
- Evaluation strategy: Manuelle Bewertung von 10 Beispielen pro Strategie (Korrektheit, Verständlichkeit, Vollständigkeit 1-5)
- Results: *(nach Evaluation eintragen)*
- Error patterns and likely causes: Erwartet: Zero-Shot zu technisch; Few-Shot am ausgewogensten

#### 2B.6 Integration with Other Block(s)
- Inputs received from other block(s): CV Top-3 Diagnosen; ML Risiko-Score
- Outputs provided to other block(s): Fertiger Patientenbericht → Streamlit UI

---

### 2C. Computer Vision

#### 2C.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | HAM10000 Part 1 (Kaggle) | JPEG Bilder | 5'000 Bilder | Training / Validierung |
| 2 | HAM10000 Part 2 (Kaggle) | JPEG Bilder | 5'015 Bilder | Training / Validierung |
| 3 | HAM10000 Metadata CSV | CSV | 10'015 Zeilen | Labels für Training |

#### 2C.2 Preprocessing and Augmentation
- Image preprocessing: Resize auf 224×224; Normalisierung mit ImageNet-Mittelwert/Std; Train/Val Split 80/20 stratifiziert
- Augmentation strategy: RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(20°), ColorJitter — nur auf Trainingsset

> See *Preprocessing* in [`notebooks/03_cv_training.ipynb`](notebooks/03_cv_training.ipynb#preprocessing)

#### 2C.3 Model Selection
- Vision model(s) used: EfficientNet-B0 (ImageNet pretrained, fine-tuned)
- Why these model(s) were chosen: EfficientNet bietet gutes Accuracy/Parameter Verhältnis; B0 läuft auf Google Colab Free Tier; weit verbreitet in medizinischen CV-Aufgaben

#### 2C.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Model(s) used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline | Nur Classifier-Layer trainiert (frozen backbone) | EfficientNet-B0 | Balanced Accuracy: *(nach Training)* | – |
| 2 | Fine-tuning | Letzte 2 EfficientNet-Blöcke aufgetaut, LR 1e-4 | EfficientNet-B0 | Balanced Accuracy: *(nach Training)* | *(nach Training)* |
| 3 | Klassen-Gewichtung | Weighted CrossEntropyLoss proportional zu Klassen-Ungleichgewicht | EfficientNet-B0 | Balanced Accuracy: *(nach Training)* | *(nach Training)* |

> See *Training Loop* in [`notebooks/03_cv_training.ipynb`](notebooks/03_cv_training.ipynb#training-loop)

#### 2C.5 Evaluation and Error Analysis
- Metrics and/or visual checks: Balanced Accuracy, F1 per Klasse, Confusion Matrix, Grad-CAM Visualisierung
- Final results: *(nach Training eintragen)*
- Error patterns and limitations: Erwartet: Melanom vs. Nevi schwer zu trennen (beide ähnlich pigmentiert); seltene Klassen (vasc, df) schwächer

> See *Evaluation* in [`notebooks/03_cv_training.ipynb`](notebooks/03_cv_training.ipynb#evaluation)

#### 2C.6 Integration with Other Block(s)
- Inputs received from other block(s): Keine — CV arbeitet direkt auf rohem Bild
- Outputs provided to other block(s): Top-3 Diagnosen + Wahrscheinlichkeiten → NLP Block; Hauptdiagnose → ML zur Kombination

---

## 3. Deployment

- Deployment URL: *(nach Deployment eintragen — Streamlit Community Cloud)*
- Main user flow:
  1. Benutzer lädt Hautbild hoch
  2. Benutzer gibt Alter, Geschlecht, Körperstelle ein
  3. CV analysiert Bild → Top-3 Diagnosen
  4. ML berechnet Risiko-Score aus Metadaten
  5. NLP generiert Patientenbericht
  6. Alle Ergebnisse werden übersichtlich angezeigt
- Screenshot or short demo: *(nach Deployment Screenshot einfügen)*

> See [`app/app.py`](app/app.py) für komplette Streamlit-Implementierung

---

## 4. Execution Instructions

- Environment setup:
  ```bash
  pip install -r requirements.txt
  cp .env.example .env
  # OpenAI API Key in .env eintragen: OPENAI_API_KEY=sk-...
  ```

- Data setup:
  ```
  1. HAM10000 von Kaggle herunterladen:
     https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection
  2. Entpacken nach:
     data/raw/HAM10000_images_part_1/
     data/raw/HAM10000_images_part_2/
     data/raw/HAM10000_metadata.csv
  ```

- Training command(s):
  ```bash
  # Notebooks in dieser Reihenfolge in Jupyter ausführen:
  jupyter notebook notebooks/01_eda.ipynb
  jupyter notebook notebooks/02_ml_training.ipynb
  # Für CV: Google Colab empfohlen (GPU erforderlich)
  jupyter notebook notebooks/03_cv_training.ipynb
  jupyter notebook notebooks/04_nlp_explanation.ipynb
  jupyter notebook notebooks/05_evaluation.ipynb
  ```

- Inference/run command(s):
  ```bash
  streamlit run app/app.py
  ```

- Reproducibility notes: Python 3.10+; alle Zufallsseed auf 42 gesetzt; `requirements.txt` enthält genaue Versionen

---

## 5. Optional Bonus Evidence

- [x] Third selected block implemented with strong quality (NLP als dritter Block mit 3 Prompt-Strategien verglichen)
- [ ] More than two data sources used with clear added value
- [ ] A core section is done exceptionally well
- [ ] Extended evaluation
- [ ] Ethics, bias, or fairness analysis
- [ ] Creative or exceptional use case

Evidence for selected bonus items:

**NLP als dritter Block:**
3 Prompt-Strategien (Zero-Shot, Few-Shot, Chain-of-Thought) systematisch verglichen. Der beste Ansatz wird in der finalen App eingesetzt.
> See [`notebooks/04_nlp_explanation.ipynb`](notebooks/04_nlp_explanation.ipynb)
