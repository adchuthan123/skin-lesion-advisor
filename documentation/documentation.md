# AI Applications Project Documentation Template

Use this template to document your project concisely and completely.
Fill in all required fields. Keep answers short and precise.

## Documentation Hint

When possible, reference the corresponding code location directly in your description.

### Example: Reference to a notebook section
> See *Data Preprocessing* in [`01_eda.ipynb`](../notebooks/01_eda.ipynb#data-loading)

### Example: Reference to Python code
> [`space/app.py`, lines 60–80](../space/app.py#L60-L80)

---

## Project Metadata

- **Project title:** Skin Lesion Risk Advisor — Multimodal AI for Dermoscopy Risk Assessment
- **Student:** Adchuthan Premananthan
- **GitHub repository URL:** https://github.com/adchuthan123/skin-lesion-advisor
- **Deployment URL:** https://huggingface.co/spaces/PREMAADC/skin-lesion-advisor
- **Submission date:** 07 June 2026

### Mandatory Setup Checks

- [x] At least 2 blocks selected
- [x] Multiple and different data sources used
- [x] Deployment URL provided
- [x] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

---

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [x] Computer Vision

Primary blocks used for core solution:
- **Primary block 1:** Computer Vision
- **Primary block 2:** ML Numeric Data

Third block (NLP) is implemented as extra work for bonus points.

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition

- **Problem statement:** Dermoscopy images of skin lesions are difficult to assess without specialist expertise. Early detection of melanoma and other malignant lesions significantly improves patient outcomes, yet access to dermatologists is limited.
- **Goal:** Build a multimodal AI system that classifies a skin lesion image into one of 7 diagnostic categories, combines this with patient metadata to estimate risk level (low / medium / high), and generates a plain-language explanation for the patient.
- **Success criteria:**
  - CV model: balanced accuracy > 0.65 on HAM10000 test set
  - ML model: balanced accuracy > 0.50 on metadata-only prediction
  - Combined pipeline: balanced accuracy > 0.68
  - NLP: coherent, risk-adapted German explanation with correct disclaimer
  - Working Gradio app deployed on HuggingFace Spaces

### 1.2 Integration Logic

- **How the selected blocks interact:**
  1. **CV** receives the skin lesion image → outputs lesion type + confidence
  2. **ML** receives patient metadata (age, sex, localization) → outputs risk level
  3. Both outputs are **combined** using a conservative rule (take higher risk when they disagree)
  4. **NLP** receives the combined result → generates patient-facing explanation in German

- **Data and output flow between blocks:**

```
Skin Image ──────────────► [CV: ViT fine-tuned on HAM10000]
                                     │
                                     ▼ lesion_type + confidence
Patient Metadata ────────► [ML: Random Forest / XGBoost]
(age, sex, localization)             │
                                     ▼ risk_level (low/medium/high)
                           [Combination Rule: max risk]
                                     │
                                     ▼ final_risk + findings
                           [NLP: GPT-4o-mini prompt]
                                     │
                                     ▼
                           Patient Explanation (German)
```

> See pipeline overview in [`space/app.py`, lines 80–110](../space/app.py#L80-L110)

---

## 2. Block Documentation

### 2A. ML Numeric Data

#### 2A.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | HAM10000_metadata.csv (ISIC Archive / Kaggle) | Structured CSV | 10,015 rows, 7 columns | Training data for risk classifier |

Columns used: `age` (numeric), `sex` (categorical), `localization` (categorical), `dx` (target label mapped to risk_level).

> See *Data Loading* in [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb#1-daten-laden)

#### 2A.2 Preprocessing and Features

- **Cleaning steps:**
  - Missing `age` values filled with median (57 missing of 10,015)
  - Rows with `sex = unknown` removed (45 rows)
  - Image paths resolved and stored for CV block
  - See [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb#6-daten-bereinigen-und-speichern)

- **Preprocessing steps:**
  - `sex` label-encoded (male=1, female=0)
  - `localization` label-encoded (14 unique values)
  - `dx` mapped to `risk_level`: mel/bcc → high, akiec → medium, nv/bkl/vasc/df → low
  - 80/20 stratified train/test split

- **Feature engineering and selection (Iteration 2):**
  - Iteration 1 (baseline): 3 features — `age`, `sex_enc`, `loc_enc`
  - Iteration 2 (engineered): 8 features — added `age_group` (bins: <30 / 30–60 / >60), `is_face`, `is_trunk`, `is_extremity`, `is_high_risk_location` (binary flags from EDA heatmap: back + lower extremity are most common melanoma sites)
  - Flags encode domain knowledge from EDA (Notebook 01) directly as features, analogous to LN1 apartment prediction where distance and amenity flags improved the model
  - See [`notebooks/02_ml_training.ipynb`](../notebooks/02_ml_training.ipynb#iteration-2-feature-engineering)

#### 2A.3 Model Selection

- **Models tested:** Random Forest (sklearn), XGBoost
- **Why these models were chosen:** Both are standard tree-based classifiers suitable for small tabular datasets with categorical and numeric features. They handle class imbalance well with `class_weight='balanced'` / sample weights. Same choice as in LN1 (apartment price prediction).

#### 2A.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline risk classification | 3 features (age, sex_enc, loc_enc), class_weight='balanced', n=200 | Random Forest, XGBoost | Balanced Acc: RF=0.635 / XGB=0.657 | — |
| 2 | Feature Engineering | +5 engineered features: age_group, is_face, is_trunk, is_extremity, is_high_risk_location | Random Forest, XGBoost | Balanced Acc: RF=0.627 / XGB=0.651 · F1 Macro: RF=0.475 / XGB=**0.483** | XGB F1 +0.006; Balanced Acc slight decrease — metadata ceiling effect |

> See full comparison in [`notebooks/02_ml_training.ipynb`](../notebooks/02_ml_training.ipynb#3-iterationsvergleich--visualisierung)

#### 2A.5 Evaluation and Error Analysis

- **Metrics used:** Balanced Accuracy, F1-Score (macro), Confusion Matrix
- **Final results (Iteration 2, 8 features):**
  - Random Forest: Balanced Accuracy 0.627, F1 Macro 0.475
  - XGBoost: Balanced Accuracy 0.651, F1 Macro 0.483 ← best model saved
  - Best model: XGBoost Iteration 2 → `models/ml_risk_classifier.pkl`
  - Feature Importance (top 3): `age` (0.420), `loc_enc` (0.190), `age_group` (0.101)
  - Critical errors (high → low): 74 of 326 high-risk cases — corrected by CV block in pipeline
- **Error patterns and likely causes:**
  - Class `high` (melanoma, BCC) frequently misclassified — metadata alone is insufficient to distinguish malignant from benign lesions
  - This is **expected and intentional**: it motivates the CV block as the primary diagnostic input
  - See [`notebooks/02_ml_training.ipynb`](../notebooks/02_ml_training.ipynb#7-fehleranalyse)

#### 2A.6 Integration with Other Block(s)

- **Inputs received from other block(s):** None (metadata is independent of CV output at training time)
- **Outputs provided to other block(s):**
  - `risk_level` (low/medium/high) → combined with CV-derived risk in [`space/app.py`](../space/app.py#L105-L110)
  - `risk_level` → passed as input to NLP block for explanation generation

---

### 2B. NLP

#### 2B.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | GPT-4o-mini (OpenAI API) | LLM API | — | Generate patient explanations in German |
| 2 | HAM10000_metadata.csv | Structured CSV | 10,015 rows | Patient context (age, sex, localization) injected into prompts |

#### 2B.2 Preprocessing and Prompt Design

- **Text preprocessing:** No raw text corpus — input is structured data (CV label, confidence, risk level, patient metadata) assembled dynamically into a prompt string.
- **Prompt design:**
  - System prompt: defines assistant role as skin health advisor (not a doctor)
  - User prompt: injects CV prediction, confidence, risk level, patient data, desired tone, and standard recommendation
  - Output constrained to JSON `{"answer": "..."}` for reliable parsing
  - Ends with mandatory disclaimer: *"Hinweis: Diese Analyse ersetzt keine ärztliche Untersuchung."*
  - See [`notebooks/04_nlp_explanation.ipynb`](../notebooks/04_nlp_explanation.ipynb#6-finale-prompt-funktion-fur-die-app)

#### 2B.3 Approach Selection

- **Approach used:** Prompt engineering with GPT-4o-mini (OpenAI API)
- **Alternatives considered:**
  - Classical NLP (e.g., rule-based text generation) — rejected, too rigid for personalized explanations
  - RAG — rejected, no document corpus needed; all context comes from structured model outputs
  - Prompt engineering — chosen for simplicity, flexibility, and high output quality

#### 2B.4 Comparison and Iterations

| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline explanation | Simple 2-sentence prompt, no structure | GPT-4o-mini | Verständlichkeit: 3/5, Disclaimer: 2/5 | — |
| 2 | Structured output | Added 4 required sections (finding, meaning, recommendation, disclaimer), JSON output | GPT-4o-mini | Vollständigkeit: 5/5, Disclaimer: 5/5 | +clarity |
| 3 | Risk-adaptive tone | Tone instruction varies by risk level (urgent/concerned/reassuring), action_map per risk | GPT-4o-mini | Risikoanpassung: 5/5, overall score: 5/5 | +personalisation |

> See full comparison in [`notebooks/04_nlp_explanation.ipynb`](../notebooks/04_nlp_explanation.ipynb#5-qualitativer-vergleich-der-strategien)

#### 2B.5 Evaluation and Error Analysis

- **Evaluation strategy:** Qualitative scoring on 5 criteria (Verständlichkeit, Vollständigkeit, Risiko-Anpassung, Disclaimer, Präzision) across 3 test scenarios (high/medium/low risk), scored 1–5.
- **Results:**
  - Strategy A (simple): avg 2.4/5
  - Strategy B (structured): avg 4.2/5
  - Strategy C (adaptive): avg **4.8/5** → selected for app
- **Error patterns and likely causes:**
  - Occasionally too verbose (>100 words) at temperature 0.3 — solved by explicit word limit
  - JSON parsing fails if LLM adds markdown fences — handled by `parse_json_response()` in [`space/app.py`](../space/app.py#L55-L65)

#### 2B.6 Integration with Other Block(s)

- **Inputs received from other block(s):**
  - `cv_prediction` + `cv_confidence` from CV block
  - `risk_level` from ML block (and combined risk from pipeline)
  - `age`, `sex`, `localization` from user input
- **Outputs provided to other block(s):** Final German explanation displayed to user — end of pipeline.

---

### 2C. Computer Vision

#### 2C.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | HAM10000 Part 1 (ISIC Archive / Kaggle) | JPEG images | 5,000 images | Training & evaluation |
| 2 | HAM10000 Part 2 (ISIC Archive / Kaggle) | JPEG images | 5,015 images | Training & evaluation |
| 3 | HAM10000_metadata.csv | CSV | 10,015 rows | Labels (dx column) for supervised training |

7 classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`

> See *Klassenverteilung* in [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb#2-klassenverteilung-dx)

#### 2C.2 Preprocessing and Augmentation

- **Image preprocessing:**
  - Resize to 224×224 pixels (ViT requirement)
  - Normalize with ImageNet mean/std ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  - Applied via `AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')`
  - See [`notebooks/03_cv_training.ipynb`](../notebooks/03_cv_training.ipynb#2-preprocessing-autoimageprocessor)

- **Augmentation strategy:**
  - No explicit augmentation applied during training (ViT Trainer handles this internally)
  - Class imbalance addressed via `WeightedRandomSampler` in training loader

#### 2C.3 Model Selection

- **Vision models used:**
  1. `google/vit-base-patch16-224` (fine-tuned) — primary model
  2. `openai/clip-vit-base-patch32` (zero-shot) — comparison
  3. `gpt-4o-mini` with vision (closed-source) — comparison

- **Why these models were chosen:**
  - ViT: same architecture used in course exercises (LN2, Week 6/8) — known workflow
  - CLIP: enables zero-shot comparison without training — shows value of fine-tuning
  - OpenAI Vision: closed-source baseline — same comparison pattern as Week 7 (Oxford Pets app)

#### 2C.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Model(s) used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Transfer learning baseline | Freeze all layers, train classifier head only (5 epochs) | ViT-Base | Balanced Acc: ~0.55 | — |
| 2 | Full fine-tuning | Unfreeze all layers, lr=3e-4, 5 more epochs | ViT-Base | Balanced Acc: ~0.72 (full test set, NB03 Colab — outputs not saved) | +0.17 |
| 3 | Model comparison | No training — compare with CLIP & OpenAI Vision on 200-image sample (NB05) | ViT vs CLIP vs OpenAI | Balanced Acc: ViT 0.549 (Acc 0.735), CLIP ~0.35, OpenAI ~0.48 | Context only |

> See full comparison in [`notebooks/03_cv_training.ipynb`](../notebooks/03_cv_training.ipynb#7-vergleichs-tabelle)

#### 2C.5 Evaluation and Error Analysis

- **Metrics and/or visual checks:**
  - Balanced Accuracy, F1-Score (macro), Confusion Matrix (normalized)
  - Visual: sample predictions with correct/incorrect labels
  - See [`notebooks/03_cv_training.ipynb`](../notebooks/03_cv_training.ipynb#5-evaluation--unser-vit-modell)

- **Final results (200-image test sample, NB05):**
  - ViT (fine-tuned): Accuracy 0.735, Balanced Accuracy 0.549, F1 Macro 0.590 ← deployed model (`PREMAADC/vit-base-ham10000`)
  - CLIP (zero-shot): Balanced Accuracy ≈ 0.35 (no fine-tuning, text-image matching only)
  - OpenAI Vision: Balanced Accuracy ≈ 0.48 (qualitative sample only — not evaluated on full dataset to avoid API costs)
  - Combined pipeline: Balanced Accuracy 0.756, F1 Macro 0.597
  - See full evaluation in [`notebooks/05_evaluation.ipynb`](../notebooks/05_evaluation.ipynb)

- **Error patterns and limitations:**
  - Classes `df` (dermatofibroma) and `vasc` (vascular lesions) hardest to classify — smallest classes in dataset
  - `nv` (dominant class, 67%) still occasionally confused with `mel` — critical false negative
  - Model trained on HAM10000 only — may not generalise to smartphone images or other populations

#### 2C.6 Integration with Other Block(s)

- **Inputs received from other block(s):** Raw skin lesion image from user (no preprocessing by other blocks)
- **Outputs provided to other block(s):**
  - `lesion_type` (string, one of 7 classes) → used by NLP block for explanation
  - `cv_risk` derived from lesion_type → combined with ML `risk_level` in [`space/app.py`](../space/app.py#L105-L110)
  - `confidence` score → shown to user and passed to NLP prompt

---

## 3. Deployment

- **Deployment URL:** https://huggingface.co/spaces/PREMAADC/skin-lesion-advisor
- **Main user flow:**
  1. User uploads dermoscopy image
  2. User enters patient data (age, sex, body localization)
  3. Clicks "Analysieren"
  4. App shows: ViT / CLIP / OpenAI predictions side by side + combined risk level + German explanation
- **Screenshot or short demo:** App live at https://huggingface.co/spaces/PREMAADC/skin-lesion-advisor — accepts dermoscopy image + patient metadata, returns ViT/CLIP/OpenAI predictions, combined risk level, and German explanation.

> App entry point: [`space/app.py`](../space/app.py)

---

## 4. Execution Instructions

- **Environment setup:**
```bash
pip install -r requirements.txt
cp .env.example .env
# Add your OpenAI API key to .env:
# OPENAI_API_KEY=your_key_here
# HF_MODEL_ID=your_hf_username/vit-base-ham10000
```

- **Data setup:**
```
Place HAM10000 archive at:
KI-Anwendungen/
├── archive/
│   ├── HAM10000_images_part_1/   (5,000 images)
│   ├── HAM10000_images_part_2/   (5,015 images)
│   └── HAM10000_metadata.csv
└── skin-lesion-advisor/          (this project)
```

- **Training command(s):**
```bash
# Run notebooks in order (03 requires GPU — use Google Colab):
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_ml_training.ipynb
# Upload 03 to Google Colab for GPU training:
jupyter notebook notebooks/03_cv_training.ipynb
jupyter notebook notebooks/04_nlp_explanation.ipynb
jupyter notebook notebooks/05_evaluation.ipynb
```

- **Inference/run command(s):**
```bash
# Local app (after models are trained):
python space/app.py
# Or via gradio:
gradio space/app.py
```

- **Reproducibility notes:**
  - All notebooks use `random_state=42`
  - Python 3.12, package versions pinned in `requirements.txt`
  - CV model pushed to HuggingFace Hub — reproducible via model ID
  - ML model saved as `space/models/ml_risk_classifier.pkl`

---

## 5. Optional Bonus Evidence

- [x] **Third selected block implemented with strong quality** — NLP block fully implemented with 3 prompt strategies compared, qualitative evaluation, JSON parsing, risk-adaptive tone
- [x] **More than two data sources used with clear added value** — HAM10000 images (CV), HAM10000 metadata CSV (ML), OpenAI API (NLP)
- [x] **Extended evaluation** — 3-way model comparison (ViT vs CLIP vs OpenAI) in CV block; combined pipeline evaluated in Notebook 05
- [x] **Ethics, bias, or fairness analysis** — Model trained on HAM10000 which has demographic imbalance (predominantly light-skinned patients). This is noted as a limitation in Notebooks 03 and 05. All outputs include a mandatory medical disclaimer.

Evidence for selected bonus items:

**Third block (NLP):**
> See prompt comparison table in [`notebooks/04_nlp_explanation.ipynb`](../notebooks/04_nlp_explanation.ipynb#5-qualitativer-vergleich-der-strategien) — 3 strategies evaluated on 5 criteria across 3 risk scenarios.

**Extended CV evaluation:**
> See 3-model comparison in [`notebooks/03_cv_training.ipynb`](../notebooks/03_cv_training.ipynb#7-vergleichs-tabelle) and [`reports/figures/cv_02_model_comparison.png`](../reports/figures/cv_02_model_comparison.png)

**Ethics/Bias:**
> HAM10000 demographics are skewed toward lighter skin tones (Fitzpatrick types I–III). The model may underperform on darker skin. This is documented in [`notebooks/05_evaluation.ipynb`](../notebooks/05_evaluation.ipynb#9-zusammenfassung--interpretation) under "Limitierungen".
