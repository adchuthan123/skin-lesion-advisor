# Skin Lesion Risk Advisor

A multimodal AI system that analyzes a skin lesion image and patient metadata to assess risk and generate a personalized explanation.

## Blocks Used
- **Computer Vision** — Fine-tuned EfficientNet-B0 on HAM10000 (7 lesion classes)
- **ML Numeric Data** — Risk classification from patient metadata (age, sex, localization)
- **NLP** — LLM-generated explanation of findings in plain language

## Pipeline

```
Skin Image → [CV: Lesion Type Classification]
                        ↓
             Patient Metadata (age, sex, localization)
                        ↓
             [ML: risk_level = low / medium / high]
                        ↓
             [NLP: LLM Explanation + Recommendation]
```

## Data Sources
| Source | Type | Used For |
|---|---|---|
| HAM10000 (ISIC Archive / Kaggle) | 10,015 images | CV Training |
| HAM10000_metadata.csv | Structured CSV | ML Numeric |

## Project Structure

```
skin-lesion-advisor/
├── app/                    # Streamlit App (inference only)
├── data/
│   └── processed/          # Cleaned data, splits
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_ml_training.ipynb
│   ├── 03_cv_training.ipynb
│   ├── 04_nlp_explanation.ipynb
│   └── 05_evaluation.ipynb
├── src/                    # Reusable modules
├── models/                 # Saved models
├── reports/figures/        # EDA plots
└── documentation/
```

## Data Setup

Place the HAM10000 archive folder at the same level as this project:
```
KI-Anwendungen/
├── archive/
│   ├── HAM10000_images_part_1/
│   ├── HAM10000_images_part_2/
│   └── HAM10000_metadata.csv
└── skin-lesion-advisor/   ← this project
```

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your API key:
```
OPENAI_API_KEY=your_key_here
```

## Run Notebooks (in order)
1. `notebooks/01_eda.ipynb` — Data exploration
2. `notebooks/02_ml_training.ipynb` — Train ML classifier
3. `notebooks/03_cv_training.ipynb` — Fine-tune CV model (run on Google Colab)
4. `notebooks/04_nlp_explanation.ipynb` — Prompt engineering & NLP evaluation
5. `notebooks/05_evaluation.ipynb` — Full end-to-end evaluation

## Run App

```bash
streamlit run app/app.py
```

## Deployment
Live demo: [HuggingFace Spaces URL — add after deployment]

## Submission
- Deadline: 07 June 2026, 18:00
- GitHub collaborators: jasminh, bkuehnis
