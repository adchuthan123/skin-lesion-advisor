import base64
import json
import os

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from transformers import pipeline

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────────
HF_MODEL_ID   = os.getenv('HF_MODEL_ID', 'YOUR_HF_USERNAME/vit-base-ham10000')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
LLM_MODEL      = os.getenv('LLM_MODEL', 'gpt-4o-mini')

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
DX_NAMES = {
    'nv':    'Melanocytic Nevus (gutartiger Leberfleck)',
    'mel':   'Melanoma',
    'bkl':   'Benigne Keratose',
    'bcc':   'Basalzellkarzinom',
    'akiec': 'Aktinische Keratose',
    'vasc':  'Vaskuläre Läsion',
    'df':    'Dermatofibrom'
}
RISK_MAP = {
    'mel': 'high', 'bcc': 'high',
    'akiec': 'medium',
    'bkl': 'low', 'nv': 'low', 'vasc': 'low', 'df': 'low'
}
CLIP_DESCRIPTIONS = [
    'actinic keratosis skin lesion',
    'basal cell carcinoma skin lesion',
    'benign keratosis skin lesion',
    'dermatofibroma skin lesion',
    'melanoma skin lesion',
    'melanocytic nevus mole',
    'vascular lesion skin'
]
CLIP_LABEL_MAP = dict(zip(CLIP_DESCRIPTIONS, CLASS_NAMES))

# ── Load Models ──────────────────────────────────────────────────────────────────
print('Lade ViT Modell...')
vit_classifier = pipeline('image-classification', model=HF_MODEL_ID)

print('Lade CLIP Modell...')
clip_detector = pipeline(
    model='openai/clip-vit-large-patch14',
    task='zero-shot-image-classification'
)

print('Lade ML Modell...')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
ml_model = joblib.load(os.path.join(MODELS_DIR, 'ml_risk_classifier.pkl'))
le_sex   = joblib.load(os.path.join(MODELS_DIR, 'le_sex.pkl'))
le_loc   = joblib.load(os.path.join(MODELS_DIR, 'le_loc.pkl'))
le_risk  = joblib.load(os.path.join(MODELS_DIR, 'le_risk.pkl'))

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
print('Alle Modelle geladen.')

# ── Helper Functions ──────────────────────────────────────────────────────────────
def encode_image(image: Image.Image) -> str:
    import io
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def parse_json_response(raw: str, required_keys: tuple) -> dict:
    cleaned = (raw or '').strip()
    if cleaned.startswith('```'):
        lines = cleaned.split('\n')
        inner = lines[1:-1] if lines[-1].strip() == '```' else lines[1:]
        cleaned = '\n'.join(inner).strip()
    parsed = json.loads(cleaned)
    missing = [k for k in required_keys if k not in parsed]
    if missing:
        raise ValueError(f'JSON fehlt Felder: {missing}')
    return parsed

# ── Classification Functions ──────────────────────────────────────────────────────
def classify_vit(image: Image.Image) -> dict:
    results = vit_classifier(image)
    return {r['label']: round(r['score'], 4) for r in results}

def classify_clip(image: Image.Image) -> dict:
    results = clip_detector(image, candidate_labels=CLIP_DESCRIPTIONS)
    return {CLIP_LABEL_MAP[r['label']]: round(r['score'], 4) for r in results}

def classify_openai(image: Image.Image) -> dict:
    if openai_client is None:
        return {'error': 'OPENAI_API_KEY fehlt. In .env eintragen.'}
    prompt = (
        f'Classify this skin lesion. Choose one label from: {CLASS_NAMES}. '
        'Return valid JSON: {"label": "...", "confidence": 0.0-1.0, "reasoning": "..."}'
    )
    b64 = encode_image(image)
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}}
            ]
        }],
        max_tokens=200
    )
    try:
        return parse_json_response(response.choices[0].message.content, ('label', 'confidence'))
    except Exception as e:
        return {'error': str(e), 'raw': response.choices[0].message.content}

def predict_ml_risk(age: int, sex: str, localization: str) -> str:
    known_locs = set(le_loc.classes_)
    loc_safe = localization if localization in known_locs else le_loc.classes_[0]

    # Iteration 2: engineered features (must match 02_ml_training.ipynb FEATURES_V2)
    age_group = 0 if age < 30 else (1 if age <= 60 else 2)
    is_face              = int(localization == 'face')
    is_trunk             = int(localization in ['back', 'abdomen', 'chest', 'trunk'])
    is_extremity         = int(localization in ['lower extremity', 'upper extremity', 'foot', 'hand'])
    is_high_risk_location = int(localization in ['back', 'lower extremity'])

    X = pd.DataFrame({
        'age':                  [age],
        'sex_enc':              le_sex.transform([sex]),
        'loc_enc':              le_loc.transform([loc_safe]),
        'age_group':            [age_group],
        'is_face':              [is_face],
        'is_trunk':             [is_trunk],
        'is_extremity':         [is_extremity],
        'is_high_risk_location':[is_high_risk_location]
    })
    return le_risk.inverse_transform(ml_model.predict(X))[0]

def generate_explanation(cv_label: str, cv_conf: float, risk: str, age: int, sex: str, loc: str) -> str:
    if openai_client is None:
        return f'Befund: {DX_NAMES.get(cv_label, cv_label)} | Risiko: {risk.upper()}\n\nHinweis: Diese Analyse ersetzt keine ärztliche Untersuchung.'

    action_map = {
        'high':   'Bitte suchen Sie umgehend einen Dermatologen auf.',
        'medium': 'Ein Dermatologenbesuch innerhalb der nächsten Wochen wird empfohlen.',
        'low':    'Beobachten Sie die Stelle und konsultieren Sie bei Veränderungen einen Arzt.'
    }
    tone_map = {
        'high': 'ernst und dringend, aber beruhigend',
        'medium': 'besorgt aber nicht alarmistisch',
        'low': 'beruhigend und positiv'
    }
    prompt = (
        f'Kontext: Hautläsions-Analyse App.\n'
        f'Befund: {DX_NAMES.get(cv_label, cv_label)} (Konfidenz: {int(cv_conf*100)}%)\n'
        f'Risiko: {risk.upper()} | Alter: {age} | Geschlecht: {sex} | Stelle: {loc}\n'
        f'Tonalität: {tone_map[risk]}\n'
        f'Empfehlung: {action_map[risk]}\n\n'
        f'Erkläre in max. 80 Wörtern auf Deutsch. '
        f'Endet mit: "Hinweis: Diese Analyse ersetzt keine ärztliche Untersuchung." '
        f'Antworte mit JSON: {{"answer": "..."}}'
    )
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {'role': 'system', 'content': 'Du bist ein KI-Assistent für Hautgesundheit.'},
            {'role': 'user', 'content': prompt}
        ],
        temperature=0.3, max_tokens=300
    )
    try:
        result = parse_json_response(response.choices[0].message.content, ('answer',))
        return result['answer']
    except Exception:
        return response.choices[0].message.content

# ── Main Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(image, age, sex, localization):
    if image is None:
        return {}, {}, {}, '', '', ''

    pil_img = Image.fromarray(image).convert('RGB')

    # 1. CV — alle 3 Modelle
    vit_output   = classify_vit(pil_img)
    clip_output  = classify_clip(pil_img)
    openai_output = classify_openai(pil_img)

    # 2. Bestes ViT-Label
    vit_label = max(vit_output, key=vit_output.get) if vit_output else 'nv'
    vit_conf  = vit_output.get(vit_label, 0.5)
    cv_risk   = RISK_MAP.get(vit_label, 'medium')

    # 3. ML — Metadaten-Risiko
    ml_risk = predict_ml_risk(age, sex, localization)

    # 4. Kombination
    order = {'low': 0, 'medium': 1, 'high': 2}
    final_risk = vit_label if cv_risk == ml_risk else max([cv_risk, ml_risk], key=lambda x: order.get(x, 0))
    final_risk = cv_risk if cv_risk == ml_risk else max([cv_risk, ml_risk], key=lambda x: order.get(x, 0))

    # 5. NLP Erklärung
    explanation = generate_explanation(vit_label, vit_conf, final_risk, age, sex, localization)

    risk_emoji = {'low': '🟢 Niedrig', 'medium': '🟡 Mittel', 'high': '🔴 Hoch'}
    summary = (
        f"**Bildanalyse:** {DX_NAMES.get(vit_label, vit_label)} ({vit_conf:.0%})\n\n"
        f"**ML-Risiko (Metadaten):** {risk_emoji.get(ml_risk, ml_risk)}\n\n"
        f"**Gesamtrisiko:** {risk_emoji.get(final_risk, final_risk)}"
    )

    return vit_output, clip_output, openai_output, summary, explanation

# ── Gradio UI ──────────────────────────────────────────────────────────────────────
with gr.Blocks(title='Skin Lesion Risk Advisor') as demo:
    gr.Markdown("""
    # 🔬 Skin Lesion Risk Advisor
    **ZHAW KI-Anwendungen Projekt** — Multimodales KI-System zur Risikoeinschätzung von Hautläsionen

    ⚠️ *Dieses Tool ist ein Forschungsprototyp und ersetzt keine ärztliche Untersuchung.*
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label='Bild der Hautläsion hochladen')
            age_input   = gr.Slider(10, 90, value=45, label='Alter', step=1)
            sex_input   = gr.Dropdown(['male', 'female'], value='male', label='Geschlecht')
            loc_input   = gr.Dropdown(
                ['back', 'lower extremity', 'trunk', 'upper extremity',
                 'abdomen', 'face', 'chest', 'foot', 'scalp', 'neck', 'hand', 'ear'],
                value='back', label='Körperstelle'
            )
            submit_btn = gr.Button('🔍 Analysieren', variant='primary')

        with gr.Column():
            summary_out     = gr.Markdown(label='Zusammenfassung')
            explanation_out = gr.Textbox(label='📝 Erklärung', lines=6)

    gr.Markdown('### Modellvergleich: ViT vs. CLIP vs. OpenAI Vision')
    with gr.Row():
        vit_out    = gr.Label(label='ViT (Fine-tuned)', num_top_classes=7)
        clip_out   = gr.Label(label='CLIP (Zero-Shot)', num_top_classes=7)
        openai_out = gr.JSON(label='OpenAI Vision')

    gr.Examples(
        examples=[
            ['../../archive/HAM10000_images_part_1/ISIC_0024306.jpg', 45, 'male', 'back'],
            ['../../archive/HAM10000_images_part_1/ISIC_0024307.jpg', 62, 'female', 'face'],
        ],
        inputs=[image_input, age_input, sex_input, loc_input]
    )

    submit_btn.click(
        fn=run_pipeline,
        inputs=[image_input, age_input, sex_input, loc_input],
        outputs=[vit_out, clip_out, openai_out, summary_out, explanation_out]
    )

    gr.Markdown('---\n*ZHAW KI-Anwendungen — Nur für Bildungszwecke | Dataset: HAM10000 (ISIC)*')

demo.launch()
