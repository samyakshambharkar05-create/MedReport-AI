"""
Medical Report Analyzer -- Streamlit App
OCR -> Hybrid (regex + biomedical NER) Extraction -> Plain-language explanation -> Trend tracking

Run with: streamlit run app.py
"""

import streamlit as st
import re
import json
import os
import io
import base64
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors as rl_colors
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="MedReport AI", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

HISTORY_FILE = "patient_history.json"

ACCENT = "#5B7FE0"
ACCENT_PINK = "#E07FA8"
SIDEBAR_BG = "#1B1A23"
PAGE_BG = "#FDF3F8"
CARD_BG = "#FFFFFF"

STATUS_HEX = {"LOW": "#3A5FA8", "HIGH": "#B8568A", "NORMAL": "#4C7A6E", "UNKNOWN": "#7A7685"}
STATUS_BG = {"LOW": "#E7EDFB", "HIGH": "#FBE7F1", "NORMAL": "#E6F2EE", "UNKNOWN": "#EFEDF3"}
STATUS_DOT = {"LOW": "#6C8FE0", "HIGH": "#E0729F", "NORMAL": "#6FA893", "UNKNOWN": "#A39FB0"}
PIE_PALETTE = ["#6C8FE0", "#E0729F", "#F2A6C6", "#8FB3EA", "#C97FB0", "#5B7FE0",
               "#F0C2D8", "#A7C0F0", "#D88BAE", "#7FA0E8", "#EFA8C8"]

KNOWN_PARAMETERS_SORTED = sorted([
    # --- CBC / Hematology ---
    "White Blood Cell", "WBC", "TLC", "Total Leucocyte Count",
    "Red Blood Cell", "RBC", "R B C",
    "Hemoglobin", "Haemoglobin", "Hgb", "HB",
    "Hematocrit", "Haematocrit", "HCT", "PCV",
    "Mean Cell Volume", "Mean Corp Volume", "Mean Volume", "MCV", "M C V",
    "Mean Cell Hemoglobin", "Mean Corp Hb", "Mean Hb", "MCH", "M C H",
    "Mean Cell Hb Conc", "Mean Corp Hb Conc", "MCHC", "M C H C",
    "Red Cell Dist Width", "RDW",
    "Platelet count", "Platelet Count", "Platelet",
    "Mean Platelet Volume", "MPV",
    "Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil",
    # --- Renal Profile ---
    "Blood Urea", "Urea",
    "Serum Creatinine", "Creatinine",
    "Serum Sodium", "Sodium",
    "Serum Potassium", "S. Potassium", "Potassium",
    # --- Liver Profile ---
    "Serum Alkaline Phosphatase", "S. Alkaline Phosphatase", "Alkaline Phosphatase",
    "SGOT", "AST", "SGOT/AST",
    "SGPT", "ALT", "SGPT/ALT", "SGPT IALT",
    "Serum Bilirubin", "Bilirubin",
], key=len, reverse=True)

FRAGMENT_ALIASES = {"##hils": "Neutrophil", "l": "Lymphocyte", "e": "Eosinophil", "mono": "Monocyte"}

EXPLANATIONS = {
    "Hemoglobin": "Think of hemoglobin as the delivery truck for oxygen in your blood — it picks up oxygen from your lungs and drops it off everywhere your body needs it. If it's low, your body isn't getting enough oxygen, which is why you feel tired or weak.",
    "RBC": "Red blood cells are tiny disc-shaped cells that carry hemoglobin around your body. Think of them as the trucks themselves — the more you have, the more oxygen can be delivered.",
    "WBC": "White blood cells are your body's soldiers — they fight off bacteria, viruses, and infections. A high count usually means your body is currently fighting something; a very low count means your defences are down.",
    "TLC": "This is just a count of all the different types of white blood cells (your infection-fighting soldiers) added together. It gives doctors a quick overview of how your immune system is doing.",
    "MCH": "This tells you how much oxygen-carrying protein (hemoglobin) is packed into each individual red blood cell. Think of it as checking whether each delivery truck is fully loaded or running light.",
    "MCHC": "Similar to MCH, but this measures the concentration — how dense the hemoglobin is inside each red blood cell. Like checking how tightly packed the cargo is.",
    "RDW": "This checks whether your red blood cells are all roughly the same size or very uneven. A high value means they're quite varied in size, which can point to nutritional deficiencies.",
    "Platelet count": "Platelets are the tiny cells that form clots when you bleed — like the repair crew that patches up leaks. Too few means you might bleed more than normal; too many can increase clotting risk.",
    "Neutrophil": "Neutrophils are the first responders of your immune system — they rush to the site of bacterial infections and destroy the bacteria. Think of them as your body's emergency response team.",
    "Lymphocyte": "Lymphocytes are the specialists of your immune system — they remember past infections and create antibodies. They're especially active against viral infections.",
    "Eosinophil": "Eosinophils deal mainly with allergies and parasites. If this is high, your body may be reacting to an allergen (like pollen or dust) or fighting a parasitic infection.",
    "Monocyte": "Monocytes are the cleanup crew — they eat up dead cells, bacteria, and debris after an infection is fought off. They're also involved in long-term immune responses.",
    "Hematocrit": "This is simply the percentage of your blood that is made up of red blood cells (the rest is liquid plasma). Low means not enough red cells; high means the blood is too thick.",
    "Mean Cell Volume": "This measures the average size of your red blood cells. Too small often points to iron deficiency; too large often points to vitamin B12 or folate deficiency.",
    "Mean Platelet Volume": "This measures the average size of your platelets. Larger platelets are generally more active and work better at clotting.",
    "Blood Urea": "Urea is a waste product your body makes when it breaks down protein — your kidneys filter it out of your blood. High levels usually mean the kidneys aren't filtering as well as they should.",
    "Creatinine": "Creatinine is another waste product (from muscle activity) that your kidneys filter out. It's one of the most reliable ways to check kidney health — high levels are a red flag.",
    "Sodium": "Sodium controls the balance of water in and around your cells — think of it as regulating how swollen or shrivelled your cells are. It also plays a role in nerve and muscle signals.",
    "Potassium": "Potassium is critical for your heart to beat properly and for your muscles to work. Even small imbalances can affect your heartbeat, which is why doctors watch it closely.",
    "Alkaline Phosphatase": "This enzyme is found mainly in your liver and bones. High levels can indicate liver disease, bile duct problems, or bone disorders — your doctor will look at other tests together with this one.",
    "SGOT/AST": "This is a liver enzyme that leaks into the blood when liver cells are damaged. Think of it as an alarm signal — the higher it is, the more stress the liver is under.",
    "SGPT/ALT": "This is the most specific liver enzyme test — it's found almost exclusively in the liver, so a high value is a strong indicator of liver damage or inflammation.",
    "Bilirubin": "Bilirubin is a yellow substance made when old red blood cells are broken down. The liver processes and removes it. High levels cause the skin and eyes to turn yellow (jaundice) — a sign the liver may not be working properly.",
    "Mean Cell Hemoglobin": "This tells you the average weight of hemoglobin (the oxygen-carrying protein) in each red blood cell. Low values often suggest iron deficiency.",
}

NAME_CANONICALIZE = {
    "Haemoglobin": "Hemoglobin",
    "TLC": "WBC", "Total Leucocyte Count": "WBC",
    "Red Blood Cell": "RBC", "R B C": "RBC",
    "Mean Corp Volume": "Mean Cell Volume", "Mean Volume": "Mean Cell Volume", "M C V": "Mean Cell Volume",
    "Mean Corp Hb": "Mean Cell Hemoglobin", "Mean Hb": "Mean Cell Hemoglobin",
    "M C H": "Mean Cell Hemoglobin",
    "Mean Corp Hb Conc": "Mean Cell Hb Conc", "M C H C": "Mean Cell Hb Conc",
    "Haematocrit": "Hematocrit", "PCV": "Hematocrit",
    "Platelet Count": "Platelet count",
    "Urea": "Blood Urea",
    "Serum Creatinine": "Creatinine",
    "Serum Sodium": "Sodium",
    "Serum Potassium": "Potassium", "S. Potassium": "Potassium",
    "Serum Alkaline Phosphatase": "Alkaline Phosphatase",
    "S. Alkaline Phosphatase": "Alkaline Phosphatase",
    "SGOT": "SGOT/AST", "AST": "SGOT/AST",
    "SGPT": "SGPT/ALT", "ALT": "SGPT/ALT", "SGPT IALT": "SGPT/ALT",
    "Serum Bilirubin": "Bilirubin",
}


# ---------------------------------------------------------------------------
# DESIGN SYSTEM
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; font-size: 15px; }}
    .stApp {{ background-color: {PAGE_BG}; }}

    h1, h2, h3 {{ font-family: 'Source Serif 4', serif !important; color: #2B2640 !important; font-weight: 600 !important; }}
    h1 {{ letter-spacing: -0.5px; font-size: 30px !important; }}
    h4 {{ font-family: 'Inter', sans-serif !important; font-weight: 600 !important; color: #2B2640 !important; font-size: 17px !important; }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background-color: {SIDEBAR_BG};
        border-right: none;
    }}
    section[data-testid="stSidebar"] * {{ color: #D6D3E0; }}
    section[data-testid="stSidebar"] h1 {{ color: #FFFFFF !important; font-size: 18px !important; }}
    .sidebar-logo-text {{ font-size: 21px !important; font-weight: 700 !important; color: #FFFFFF !important; letter-spacing: -0.3px; }}

    /* Style the radio nav as zoom-on-hover buttons */
    section[data-testid="stSidebar"] div[role="radiogroup"] {{
        display: flex; flex-direction: column; width: 100%;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
        display: flex; align-items: center; width: 100%; box-sizing: border-box;
        background: #262335; border-radius: 10px; padding: 12px 14px; margin-bottom: 10px;
        border: 1px solid transparent;
        transition: transform 0.18s ease, background 0.18s ease, border-color 0.18s ease;
        cursor: pointer;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
        transform: scale(1.045);
        background: #2E2A42;
        border-color: {ACCENT_PINK};
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {{
        display: none;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label p {{
        font-size: 14px !important; font-weight: 500; color: #EDEBF5 !important;
    }}

    /* ---------- Cards & containers ---------- */
    .panel-box {{
        background: {CARD_BG}; border-radius: 16px; padding: 1.4rem 1.5rem;
        border: 1px solid #F0DCE8; box-shadow: 0 1px 3px rgba(120,80,110,0.06);
    }}

    .metric-card {{
        background: {CARD_BG}; border-radius: 14px; padding: 16px 18px;
        border: 1px solid #F0DCE8; border-bottom: 3px solid var(--accent, #888);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .metric-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 14px rgba(120,80,110,0.10); }}
    .metric-label {{ font-size: 12.5px; color: #8A8398; }}
    .metric-value {{ font-size: 28px; font-weight: 600; margin-top: 2px; color: #2B2640; }}

    .upload-heading {{
        font-size: 13px; font-weight: 600; letter-spacing: 0.04em; color: #B0568A;
        text-transform: uppercase; margin-bottom: 0.4rem;
    }}

    .detail-card {{
        background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_PINK} 100%);
        border-radius: 16px; padding: 1.3rem 1.4rem; color: white;
        height: 460px; box-sizing: border-box;
        display: flex; flex-direction: column; justify-content: center; gap: 8px;
    }}
    .detail-pill {{
        font-size: 11px; font-weight: 600; padding: 3px 11px; border-radius: 999px;
        background: rgba(255,255,255,0.25); letter-spacing: 0.03em;
    }}

    .result-row {{
        display: flex; align-items: center; gap: 12px; padding: 13px 18px; border-radius: 12px;
        background: {CARD_BG}; border: 1px solid #F0DCE8; margin-bottom: 9px;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }}
    .result-row:hover {{ transform: translateX(3px); box-shadow: 0 2px 10px rgba(120,80,110,0.08); }}
    .result-status-pill {{ font-size: 11.5px; font-weight: 600; padding: 4px 12px; border-radius: 999px; }}

    .stButton button {{
        border-radius: 10px; font-size: 13.5px; font-weight: 500;
        transition: transform 0.15s ease;
        border: 1px solid #E8D5E0;
    }}
    .stButton button:hover {{ transform: translateY(-1px); border-color: {ACCENT_PINK}; }}
    .stButton button[kind="primary"] {{
        background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_PINK} 100%); border: none; color: white;
    }}

    .step-row {{ display: flex; align-items: center; gap: 8px; font-size: 13px; color: #8A8398; padding: 4px 0; }}
    .step-row.done {{ color: #4C7A6E; }}
    .step-row.active {{ color: {ACCENT_PINK}; font-weight: 500; }}

    [data-testid="stFileUploaderDropzone"] {{
        background: #FFFCFD !important; border: 1.5px dashed #E8B8D0 !important; border-radius: 14px !important;
        transition: border-color 0.2s ease, background 0.2s ease;
    }}
    [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: {ACCENT_PINK} !important; background: #FFF6FA !important;
    }}

    .panel-box {{ transition: box-shadow 0.18s ease; }}
    .panel-box:hover {{ box-shadow: 0 4px 16px rgba(120,80,110,0.10); }}

    /* Native Streamlit bordered containers -- used instead of markdown div-wrapping
       (which doesn't actually enclose chart elements and was causing stray empty boxes) */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 16px !important;
        border-color: #F0DCE8 !important;
        background: {CARD_BG} !important;
        transition: box-shadow 0.18s ease;
    }}
    [data-testid="stVerticalBlockBorderWrapper"]:hover {{
        box-shadow: 0 4px 16px rgba(120,80,110,0.10);
    }}

    .stButton button:active {{ transform: translateY(0) scale(0.97); }}
    .stButton button[kind="secondary"] {{
        background: #FFFFFF; color: #2B2640; font-size: 12px; padding: 6px 10px;
    }}
    .stButton button[kind="primary"] {{
        font-size: 12px; padding: 6px 10px;
    }}

    [data-testid="stSelectbox"] > div > div {{
        border-radius: 10px !important; border-color: #E8D5E0 !important;
        transition: border-color 0.18s ease;
    }}
    [data-testid="stSelectbox"] > div > div:hover {{ border-color: {ACCENT_PINK} !important; }}

    [data-testid="stExpander"] {{
        border-radius: 12px !important; border-color: #F0DCE8 !important;
        background: {CARD_BG} !important;
    }}
    [data-testid="stExpander"] summary {{
        color: #2B2640 !important;
    }}
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p {{
        color: #2B2640 !important;
    }}

    /* Tighten default Streamlit block spacing so nothing feels like leftover dead space */
    .block-container {{ padding-top: 2.2rem; }}
    div[data-testid="stVerticalBlock"] > div {{ gap: 0.5rem; }}
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CACHED MODEL LOADERS
# ---------------------------------------------------------------------------
@st.cache_resource
def load_ocr_reader():
    try:
        import easyocr
        return easyocr.Reader(['en'], gpu=False)
    except Exception:
        return None  # EasyOCR unavailable (memory limit etc.) -- Gemini handles extraction


@st.cache_resource
def load_ner_pipeline():
    try:
        from transformers import pipeline
        return pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")
    except Exception:
        return None  # NER unavailable -- row-based extraction still works without it


# ---------------------------------------------------------------------------
# PIPELINE FUNCTIONS
# ---------------------------------------------------------------------------
def pdf_to_images(pdf_path):
    from pdf2image import convert_from_path
    poppler_path = os.environ.get("POPPLER_PATH", None)
    if poppler_path:
        return convert_from_path(pdf_path, dpi=200, poppler_path=poppler_path)
    return convert_from_path(pdf_path, dpi=200)


def run_ocr(reader, image_path):
    """
    Returns (full_text, average_confidence, detections).
    If reader is None (EasyOCR unavailable), returns empty results so the
    Gemini Vision fallback can handle extraction instead.
    """
    if reader is None:
        return "", 0.0, []

    results = reader.readtext(image_path)
    full_text = ""
    confidences = []
    detections = []
    for (bbox, text, confidence) in results:
        if confidence >= 0.4:
            full_text += text + " "
            confidences.append(confidence)
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            detections.append({
                "text": text, "confidence": confidence,
                "center_y": sum(ys) / len(ys), "left_x": min(xs), "height": max(ys) - min(ys),
            })
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    return full_text.strip(), avg_conf, detections


def group_into_rows(detections):
    """
    Clusters OCR detections into visual rows based on vertical (Y) position, then
    sorts each row left-to-right by X position.

    Key design choices:
    - Tolerance = height * 0.4 (tighter than the usual 0.5-0.7) because this report
      has closely-spaced rows that were merging together at higher tolerance values,
      causing values from one row to bleed into the adjacent row's label.
    - We compare each new detection against the FIRST detection in the current row
      (not the last), because as rows accumulate left-to-right, the last detection's
      Y may have drifted slightly, causing cascading row-merges.
    """
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d["center_y"])
    rows = []
    current_row = [sorted_dets[0]]
    row_anchor_y = sorted_dets[0]["center_y"]
    row_tolerance = sorted_dets[0]["height"] * 0.4

    for det in sorted_dets[1:]:
        if abs(det["center_y"] - row_anchor_y) <= row_tolerance:
            current_row.append(det)
        else:
            rows.append(sorted(current_row, key=lambda d: d["left_x"]))
            current_row = [det]
            row_anchor_y = det["center_y"]
            row_tolerance = det["height"] * 0.4
    rows.append(sorted(current_row, key=lambda d: d["left_x"]))
    return rows


def clean_parameter_name(raw_guess):
    """
    Matches OCR text against known parameter names, with two normalizations applied
    before matching to handle common OCR misread patterns:

    1. Collapse spaces between single letters: 'R B C' → 'RBC', 'M C H C' → 'MCHC'
       This handles the common OCR pattern where abbreviations get spaces inserted
       between each letter.

    2. Replace capital I with slash: 'SGPT IALT' → 'SGPT/ALT', 'UIL' → 'U/L'
       OCR frequently misreads '/' as 'I' in certain fonts.
    """
    # Normalize: collapse spaced single-letter abbreviations ('R B C' → 'RBC')
    normalized = re.sub(r'\b([A-Za-z])\s(?=[A-Za-z]\b)', r'\1', raw_guess)
    # Normalize: replace ' I' (space + capital I) with '/' for slash misreads
    normalized = re.sub(r'\s[I]\s', '/', normalized)

    for known_name in KNOWN_PARAMETERS_SORTED:
        if known_name.lower() in normalized.lower():
            return NAME_CANONICALIZE.get(known_name, known_name)
        # Also try original raw_guess in case normalization went wrong
        if known_name.lower() in raw_guess.lower():
            return NAME_CANONICALIZE.get(known_name, known_name)

    alias = FRAGMENT_ALIASES.get(raw_guess.strip().lower())
    return NAME_CANONICALIZE.get(alias, alias)


UNIT_PATTERN = r'(g\s?[/l]\s?d[lL]|mg\s?/\s?dL|mill\s?/\s?cumm|millions\s?[/I]\s?cu\s?mm|[/I]\s?cu\s?mm|K\s?/\s?mcL|M\s?/\s?mcL|fL|fl|pg|cumm|/\s?uL|GM\s?%|mmol\s?/\s?L|U\s?[/I]\s?L|mg\s?%|[xX]\s?10[\d³\']?|%)'
NUMBER_PATTERN = r'\d+\.?\d*'


def extract_by_row(rows):
    """
    Position-aware row-based extraction.

    Key insight: in column-style lab reports (Label | Value | Unit | Range),
    the label is always in the LEFTMOST column. So instead of searching the
    whole row text for a known parameter name (which picks up stray text),
    we:
    1. Look for the parameter name in the LEFT portion of the row only
       (detections whose left_x is less than 40% of the row's width)
    2. Grab numbers ONLY from detections to the RIGHT of the label detection
       (detections whose left_x is greater than the label's right edge)

    This prevents range numbers from a higher row contaminating a lower row's
    label, and prevents value/unit columns from being misread as parameter names.
    """
    if not rows:
        return []

    # Estimate page width from the rightmost detection across all rows
    all_xs = [d["left_x"] for row in rows for d in row]
    page_right = max(all_xs) if all_xs else 1000
    label_zone_cutoff = page_right * 0.40  # left 40% = label column

    records = []
    for row in rows:
        if not row:
            continue

        # Step 1: find the label — search only left-zone detections
        label_dets = [d for d in row if d["left_x"] <= label_zone_cutoff]
        label_text = " ".join(d["text"] for d in label_dets)
        matched_name = clean_parameter_name(label_text)

        if not matched_name:
            # Fallback: try the full row text in case label spans into mid-column
            full_row_text = " ".join(d["text"] for d in row)
            matched_name = clean_parameter_name(full_row_text)
            if not matched_name:
                continue

        # Step 2: find where the label ends (rightmost X of label detections)
        label_right = max(d["left_x"] for d in (label_dets if label_dets else row))

        # Step 3: grab numbers and unit ONLY from detections to the right of the label
        value_dets = [d for d in row if d["left_x"] > label_right]
        if not value_dets:
            continue
        value_text = " ".join(d["text"] for d in value_dets)

        # Handle comma-separated thousands (8,000 → 8000) and split decimals (0. 8 → 0.8)
        value_text_clean = clean_ocr_numbers(value_text)

        numbers = re.findall(NUMBER_PATTERN, value_text_clean)
        if not numbers:
            continue

        unit_match = re.search(UNIT_PATTERN, value_text, re.IGNORECASE)
        unit = unit_match.group(1).strip() if unit_match else "NOT_FOUND"

        records.append({"parameter": matched_name, "numbers_found": numbers[:3], "unit": unit})
    return records


def extract_with_hybrid(text, ner_results):
    """Text-order fallback extraction (used for reports where rows aren't cleanly row-separated)."""
    records = []
    procedures = [r for r in ner_results if r['entity_group'] == 'Diagnostic_procedure']
    number_pattern = r'\d+\.?\d*'

    for i, proc in enumerate(procedures):
        name = proc['word']
        start = proc['end']
        end = procedures[i + 1]['start'] if i + 1 < len(procedures) else len(text)
        chunk = text[start:end]

        numbers = re.findall(number_pattern, chunk)
        unit_match = re.search(UNIT_PATTERN, chunk, re.IGNORECASE)
        unit = unit_match.group(1) if unit_match else "NOT_FOUND"

        matched_name = clean_parameter_name(name)
        if matched_name and numbers:
            records.append({"parameter": matched_name, "numbers_found": numbers[:3], "unit": unit})
    return records


def merge_records(row_records, text_order_records):
    """
    Combines both extraction strategies: row-based results are trusted first (since
    they're geometrically grounded and immune to text-order bugs); text-order results
    fill in any parameter the row-based pass missed entirely (e.g. a label/value pair
    OCR happened to read on a single line already).
    """
    merged = {}
    for r in row_records:
        merged[r["parameter"]] = r
    for r in text_order_records:
        if r["parameter"] not in merged:
            merged[r["parameter"]] = r
    return list(merged.values())


def flag_value(value, low, high):
    try:
        v, lo, hi = float(value), float(low), float(high)
    except (ValueError, TypeError):
        return "UNKNOWN"
    if v < lo:
        return "LOW"
    if v > hi:
        return "HIGH"
    return "NORMAL"


def clean_ocr_numbers(text):
    """
    Fixes three distinct OCR misread patterns seen in real Indian lab reports:

    1. SPLIT DECIMAL TOKEN ('0. 8' → '0.8'): OCR reads dot but adds space after it.
    2. COMMA AS DECIMAL ('3,6' → '3.6') vs THOUSANDS ('8,000' → '8000'):
       If digits after comma are 1-2 chars → decimal. If 3+ chars → thousands separator.
    3. SPACE AS DECIMAL ('90 0' → '90.0', '0 8' → '0.8'):
       OCR drops the decimal point entirely, leaving a space.
       Rule: single digit space single digit → decimal (catches '0 8', '3 6' etc.)
       Multi-digit space single digit → decimal (catches '90 0', '30 9' etc.)
       Multi-digit space multi-digit → leave alone (keeps range pairs like '40 80').
    """
    # Fix 1: split dot token ('0. 8' → '0.8')
    text = re.sub(r'(\d+)\.\s+(\d+)', r'\1.\2', text)

    # Fix 2: comma as decimal vs thousands
    def fix_comma(m):
        before, after = m.group(1), m.group(2)
        return f"{before}.{after}" if len(after) <= 2 else f"{before}{after}"
    text = re.sub(r'(\d+),(\d+)', fix_comma, text)

    # Fix 3: space as decimal — single digit on either side
    # '\d \d(?!\d)' catches '0 8', '3 6', '90 0', '30 9' without touching '40 80'
    text = re.sub(r'(\d)\s(\d)(?!\d)', r'\1.\2', text)

    return text


def explain_record(r):
    nums = r['numbers_found']
    val = nums[0]
    low = nums[1] if len(nums) > 1 else None
    high = nums[2] if len(nums) > 2 else None

    status = flag_value(val, low, high)
    description = EXPLANATIONS.get(r['parameter'], "No description available for this test.")

    if low and high:
        range_str = f"{low}-{high}"
    elif low:
        range_str = f">{low}"
    else:
        range_str = "Not found in report"
        status = "UNKNOWN"

    return {
        "parameter": r['parameter'],
        "value": val,
        "unit": r['unit'],
        "range": range_str,
        "status": status,
        "description": description,
    }


def extract_with_gemini(image_path, api_key):
    """
    Sends the lab report image to Gemini Vision API and asks it to extract
    all lab values as structured JSON. Used as a fallback when the local
    OCR+NER pipeline extracts fewer values than expected or has low confidence.

    Returns a list of explained records in the same format as explain_record(),
    or None if the API call fails.
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=st.secrets["api_key"])

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        image_part = {
            "mime_type": "image/png",
            "data": base64.b64encode(image_bytes).decode("utf-8")
        }

        prompt = """You are a medical lab report analyzer. Look at this lab report image carefully.

Extract ALL test parameters you can see — including CBC, Renal Profile, Liver Profile, or any other panels present.

Return ONLY a valid JSON array, no other text, no markdown, no explanation.
Each item must have exactly these fields:
- "parameter": the test name (e.g. "Hemoglobin", "WBC", "Creatinine")
- "value": the numeric result as a string (e.g. "10.8", "8000")
- "unit": the unit (e.g. "g/dL", "/cu mm", "mg%") or "NOT_FOUND" if not visible
- "range": the reference range as "low-high" (e.g. "12.0-16.0") or "Not found in report" if not visible
- "status": "LOW", "HIGH", "NORMAL", or "UNKNOWN" based on comparing value to range
- "description": one plain-English sentence explaining what this test measures

Example output format:
[
  {
    "parameter": "Hemoglobin",
    "value": "10.8",
    "unit": "g/dL",
    "range": "12.0-16.0",
    "status": "LOW",
    "description": "Hemoglobin carries oxygen in your blood."
  }
]

Extract every single test you can see. Do not skip any."""

        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content([
            {"role": "user", "parts": [
                {"text": prompt},
                {"inline_data": image_part}
            ]}
        ])

        raw = response.text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
        raw = raw.strip()

        records = json.loads(raw)

        # Validate and normalize each record
        cleaned = []
        for r in records:
            if not isinstance(r, dict):
                continue
            cleaned.append({
                "parameter": str(r.get("parameter", "Unknown")),
                "value":     str(r.get("value", "N/A")),
                "unit":      str(r.get("unit", "NOT_FOUND")),
                "range":     str(r.get("range", "Not found in report")),
                "status":    str(r.get("status", "UNKNOWN")).upper(),
                "description": str(r.get("description", "No description available.")),
            })
        return cleaned

    except Exception as e:
        st.session_state["gemini_last_error"] = str(e)
        return None  # Caller checks st.session_state['gemini_last_error'] if needed


def normalize_parameter_name(name):
    """
    Normalizes a parameter name to a consistent canonical form before saving
    to history, so variants like 'HAEMOGLOBIN', 'Haemoglobin', 'haemoglobin'
    all get stored under the same key and appear as one trend line.
    Checks title-case, original, and uppercase forms against NAME_CANONICALIZE.
    """
    s = name.strip()
    for candidate in [s, s.title(), s.upper()]:
        if candidate in NAME_CANONICALIZE:
            return NAME_CANONICALIZE[candidate]
    return s.title()  # default: title-case for consistent display


def save_to_history(results):
    # Normalize parameter names before saving so variant spellings
    # from different reports merge into one trend line
    normalized_results = []
    for r in results:
        normalized = dict(r)
        normalized["parameter"] = normalize_parameter_name(r["parameter"])
        normalized_results.append(normalized)

    record = {"uploaded_at": datetime.now().isoformat(), "results": normalized_results}
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    history.append(record)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    return len(history)


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
        # Normalize parameter names on load to fix any pre-existing variant spellings
        for entry in history:
            for r in entry.get("results", []):
                r["parameter"] = normalize_parameter_name(r["parameter"])
        return history
    return []


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:11px; margin-bottom:2rem;">
            <div style="width:40px; height:40px; border-radius:11px;
                        background:linear-gradient(135deg,{ACCENT} 0%,{ACCENT_PINK} 100%);
                        display:flex; align-items:center; justify-content:center; font-size:20px; flex-shrink:0;">🩺</div>
            <span class="sidebar-logo-text">MedReport AI</span>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "nav", ["📄   Analyze report", "📈   Trend history"],
            label_visibility="collapsed", key="nav_page"
        )
    return page


# ---------------------------------------------------------------------------
# ANALYZE PAGE
# ---------------------------------------------------------------------------
def render_analyze_page():
    st.markdown("# MedReport AI")
    st.markdown(
        '<p style="color:#8A8398; margin-top:-0.6rem; font-size:15px;">Upload a lab report and get a clear, '
        'plain-language breakdown of your results — in under a minute.</p>', unsafe_allow_html=True
    )

    # --- Feature row: gives the empty state some life instead of jumping straight to a bare upload box ---
    if "processed_file_signature" not in st.session_state:
        st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        features = [
            (f1, "🔍", "OCR + AI extraction", "Reads PDFs and photos of lab reports using OCR and biomedical NLP."),
            (f2, "🚩", "Flags what matters", "Automatically highlights values outside the normal range."),
            (f3, "📈", "Tracks over time", "Save reports to see how your values trend across visits."),
        ]
        for col, icon, title, desc in features:
            with col:
                st.markdown(f"""
                <div class="panel-box" style="text-align:left; min-height:118px;">
                    <div style="font-size:22px; margin-bottom:6px;">{icon}</div>
                    <div style="font-weight:600; font-size:14px; color:#2B2640; margin-bottom:3px;">{title}</div>
                    <div style="font-size:12.5px; color:#8A8398; line-height:1.45;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="upload-heading">Step 1 — Upload your report</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a lab report (PDF)", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is None:
        return

    # --- Cache key: only reprocess if this is a NEW file, not on every UI interaction ---
    file_signature = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.get("processed_file_signature") != file_signature:
        with open("temp_upload.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # --- Step-by-step processing indicator ---
        step_box = st.empty()
        steps = ["Reading PDF", "Running OCR", "Loading biomedical NER", "Extracting values", "Generating explanations"]

        def render_steps(active_idx):
            html = ""
            for i, s in enumerate(steps):
                cls = "done" if i < active_idx else ("active" if i == active_idx else "")
                icon = "✅" if i < active_idx else ("⏳" if i == active_idx else "◯")
                html += f'<div class="step-row {cls}">{icon} {s}</div>'
            step_box.markdown(html, unsafe_allow_html=True)

        render_steps(0)
        images = pdf_to_images("temp_upload.pdf")
        images[0].save("temp_page.png")

        render_steps(1)
        reader = load_ocr_reader()
        raw_text, avg_confidence, detections = run_ocr(reader, "temp_page.png")

        render_steps(2)
        ner_pipe = load_ner_pipeline()

        render_steps(3)
        ner_results = ner_pipe(raw_text) if ner_pipe else []
        text_order_records = extract_with_hybrid(raw_text, ner_results)
        rows = group_into_rows(detections)
        row_records = extract_by_row(rows)
        hybrid_records = merge_records(row_records, text_order_records)

        render_steps(4)
        explained = [explain_record(r) for r in hybrid_records]
        step_box.empty()

        # Save local OCR results before Gemini potentially replaces them
        # so the user can toggle between both views
        local_explained = explained[:]

        # --- Gemini Vision fallback ---
        # Local pipeline runs first (above). Gemini is only called when OCR confidence
        # falls below GEMINI_CONFIDENCE_THRESHOLD -- i.e. when local extraction is
        # likely unreliable.
        #
        GEMINI_CONFIDENCE_THRESHOLD = 0.95

        # Key priority: Streamlit Secrets (cloud) → environment variable (Colab) → empty
        gemini_key = ""
        try:
            gemini_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            gemini_key = os.environ.get("GEMINI_API_KEY", "")

        extraction_method = "local"

        if gemini_key and avg_confidence < GEMINI_CONFIDENCE_THRESHOLD:
            with st.spinner(f"⚠️ OCR confidence ({avg_confidence*100:.0f}%) below {GEMINI_CONFIDENCE_THRESHOLD*100:.0f}% — refining with Gemini Vision AI..."):
                gemini_results = extract_with_gemini("temp_page.png", gemini_key)

            if gemini_results:
                explained = gemini_results
                extraction_method = "gemini"
            else:
                error_detail = st.session_state.get("gemini_last_error", "Unknown error")
                st.warning(f"Gemini Vision API call failed — showing local extraction results.\n\nError: {error_detail}")

        # Save results into session_state so future reruns (pie clicks, dropdowns) skip reprocessing
        st.session_state.processed_file_signature = file_signature
        st.session_state.cached_explained = explained
        st.session_state.cached_local_explained = local_explained
        st.session_state.cached_raw_text = raw_text
        st.session_state.cached_avg_confidence = avg_confidence
        st.session_state.cached_filename = uploaded_file.name
        st.session_state.cached_extraction_method = extraction_method
        st.session_state.selected_param_idx = 0  # reset selection for the new report

    # --- Load from cache (instant on reruns triggered by chart/dropdown clicks) ---
    explained = st.session_state.cached_explained
    local_explained = st.session_state.get("cached_local_explained", explained)
    raw_text = st.session_state.cached_raw_text
    avg_confidence = st.session_state.cached_avg_confidence
    filename = st.session_state.cached_filename
    extraction_method = st.session_state.get("cached_extraction_method", "local")

    if not explained:
        st.error("No recognizable lab values were found in this report. Try a different file.")
        return

    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="upload-heading">Step 2 — Results</div>', unsafe_allow_html=True)
    st.markdown(f"#### {filename}")
    method_badge = (
        '&nbsp;&nbsp;<span style="background:#E6F2EE; color:#3B6D11; font-size:11px; '
        'font-weight:600; padding:2px 9px; border-radius:999px;">✨ Gemini Vision</span>'
        if extraction_method == "gemini" else
        '&nbsp;&nbsp;<span style="background:#E7EDFB; color:#3A5FA8; font-size:11px; '
        'font-weight:600; padding:2px 9px; border-radius:999px;">⚙️ Local AI</span>'
    )
    st.markdown(
        f'<div style="font-size:13px; color:#8A8398; margin-top:-0.4rem;">'
        f'Analyzed just now &nbsp;·&nbsp; OCR confidence {avg_confidence*100:.0f}%'
        f'{method_badge}</div>',
        unsafe_allow_html=True
    )

    # --- View toggle (only shown when Gemini was used and local results differ) ---
    both_available = extraction_method == "gemini" and len(local_explained) > 0
    if both_available:
        st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
        view_options = ["✨ Gemini Vision (default)", "⚙️ Local OCR pipeline"]
        selected_view = st.radio(
            "View results from:",
            view_options,
            horizontal=True,
            label_visibility="collapsed",
            key="results_view_toggle"
        )
        if selected_view == view_options[1]:
            explained = local_explained
        st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)

    # --- Metric row ---
    total = len(explained)
    n_normal = sum(1 for e in explained if e['status'] == 'NORMAL')
    n_high = sum(1 for e in explained if e['status'] == 'HIGH')
    n_low = sum(1 for e in explained if e['status'] == 'LOW')
    m1, m2, m3, m4 = st.columns(4)
    metrics = [(m1, "Total values", total, "#8FA8E0"), (m2, "Normal", n_normal, "#6FA893"),
               (m3, "High", n_high, "#E0729F"), (m4, "Low", n_low, "#6C8FE0")]
    for col, label, value, color in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="--accent:{color};">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # --- Full pie chart (all slices visible) + clickable legend + detail panel ---
    col_chart, col_detail = st.columns([1.15, 1])

    param_names = [e['parameter'] for e in explained]
    values_for_pie = [1 for _ in explained]  # equal-weight slices so every parameter gets its own visible wedge
    slice_colors = [PIE_PALETTE[i % len(PIE_PALETTE)] for i in range(len(param_names))]

    if "selected_param_idx" not in st.session_state:
        st.session_state.selected_param_idx = 0

    with col_chart:
        with st.container(border=True):
            fig = go.Figure(data=[go.Pie(
                labels=param_names, values=values_for_pie, hole=0,
                marker=dict(colors=slice_colors, line=dict(color="#FFFFFF", width=2)),
                textinfo="label", textfont=dict(family="Inter", size=11, color="#2B2640"),
                pull=[0.07 if i == st.session_state.selected_param_idx else 0 for i in range(len(param_names))],
                sort=False,
            )])
            fig.update_layout(
                showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=300,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True, key="status_pie")

            # Clickable legend -- color-dot buttons matched to each slice.
            # (Plotly's native click-to-select on pie charts is unreliable in Streamlit,
            # so this is the guaranteed-to-work way to let you pick a slice by clicking.)
            st.markdown('<div style="font-size:11.5px; color:#8A8398; margin:8px 0 4px;">Click a value:</div>', unsafe_allow_html=True)
            n_cols = 4
            legend_cols = st.columns(n_cols)
            for i, (name, color) in enumerate(zip(param_names, slice_colors)):
                with legend_cols[i % n_cols]:
                    is_selected = (i == st.session_state.selected_param_idx)
                    label = f"● {name}"
                    if st.button(label, key=f"legend_btn_{i}", use_container_width=True,
                                 type="primary" if is_selected else "secondary"):
                        st.session_state.selected_param_idx = i
                        st.rerun()

    with col_detail:
        idx = min(st.session_state.selected_param_idx, len(explained) - 1)
        sel = explained[idx]
        st.markdown(f"""
        <div class="detail-card">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-size:18px; font-weight:600;">{sel['parameter']}</span>
                <span class="detail-pill">{sel['status']}</span>
            </div>
            <div style="font-size:13.5px; opacity:0.9;">{sel['value']} {sel['unit']} &nbsp;·&nbsp; normal range {sel['range']}</div>
            <div style="font-size:13.5px; margin-top:8px; line-height:1.6;">{sel['description']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # --- Results table (styled like the original table) ---
    st.markdown("#### All results")
    df = pd.DataFrame(explained)
    df_display = df.rename(columns={
        "parameter": "Parameter", "value": "Result", "unit": "Unit",
        "range": "Reference Range", "status": "Flag"
    })[["Parameter", "Result", "Unit", "Reference Range", "Flag"]]

    def highlight_status(row):
        bg = STATUS_BG.get(row['Flag'], "#EFEDF3")
        return [f"background-color: {bg}; color: #2B2640; font-weight: 500;"] * len(row)

    styled_table = df_display.style.apply(highlight_status, axis=1).set_table_styles([
        {"selector": "th", "props": [("background-color", "#1B1A23"), ("color", "#FFFFFF"),
                                      ("font-weight", "600"), ("padding", "10px 14px")]},
        {"selector": "td", "props": [("padding", "9px 14px")]},
    ])
    st.dataframe(styled_table, use_container_width=True, hide_index=True)

    # --- Actions ---
    def build_pie_chart_image(param_names, slice_colors_local):
        """Renders the status pie chart as a PNG in-memory using matplotlib (for embedding in the PDF)."""
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        sizes = [1 for _ in param_names]
        ax.pie(sizes, labels=param_names, colors=slice_colors_local,
               textprops={'fontsize': 8, 'color': '#2B2640'}, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
        ax.set_aspect('equal')
        fig.patch.set_alpha(0)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        plt.close(fig)
        buf.seek(0)
        return buf

    def build_report_pdf():
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter,
                                 topMargin=0.6 * inch, bottomMargin=0.7 * inch,
                                 leftMargin=0.6 * inch, rightMargin=0.6 * inch)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('TitleStyle', parent=styles['Title'], fontSize=20,
                                      textColor=rl_colors.HexColor("#2B2640"), spaceAfter=2)
        subtitle_style = ParagraphStyle('SubtitleStyle', parent=styles['Normal'], fontSize=10,
                                         textColor=rl_colors.HexColor("#8A8398"), spaceAfter=14)
        section_style = ParagraphStyle('SectionStyle', parent=styles['Heading2'], fontSize=13,
                                        textColor=rl_colors.HexColor("#2B2640"), spaceBefore=14, spaceAfter=8)
        normal_style = styles['Normal']

        story = []

        # --- Header ---
        story.append(Paragraph("MedReport AI", title_style))
        story.append(Paragraph(
            f"Lab Report Analysis &nbsp;|&nbsp; {filename} &nbsp;|&nbsp; "
            f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')}", subtitle_style))

        # --- Summary cards (as a small table) ---
        story.append(Paragraph("Summary", section_style))
        summary_data = [["Total Values", "Normal", "High", "Low", "OCR Confidence"],
                         [str(total), str(n_normal), str(n_high), str(n_low), f"{avg_confidence*100:.0f}%"]]
        summary_table = Table(summary_data, colWidths=[1.7 * inch] * 5)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor("#1B1A23")),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, 1), 14),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#E8D5E0")),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, 1), rl_colors.HexColor("#FDF3F8")),
        ]))
        story.append(summary_table)

        # --- Pie chart ---
        story.append(Paragraph("Result Breakdown", section_style))
        chart_buf = build_pie_chart_image(param_names, slice_colors)
        story.append(RLImage(chart_buf, width=3.4 * inch, height=3.4 * inch))

        # --- Full results table ---
        story.append(Paragraph("Full Results", section_style))
        table_data = [["Parameter", "Result", "Unit", "Reference Range", "Flag"]]
        row_colors = []
        for e in explained:
            table_data.append([e['parameter'], str(e['value']), e['unit'], e['range'], e['status']])
            row_colors.append(STATUS_BG.get(e['status'], "#EFEDF3"))

        results_table = Table(table_data, colWidths=[1.6 * inch, 0.9 * inch, 0.9 * inch, 1.6 * inch, 0.85 * inch])
        table_style_cmds = [
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor("#1B1A23")),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#E8D5E0")),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]
        for i, bg in enumerate(row_colors, start=1):
            table_style_cmds.append(('BACKGROUND', (0, i), (-1, i), rl_colors.HexColor(bg)))
        results_table.setStyle(TableStyle(table_style_cmds))
        story.append(results_table)

        story.append(Spacer(1, 24))
        footer_style = ParagraphStyle('FooterStyle', parent=styles['Normal'], fontSize=8,
                                       textColor=rl_colors.HexColor("#A39FB0"), alignment=TA_CENTER)
        story.append(Paragraph(
            "Generated by MedReport AI &mdash; for educational purposes only. "
            "Not a substitute for professional medical advice.", footer_style))

        doc.build(story)
        buf.seek(0)
        return buf.getvalue()

    pdf_bytes = build_report_pdf()
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("💾 Save to history", type="primary", use_container_width=True):
            count = save_to_history(explained)
            st.success(f"Saved. You now have {count} report(s) in history.")
    with col2:
        st.download_button("⬇️ Download report (PDF)", pdf_bytes,
                            f"medreport_{filename.rsplit('.', 1)[0]}.pdf", "application/pdf",
                            use_container_width=True)

    with st.expander("View raw OCR text (debug)"):
        st.text(raw_text)


# ---------------------------------------------------------------------------
# TREND PAGE
# ---------------------------------------------------------------------------
def render_trend_page():
    st.markdown("# Trend history")
    st.markdown(
        '<p style="color:#8A8398; margin-top:-0.6rem; font-size:15px;">See how a value has changed across your saved reports.</p>',
        unsafe_allow_html=True,
    )

    history = load_history()
    if not history:
        st.markdown(
            '<div class="panel-box" style="text-align:center; padding:3rem 1.5rem;">'
            '<div style="font-size:15px; color:#2B2640;">No saved reports yet.</div>'
            '<div style="font-size:13px; color:#8A8398; margin-top:6px;">'
            'Analyze a report and click <b>Save to history</b> to start tracking trends.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    all_params = sorted({r['parameter'] for entry in history for r in entry['results']})

    st.markdown('<div class="upload-heading">Choose a value to track</div>', unsafe_allow_html=True)
    selected_param = st.selectbox("Select a parameter to view trend", all_params, label_visibility="collapsed")

    dates, values, statuses = [], [], []
    for entry in history:
        for r in entry['results']:
            if r['parameter'] == selected_param:
                try:
                    values.append(float(r['value']))
                    dates.append(entry['uploaded_at'][:10])
                    statuses.append(r.get('status', 'UNKNOWN'))
                except ValueError:
                    pass

    if not values:
        st.info(f"No numeric history found for {selected_param}.")
        return

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # --- Plain-language summary sentence -- this is the part that makes it understandable ---
    latest_val, latest_status = values[-1], statuses[-1]
    status_word = {"NORMAL": "within the normal range", "HIGH": "above the normal range",
                   "LOW": "below the normal range", "UNKNOWN": "of unknown status"}[latest_status]

    if len(values) >= 2:
        prev_val = values[-2]
        diff = latest_val - prev_val
        if abs(diff) < 1e-9:
            trend_sentence = f"stayed the same since your last report ({prev_val:g} → {latest_val:g})"
        else:
            direction = "increased" if diff > 0 else "decreased"
            trend_sentence = f"{direction} from {prev_val:g} to {latest_val:g} since your last report"
    else:
        trend_sentence = "has only been recorded once so far"

    pill_color = STATUS_DOT.get(latest_status, "#888")
    st.markdown(f"""
    <div class="panel-box" style="background:linear-gradient(135deg,{ACCENT}11,{ACCENT_PINK}11); border-color:{pill_color}55;">
        <span style="font-size:15px; color:#2B2640;">
            Your latest <b>{selected_param}</b> is <b>{latest_val:g}</b>, which is <b style="color:{pill_color};">{status_word}</b>.
            It has <b>{trend_sentence}</b>.
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)

    # --- ONE chart: line trend, color-coded by status at each point ---
    fig = go.Figure(data=[go.Scatter(
        x=dates, y=values, mode="lines+markers",
        line=dict(color="#D9C2D0", width=2.5),
        marker=dict(size=12, color=[STATUS_DOT.get(s, ACCENT) for s in statuses], line=dict(color="white", width=2)),
        hovertemplate="%{x}<br>" + selected_param + ": %{y}<extra></extra>",
    )])
    fig.update_layout(
        height=280,
        margin=dict(t=20, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter", color="#2B2640", size=12),
        xaxis=dict(title=None, gridcolor="#F0DCE8", showline=True, linecolor="#E8D5E0", type="category",
                    tickfont=dict(color="#2B2640", size=11)),
        yaxis=dict(title=dict(text=selected_param, font=dict(color="#2B2640", size=13)),
                    gridcolor="#F0DCE8", showline=True, linecolor="#E8D5E0",
                    tickfont=dict(color="#2B2640", size=11)),
    )
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True, key="trend_line_chart")
        legend_html = " &nbsp;&nbsp; ".join(
            f'<span style="color:{STATUS_DOT[s]};">●</span> {s.title()}' for s in ["NORMAL", "HIGH", "LOW"]
        )
        st.markdown(f'<div style="text-align:center; font-size:11.5px; color:#8A8398; margin-top:4px;">{legend_html}</div>', unsafe_allow_html=True)

    if len(values) == 1:
        st.caption("Save another report to start seeing a trend line.")

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # --- Simple visit-by-visit list (replaces redundant bar chart + raw table) ---
    st.markdown("#### Visit history")
    for i in reversed(range(len(values))):
        bg = STATUS_BG.get(statuses[i], "#F1EFE8")
        color = STATUS_HEX.get(statuses[i], "#5F5E5A")
        tag = "Latest" if i == len(values) - 1 else f"{len(values) - 1 - i} visit(s) ago"
        st.markdown(f"""
        <div class="result-row">
            <div style="flex:1;">
                <div style="font-size:13px; font-weight:500;">{dates[i]} <span style="color:#A39FB0; font-weight:400;">· {tag}</span></div>
                <div style="font-size:12px; color:#6B6760; margin-top:2px;">{selected_param}: {values[i]:g}</div>
            </div>
            <span class="result-status-pill" style="background:{bg}; color:{color};">{statuses[i]}</span>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
inject_css()
page = render_sidebar()

if "Analyze" in page:
    render_analyze_page()
else:
    render_trend_page()
