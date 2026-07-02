# 🩺 MedReport AI — Medical Report Analyzer

An AI-powered web application that analyzes medical lab reports (PDFs or scanned images), extracts structured values, flags abnormal results, and explains them in plain language anyone can understand.

## Features

- **Multi-panel support** — CBC, Renal Profile, Liver Profile
- **Dual extraction pipeline** — local OCR + biomedical NER as primary; Gemini Vision API as fallback when OCR confidence < 95%
- **Abnormal value flagging** — compares extracted values against reference ranges
- **Plain-language explanations** — written for non-medical readers
- **Interactive pie chart** — click any segment to see details
- **Trend tracking** — save multiple reports and visualize how values change over time
- **PDF export** — structured report with header, summary, chart, and full results table
- **Toggle view** — compare Gemini Vision results vs local OCR pipeline results side by side

## Tech Stack

| Layer | Technology |
|---|---|
| OCR | EasyOCR |
| Biomedical NER | BioBERT (`d4data/biomedical-ner-all`) |
| Vision AI | Google Gemini 2.5 Flash Lite |
| Extraction | Hybrid: coordinate-based row clustering + regex + NER |
| UI | Streamlit + custom CSS |
| Charts | Plotly + Matplotlib |
| PDF Export | ReportLab |

## Setup

### Local (VS Code)

```bash
pip install -r requirements.txt
```

Set your Gemini API key as an environment variable:
```bash
export GEMINI_API_KEY=your_key_here   # Mac/Linux
set GEMINI_API_KEY=your_key_here      # Windows
```

Run:
```bash
streamlit run app.py
```

### Google Colab

```python
# Load key into environment before starting Streamlit
import os
from google.colab import userdata
os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')

# Install system dependency
!apt-get install -y poppler-utils

# Start Streamlit
import subprocess, time
subprocess.Popen(['streamlit', 'run', 'app.py'],
                  stdout=open('/content/logs.txt', 'w'),
                  stderr=subprocess.STDOUT)
time.sleep(10)
```

### Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub
3. Select this repo, `main` branch, `app.py` as the main file
4. Under **Advanced settings → Secrets**, add:
```toml
GEMINI_API_KEY = "your_key_here"
```
5. Click Deploy

## Get a Gemini API Key

Free tier available at [aistudio.google.com](https://aistudio.google.com) — no billing required for testing.

## Project Structure

```
medical-report-analyzer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── packages.txt        # System dependencies (poppler)
└── README.md
```

## AI/ML Concepts Used

- OCR with confidence scoring
- Named Entity Recognition (biomedical NER)
- Coordinate-based table row reconstruction
- Vision-language model integration (Gemini)
- Hybrid extraction pipeline with intelligent fallback
- Trend visualization and time-series tracking

---

*Built for educational purposes. Not a substitute for professional medical advice.*
