# ✦ Text Analysis Studio

A small NLP playground for curious minds: paste an English text, get linguistic statistics and charts; paste several texts, compare their vocabulary and similarity.

**[🚀 Try it online — no installation needed](https://text-analysis-studio.streamlit.app/)**

> 💤 **Note:** the app is hosted on Streamlit Community Cloud, which puts apps to sleep after a period of inactivity. If you see a "This app has gone to sleep" screen, just click **"Yes, get this app back up!"** and wait about 30–60 seconds while it wakes up.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B)
![spaCy](https://img.shields.io/badge/NLP-spaCy-09A3D5)

<img width="1920" height="866" alt="Screenshot 2026-10-17 091520" src="https://github.com/user-attachments/assets/6076deac-8d12-4eb0-8e3b-f2f7337cec03" />
<img width="1920" height="868" alt="Screenshot 2026-10-17 091537" src="https://github.com/user-attachments/assets/36446d05-3b8e-4209-853c-edd381d25903" />
<img width="1920" height="872" alt="Screenshot 2026-10-17 091547" src="https://github.com/user-attachments/assets/e8dec043-ef12-4fdd-a0ca-34954ed8538f" />

## Features

### Single text analysis
- **Basic statistics:** total words, unique words, number of sentences, average sentence length
- **Lexical diversity:** ratio of unique words to total words (closer to 1 means more varied vocabulary)
- **Most frequent words** and **most frequent lemmas** (top 15)
- **Part-of-speech distribution**
- **Word frequency bar chart** (top 10 words)

### Text comparison (TF-IDF)
- Compare **2 to 10 texts** at once
- Shared vocabulary across texts (based on lemmas)
- Average pairwise **vocabulary overlap**
- **TF-IDF** and **cosine similarity** to measure how close the texts are and which terms matter most in each

### Interface
- Custom-styled, responsive wide layout
- Styled tables and charts that match the app theme
- Input validation with friendly warnings

## Tech stack

| Purpose | Library |
| --- | --- |
| Web app / UI | [Streamlit](https://streamlit.io/) |
| Tokenization, lemmas, POS tags, sentence splitting | [spaCy](https://spacy.io/) (`en_core_web_sm`) |
| Data tables | [pandas](https://pandas.pydata.org/) |
| Charts | [Matplotlib](https://matplotlib.org/) |
| TF-IDF and cosine similarity | [scikit-learn](https://scikit-learn.org/) |

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/kisarrrrav/Text-Analysis-Dashboard.git
cd Text-Analysis-Dashboard
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This also installs the English spaCy model `en_core_web_sm` (the model wheel is listed in `requirements.txt`). If you ever need to install it manually:

```bash
python -m spacy download en_core_web_sm
```

### 4. Run the app

```bash
streamlit run text_analysis_dashboard.py
```

The app opens at `http://localhost:8501`.

## How to use

1. **Single text:** paste an English text into the first box and click **✦ Analyze text**.
2. **Comparison:** scroll to *TF-IDF & Text Comparison*, fill in at least two texts (use **＋ Add text** for more, up to 10) and click **✦ Compare texts**.

## Project structure

```
Text-Analysis-Dashboard/
├── text_analysis_dashboard.py   # Streamlit app (UI, analysis, charts)
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

## Limitations

- Only **English** texts are supported (the app uses the English spaCy model).
- Word statistics count every alphabetic token, including common function words like "the" or "of".

## Roadmap ideas

- Support for more languages
- Upload `.txt` / `.csv` files
- Export results to CSV
- Sentiment analysis and named entity recognition

## Author

Made by [kisarrrrav](https://github.com/kisarrrrav).
