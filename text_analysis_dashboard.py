import streamlit as st
import spacy
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from html import escape
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Text Analysis Studio",
    page_icon="♡",
    layout="wide"
)


# ============================================================
# LOAD SPACY
# ============================================================

@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")


nlp = load_model()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GENERAL ---------- */

    .stApp {
        background: #F7F1EC;
        color: #4B3038;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #4B3038;
    }

    p, li, label {
        color: #5A4249;
    }


    /* ---------- GITHUB CREDIT ---------- */

    .github-credit {
        position: fixed;
        top: 72px;
        right: 28px;
        z-index: 9999;
        font-size: 0.78rem;
        color: #9B7180;
        text-align: right;
        white-space: nowrap;
    }

    .github-credit a {
        color: #9B7180;
        text-decoration: none;
    }

    .github-credit a:hover {
        color: #704A56;
        text-decoration: underline;
    }


    /* ---------- HEADER ---------- */

    .main-title {
        text-align: center;
        margin-top: 85px;
        margin-bottom: 0.2rem;
        font-family: Georgia, serif;
        font-size: 3.2rem;
        color: #4B3038;
        letter-spacing: -1px;
    }

    .subtitle {
        text-align: center;
        color: #8C6873;
        font-size: 1.05rem;
        margin-bottom: 2.5rem;
    }


    /* ---------- INTRO CARDS ---------- */

    .intro-card {
        background: #FFFDFC;
        border: 1px solid #EBD9DE;
        border-radius: 18px;
        padding: 1.3rem 1.4rem;
        min-height: 150px;
        box-shadow: 0 4px 14px rgba(90, 60, 70, 0.04);
    }

    .intro-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }

    .intro-title {
        font-weight: 700;
        color: #4B3038;
        margin-bottom: 0.4rem;
    }

    .intro-text {
        color: #765A63;
        font-size: 0.92rem;
        line-height: 1.55;
    }


    /* ---------- SECTION HEADINGS ---------- */

    .section-heading {
        color: #4B3038;
        margin-top: 2rem;
        margin-bottom: 0.4rem;
    }

    .single-analysis-heading {
        margin-top: 0;
    }

    .section-description {
        color: #856973;
        margin-bottom: 1.2rem;
    }


    /* ---------- TEXT AREAS ---------- */

    textarea {
        background-color: #FFF3F6 !important;
        color: #4B3038 !important;
        border: 1px solid #E4C4CC !important;
        border-radius: 14px !important;
    }

    textarea::placeholder {
        color: #B48E99 !important;
        opacity: 1 !important;
    }

    textarea:focus {
        border-color: #C98598 !important;
        box-shadow: 0 0 0 1px #C98598 !important;
    }


    /* ---------- BUTTONS ---------- */

    .stButton {
        display: flex;
        justify-content: flex-end;
    }

    .stButton > button {
        background: #C98598;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.55rem 1.2rem;
        font-weight: 600;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        background: #A9677B;
        color: white;
        border: none;
    }


    /* ---------- RESULT BOXES ---------- */

    .result-box {
        background: #FFF3F6;
        border: 1px solid #E7C9D1;
        border-radius: 16px;
        padding: 1.2rem 1.35rem;
        margin: 0.7rem 0;
        min-height: 130px;
    }

    .result-box-title {
        font-size: 1rem;
        font-weight: 700;
        color: #704A56;
        margin-bottom: 0.65rem;
    }

    .result-box-text {
        color: #5A4249;
        line-height: 1.6;
        font-size: 0.92rem;
    }


    /* ---------- METRIC CARDS ---------- */

    .metric-card {
        background: #FFFDFC;
        border: 1px solid #EBD9DE;
        border-radius: 16px;
        padding: 1.1rem;
        text-align: center;
        min-height: 105px;
    }

    .metric-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #A9677B;
    }

    .metric-label {
        color: #80616A;
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }


    /* ---------- COMPARISON CARDS ---------- */

    .comparison-card {
        background: #FFF3F6;
        border: 1px solid #E7C9D1;
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        min-height: 105px;
    }

    .comparison-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #A9677B;
    }

    .comparison-label {
        color: #80616A;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }


    /* ---------- CUSTOM TABLES ---------- */

    .custom-table-wrapper {
        width: 100%;
        overflow-x: auto;
        margin: 0.8rem 0 1.4rem 0;
        border-radius: 14px;
        border: 1px solid #E7C9D1;
    }

    .custom-table {
        width: 100%;
        border-collapse: collapse;
        background: #FFF3F6;
        color: #4B3038;
        font-size: 0.9rem;
    }

    .custom-table th {
        background: #F9E8EC;
        color: #704A56;
        font-weight: 700;
        padding: 0.75rem 0.9rem;
        text-align: left;
        border-bottom: 1px solid #E7C9D1;
    }

    .custom-table td {
        background: #FFF3F6;
        color: #4B3038;
        padding: 0.7rem 0.9rem;
        border-bottom: 1px solid #F0DDE2;
    }

    .custom-table tr:last-child td {
        border-bottom: none;
    }

    .custom-table tr:hover td {
        background: #FCECEF;
    }


    /* ---------- DIVIDERS ---------- */

    hr {
        border: none;
        border-top: 1px solid #E5D4D9;
        margin: 2.5rem 0;
    }


    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #9B7A83;
        font-size: 0.82rem;
        padding: 1.5rem 0 0.5rem 0;
    }

    .footer-symbols {
        color: #C98598;
        font-size: 1rem;
        margin-bottom: 0.4rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER: CUSTOM TABLE
# ============================================================

def render_table(dataframe):
    if dataframe.empty:
        return

    headers = "".join(
        f"<th>{escape(str(column))}</th>"
        for column in dataframe.columns
    )

    rows = []

    for _, row in dataframe.iterrows():
        cells = "".join(
            f"<td>{escape(str(value))}</td>"
            for value in row
        )
        rows.append(f"<tr>{cells}</tr>")

    table_html = f"""
    <div class="custom-table-wrapper">
        <table class="custom-table">
            <thead>
                <tr>{headers}</tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    </div>
    """

    st.markdown(
        table_html,
        unsafe_allow_html=True
    )


# ============================================================
# HELPER: PINK CHART STYLE
# ============================================================

def style_chart(fig, ax):
    fig.patch.set_facecolor("#F7F1EC")
    ax.set_facecolor("#F7F1EC")

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(
        colors="#765A63"
    )

    ax.xaxis.label.set_color("#765A63")
    ax.yaxis.label.set_color("#765A63")
    ax.title.set_color("#4B3038")


# ============================================================
# GITHUB CREDIT
# ============================================================

st.markdown(
    """
    <div class="github-credit">
        by <a href="https://github.com/kisarrrrav" target="_blank">
        kisarrrrav (V)
        </a>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Text Analysis Studio</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Explore vocabulary, structure, lemmas and lexical diversity.
        <br>
        ✦ A small NLP playground for curious minds ✦
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INTRO CARDS
# ============================================================

intro_cols = st.columns(3)

with intro_cols[0]:
    st.markdown(
        """
        <div class="intro-card">
            <div class="intro-icon">♡</div>
            <div class="intro-title">Analyze text</div>
            <div class="intro-text">
                Explore vocabulary, sentence structure, lemmas,
                lexical diversity and basic linguistic statistics.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with intro_cols[1]:
    st.markdown(
        """
        <div class="intro-card">
            <div class="intro-icon">✦</div>
            <div class="intro-title">Compare texts</div>
            <div class="intro-text">
                Compare vocabulary overlap and semantic similarity
                between several texts using TF-IDF.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with intro_cols[2]:
    st.markdown(
        """
        <div class="intro-card">
            <div class="intro-icon">✿</div>
            <div class="intro-title">Discover patterns</div>
            <div class="intro-text">
                Visualize frequent words, shared vocabulary and
                the most important terms in your texts.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SPACE AFTER INTRO CARDS
# ============================================================

st.markdown(
    '<div style="height: 70px;"></div>',
    unsafe_allow_html=True
)


# ============================================================
# SINGLE TEXT ANALYSIS
# ============================================================

st.markdown(
    '<h2 class="section-heading single-analysis-heading">✦ Single Text Analysis</h2>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-description">
        Enter an English text and explore its linguistic features.
    </div>
    """,
    unsafe_allow_html=True
)

single_text = st.text_area(
    "Text",
    placeholder="Enter your text here...",
    height=220,
    label_visibility="collapsed",
    key="single_text"
)

single_button = st.columns([1, 1, 1])

with single_button[2]:
    analyze_clicked = st.button(
        "✦ Analyze text",
        use_container_width=True
    )


if analyze_clicked:

    if not single_text.strip():

        st.warning(
            "Please enter some text first."
        )

    else:

        doc = nlp(single_text)

        words = [
            token.text.lower()
            for token in doc
            if token.is_alpha
        ]

        lemmas = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha
        ]

        sentences = list(doc.sents)

        unique_words = set(words)

        lexical_diversity = (
            len(unique_words) / len(words)
            if words
            else 0
        )

        word_counter = Counter(words)

        lemma_counter = Counter(lemmas)

        avg_sentence_length = (
            len(words) / len(sentences)
            if sentences
            else 0
        )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        metric_cols = st.columns(4)

        metrics = [
            (len(words), "Words"),
            (len(unique_words), "Unique words"),
            (len(sentences), "Sentences"),
            (
                f"{lexical_diversity:.2f}",
                "Lexical diversity"
            )
        ]

        for col, (number, label) in zip(
            metric_cols,
            metrics
        ):

            with col:

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-number">{number}</div>
                        <div class="metric-label">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # ----------------------------------------------------
        # LEXICAL DIVERSITY
        # ----------------------------------------------------

        st.markdown(
        f"""
        <div class="result-box">
            <div class="result-box-title">♡ Lexical diversity</div>
            <div class="result-box-text">
                Lexical diversity measures how varied the vocabulary is.
                A value closer to 1 means that a larger proportion of the words are unique.
                <br><br>
                This text has a lexical diversity of <b>{lexical_diversity:.2f}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


        # ----------------------------------------------------
        # BASIC TEXT STATISTICS
        # ----------------------------------------------------

        st.markdown(
            '<h3 class="section-heading">Text statistics</h3>',
            unsafe_allow_html=True
        )

        stats_data = pd.DataFrame(
            {
                "Metric": [
                    "Total words",
                    "Unique words",
                    "Sentences",
                    "Average sentence length"
                ],
                "Value": [
                    len(words),
                    len(unique_words),
                    len(sentences),
                    f"{avg_sentence_length:.2f}"
                ]
            }
        )

        render_table(stats_data)


        # ----------------------------------------------------
        # MOST FREQUENT WORDS
        # ----------------------------------------------------

        st.markdown(
            '<h3 class="section-heading">Most frequent words</h3>',
            unsafe_allow_html=True
        )

        most_common_words = word_counter.most_common(15)

        frequency_data = pd.DataFrame(
            most_common_words,
            columns=[
                "Word",
                "Frequency"
            ]
        )

        render_table(frequency_data)


        # ----------------------------------------------------
        # LEMMAS
        # ----------------------------------------------------

        st.markdown(
            '<h3 class="section-heading">Most frequent lemmas</h3>',
            unsafe_allow_html=True
        )

        lemma_data = pd.DataFrame(
            lemma_counter.most_common(15),
            columns=[
                "Lemma",
                "Frequency"
            ]
        )

        render_table(lemma_data)


        # ----------------------------------------------------
        # POS DISTRIBUTION
        # ----------------------------------------------------

        st.markdown(
            '<h3 class="section-heading">Part-of-speech distribution</h3>',
            unsafe_allow_html=True
        )

        pos_counter = Counter(
            token.pos_
            for token in doc
            if token.is_alpha
        )

        pos_data = pd.DataFrame(
            pos_counter.most_common(),
            columns=[
                "Part of speech",
                "Count"
            ]
        )

        render_table(pos_data)


        # ----------------------------------------------------
        # WORD FREQUENCY CHART
        # ----------------------------------------------------

        st.markdown(
            '<h3 class="section-heading">Word frequency</h3>',
            unsafe_allow_html=True
        )

        chart_words = [
            item[0]
            for item in most_common_words[:10]
        ]

        chart_values = [
            item[1]
            for item in most_common_words[:10]
        ]

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        ax.bar(
            chart_words,
            chart_values,
            color="#E8AFC1"
        )

        ax.set_title(
            "Most frequent words"
        )

        ax.set_xlabel(
            "Word"
        )

        ax.set_ylabel(
            "Frequency"
        )

        ax.tick_params(
            axis="x",
            rotation=45
        )

        style_chart(
            fig,
            ax
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


# ============================================================
# TEXT COMPARISON
# ============================================================

st.markdown(
    "<hr>",
    unsafe_allow_html=True
)

st.markdown(
    '<h2 class="section-heading">✦ TF-IDF &amp; Text Comparison</h2>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-description">
        Compare vocabulary and similarity between two or more texts.
        You can add up to ten texts.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "comparison_texts" not in st.session_state:
    st.session_state.comparison_texts = [
        "",
        ""
    ]


# ============================================================
# COMPARISON TEXT INPUTS
# ============================================================

for i in range(
    len(st.session_state.comparison_texts)
):

    st.session_state.comparison_texts[i] = st.text_area(
        f"Text {i + 1}",
        value=st.session_state.comparison_texts[i],
        placeholder="Enter your text here...",
        height=160,
        key=f"comparison_text_{i}"
    )


# ============================================================
# COMPARISON BUTTONS
# ============================================================

button_cols = st.columns(
    [1, 1, 1, 1]
)

with button_cols[2]:

    add_text_clicked = st.button(
        "＋ Add text",
        use_container_width=True
    )

with button_cols[3]:

    compare_clicked = st.button(
        "✦ Compare texts",
        use_container_width=True
    )


if add_text_clicked:

    if len(st.session_state.comparison_texts) < 10:

        st.session_state.comparison_texts.append("")

        st.rerun()

    else:

        st.warning(
            "You can compare up to 10 texts."
        )


# ============================================================
# COMPARISON RESULTS
# ============================================================

if compare_clicked:

    texts = [
        text.strip()
        for text in st.session_state.comparison_texts
        if text.strip()
    ]

    if len(texts) < 2:

        st.warning(
            "Please enter at least two texts to compare."
        )

    else:

        # ----------------------------------------------------
        # WORD SETS
        # ----------------------------------------------------

        word_sets = []

        for text in texts:

            doc = nlp(text)

            words = {
                token.lemma_.lower()
                for token in doc
                if token.is_alpha
            }

            word_sets.append(words)


        # ----------------------------------------------------
        # SHARED VOCABULARY
        # ----------------------------------------------------

        shared_words = set.intersection(
            *word_sets
        )


        # ----------------------------------------------------
        # AVERAGE VOCABULARY OVERLAP
        # ----------------------------------------------------

        overlap_values = []

        for i in range(
            len(texts)
        ):

            for j in range(
                i + 1,
                len(texts)
            ):

                words_a = word_sets[i]
                words_b = word_sets[j]

                union = words_a | words_b
                intersection = words_a & words_b

                if union:

                    overlap = (
                        len(intersection)
                        / len(union)
                    )

                    overlap_values.append(
                        overlap
                    )


        average_overlap = (
            sum(overlap_values)
            / len(overlap_values)
            if overlap_values
            else 0
        )


        # ----------------------------------------------------
        # COMPARISON OVERVIEW
        # ----------------------------------------------------

        st.markdown(
            '<div style="height: 12px;"></div>',
            unsafe_allow_html=True
        )

        overview_cols = st.columns(3)

        overview_items = [
            (len(texts), "Texts compared"),
            (len(shared_words), "Shared words"),
            (f"{average_overlap:.0%}", "Average vocabulary overlap"),
        ]

        for col, (number, label) in zip(
            overview_cols,
            overview_items
        ):

            with col:

                st.markdown(
                    f'<div class="comparison-card">'
                    f'<div class="comparison-number">{number}</div>'
                    f'<div class="comparison-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )


        # ----------------------------------------------------
        # SHARED VOCABULARY
        # ----------------------------------------------------

        st.markdown(
            '<h3 class="section-heading">Shared vocabulary</h3>',
            unsafe_allow_html=True
        )

        shared_words_sorted = sorted(
            shared_words
        )

        if shared_words_sorted:

            shared_data = pd.DataFrame(
                {
                    "Shared word":
                    shared_words_sorted
                }
            )

            render_table(
                shared_data
            )

        else:

            st.info(
                "No shared vocabulary was found."
            )


        # ----------------------------------------------------
        # VOCABULARY OVERLAP MATRIX
        # ----------------------------------------------------

        st.markdown(
            '<h3 class="section-heading">Vocabulary overlap</h3>',
            unsafe_allow_html=True
        )

        overlap_matrix = []

        for i in range(
            len(texts)
        ):

            row = []

            for j in range(
                len(texts)
            ):

                union = (
                    word_sets[i]
                    | word_sets[j]
                )

                intersection = (
                    word_sets[i]
                    & word_sets[j]
                )

                if union:

                    value = (
                        len(intersection)
                        / len(union)
                    )

                else:

                    value = 0

                row.append(
                    f"{value:.0%}"
                )

            overlap_matrix.append(
                row
            )


        overlap_labels = [
            f"Text {i + 1}"
            for i in range(len(texts))
        ]

        overlap_df = pd.DataFrame(
            overlap_matrix,
            columns=overlap_labels,
            index=overlap_labels
        )

        overlap_df.insert(
            0,
            "Text",
            overlap_labels
        )

        render_table(
            overlap_df
        )


        # ----------------------------------------------------
        # TF-IDF
        # ----------------------------------------------------

        st.markdown(
            '<h3 class="section-heading">TF-IDF analysis</h3>',
            unsafe_allow_html=True
        )

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        tfidf_matrix = vectorizer.fit_transform(
            texts
        )

        feature_names = (
            vectorizer
            .get_feature_names_out()
        )


        tfidf_rows = []

        for i in range(
            len(texts)
        ):

            scores = (
                tfidf_matrix[i]
                .toarray()
                .flatten()
            )

            top_indices = (
                scores
                .argsort()[-10:][::-1]
            )

            for index in top_indices:

                if scores[index] > 0:

                    tfidf_rows.append(
                        {
                            "Text":
                                f"Text {i + 1}",

                            "Term":
                                feature_names[index],

                            "TF-IDF":
                                round(
                                    float(
                                        scores[index]
                                    ),
                                    3
                                )
                        }
                    )


        tfidf_df = pd.DataFrame(
            tfidf_rows
        )

        render_table(
            tfidf_df
        )


        # ----------------------------------------------------
        # TF-IDF CHARTS
        # ----------------------------------------------------

        st.markdown(
            '<h3 class="section-heading">Top TF-IDF terms</h3>',
            unsafe_allow_html=True
        )

        for i in range(
            len(texts)
        ):

            scores = (
                tfidf_matrix[i]
                .toarray()
                .flatten()
            )

            top_indices = (
                scores
                .argsort()[-8:][::-1]
            )

            chart_terms = [
                feature_names[index]
                for index in top_indices
                if scores[index] > 0
            ]

            chart_scores = [
                scores[index]
                for index in top_indices
                if scores[index] > 0
            ]

            if chart_terms:

                fig, ax = plt.subplots(
                    figsize=(10, 4)
                )

                ax.bar(
                    chart_terms,
                    chart_scores,
                    color="#E8AFC1"
                )

                ax.set_title(
                    f"Top TF-IDF terms: Text {i + 1}"
                )

                ax.set_xlabel(
                    "Term"
                )

                ax.set_ylabel(
                    "TF-IDF score"
                )

                ax.tick_params(
                    axis="x",
                    rotation=45
                )

                style_chart(
                    fig,
                    ax
                )

                plt.tight_layout()

                st.pyplot(fig)

                plt.close(fig)


        # ----------------------------------------------------
        # COSINE SIMILARITY
        # ----------------------------------------------------

        st.markdown(
            '<h3 class="section-heading">Text similarity</h3>',
            unsafe_allow_html=True
        )

        similarity_matrix = cosine_similarity(
            tfidf_matrix
        )

        similarity_rows = []

        for i in range(
            len(texts)
        ):

            for j in range(
                i + 1,
                len(texts)
            ):

                similarity_rows.append(
                    {
                        "Text pair":
                            f"Text {i + 1} ↔ Text {j + 1}",

                        "Cosine similarity":
                            round(
                                float(
                                    similarity_matrix[i][j]
                                ),
                                3
                            )
                    }
                )


        similarity_df = pd.DataFrame(
            similarity_rows
        )

        render_table(
            similarity_df
        )


        # ----------------------------------------------------
        # SIMILARITY CHART
        # ----------------------------------------------------

        if not similarity_df.empty:

            fig, ax = plt.subplots(
                figsize=(10, 4)
            )

            ax.bar(
                similarity_df["Text pair"],
                similarity_df["Cosine similarity"],
                color="#E8AFC1"
            )

            ax.set_title(
                "Cosine similarity between texts"
            )

            ax.set_xlabel(
                "Text pair"
            )

            ax.set_ylabel(
                "Similarity"
            )

            ax.tick_params(
                axis="x",
                rotation=45
            )

            style_chart(
                fig,
                ax
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <div class="footer-symbols">
            ✦　♡　✿　⌁　✦
        </div>
        Made with curiosity
    </div>
    """,
    unsafe_allow_html=True
)
