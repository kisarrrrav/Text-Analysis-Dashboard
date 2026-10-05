import streamlit as st
import spacy
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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
# LOAD SPACY MODEL
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

    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background-color: #F7F1EC;
        color: #4B3038;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* ========================================================
       GITHUB CREDIT
       ======================================================== */

    .github-credit {
        position: fixed;
        top: 70px;
        right: 32px;
        z-index: 9999;
        font-size: 0.82rem;
        color: #9A8087;
    }

    .github-credit a {
        color: #B66F82;
        text-decoration: none;
        font-weight: 600;
    }

    .github-credit a:hover {
        text-decoration: underline;
    }

    /* ========================================================
       HEADER
       ======================================================== */

    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 3.2rem;
        font-weight: 600;
        color: #4B3038;
        margin-top: 55px;
        margin-bottom: 0.1rem;
        line-height: 1.1;
    }

    .subtitle {
        font-size: 1.03rem;
        color: #866C73;
        margin-bottom: 1.4rem;
    }

    .decorative-line {
        color: #C98598;
        letter-spacing: 0.35rem;
        font-size: 0.9rem;
        margin-bottom: 1.3rem;
    }

    /* ========================================================
       SECTION TITLES
       ======================================================== */

    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.85rem;
        color: #4B3038;
        margin-top: 2rem;
        margin-bottom: 0.35rem;
    }

    .section-description {
        color: #8E777D;
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }

    /* ========================================================
       INTRO CARDS
       ======================================================== */

    .intro-card {
        background: #FFFDFC;
        border: 1px solid #E9D9DE;
        border-radius: 18px;
        padding: 1.2rem 1.3rem;
        height: 100%;
        box-shadow: 0 3px 14px rgba(108, 67, 78, 0.04);
    }

    .intro-icon {
        font-size: 1.45rem;
        color: #B66F82;
        margin-bottom: 0.35rem;
    }

    .intro-title {
        font-weight: 700;
        color: #4B3038;
        margin-bottom: 0.25rem;
    }

    .intro-text {
        font-size: 0.87rem;
        color: #806B72;
        line-height: 1.5;
    }

    /* ========================================================
       METRIC CARDS
       ======================================================== */

    .metric-card {
        background: #FFFDFC;
        border: 1px solid #E9D9DE;
        border-radius: 16px;
        padding: 1rem 0.7rem;
        text-align: center;
        min-height: 105px;
        box-shadow: 0 3px 12px rgba(108, 67, 78, 0.035);
    }

    .metric-label {
        color: #8E777D;
        font-size: 0.8rem;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        color: #4B3038;
        font-size: 1.55rem;
        font-weight: 700;
    }

    /* ========================================================
       COMPARISON CARDS
       ======================================================== */

    .comparison-card {
        background: #FFFDFC;
        border: 1px solid #E9D9DE;
        border-radius: 17px;
        padding: 1.05rem;
        text-align: center;
        min-height: 115px;
        box-shadow: 0 3px 12px rgba(108, 67, 78, 0.035);
    }

    .comparison-number {
        color: #B66F82;
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }

    .comparison-label {
        color: #7E666D;
        font-size: 0.84rem;
        line-height: 1.35;
    }

    /* ========================================================
       TEXT LABELS
       ======================================================== */

    .text-label {
        font-family: 'Playfair Display', serif;
        font-size: 1.25rem;
        color: #4B3038;
        margin-top: 0.8rem;
        margin-bottom: 0.3rem;
    }

    /* ========================================================
       TEXT INPUT
       ======================================================== */

    .stTextArea textarea {
        background-color: #FFF3F6 !important;
        color: #4B3038 !important;
        border: 1px solid #E7C6CF !important;
        border-radius: 14px !important;
        font-family: 'DM Sans', sans-serif !important;
        line-height: 1.55 !important;
    }

    .stTextArea textarea:focus {
        background-color: #FFF8F9 !important;
        border: 1px solid #C98598 !important;
        box-shadow: 0 0 0 1px #C98598 !important;
    }

    .stTextArea textarea::placeholder {
        color: #A98D94 !important;
        opacity: 1 !important;
    }

    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 12px !important;
        border: 1px solid #DDBBC4 !important;
        background-color: #FFF8F9 !important;
        color: #704A56 !important;
        font-weight: 600 !important;
        min-height: 42px !important;
    }

    .stButton > button:hover {
        border-color: #C98598 !important;
        background-color: #F9E8EC !important;
        color: #5D3A45 !important;
    }

    .stButton > button[kind="primary"] {
        background-color: #C98598 !important;
        color: white !important;
        border-color: #C98598 !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #B66F82 !important;
        border-color: #B66F82 !important;
    }

    /* ========================================================
       RESULT BOX
       ======================================================== */

    .result-box {
        background: #FFF7F8;
        border: 1px solid #EAD5DB;
        border-radius: 16px;
        padding: 1rem 1.15rem;
        margin-bottom: 1rem;
    }

    .result-box-title {
        font-weight: 700;
        color: #5D3A45;
        margin-bottom: 0.25rem;
    }

    .result-box-text {
        color: #806B72;
        font-size: 0.87rem;
        line-height: 1.5;
    }

    /* ========================================================
       WORD PILLS
       ======================================================== */

    .word-pill {
        display: inline-block;
        background: #F4E3E8;
        color: #704A56;
        border-radius: 20px;
        padding: 5px 10px;
        margin: 3px;
        font-size: 0.84rem;
    }

    /* ========================================================
       DATA TABLES
       ======================================================== */

    [data-testid="stDataFrame"] {
        border: 1px solid #E7C6CF !important;
        border-radius: 14px !important;
        overflow: hidden !important;
        background: #FFF3F6 !important;
    }

    [data-testid="stDataFrame"] > div {
        background: #FFF3F6 !important;
    }

    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;
        color: #9A8087;
        font-size: 0.82rem;
        padding-top: 0.5rem;
    }

    .footer-symbols {
        color: #C98598;
        letter-spacing: 0.3rem;
        margin-bottom: 0.4rem;
    }

    hr {
        border: none;
        border-top: 1px solid #E6D7DB;
        margin: 2.3rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def analyze_text(text):

    doc = nlp(text)

    words = [
        token.text.lower()
        for token in doc
        if token.is_alpha
    ]

    content_words = [
        token.text.lower()
        for token in doc
        if token.is_alpha and not token.is_stop
    ]

    lemmas = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop
    ]

    sentences = list(doc.sents)

    word_counter = Counter(words)
    content_counter = Counter(content_words)
    lemma_counter = Counter(lemmas)

    pos_counter = Counter(
        token.pos_
        for token in doc
        if token.is_alpha
    )

    return {
        "characters": len(text),
        "words": len(words),
        "unique_words": len(set(words)),
        "content_words": len(content_words),
        "sentences": len(sentences),
        "lexical_diversity": (
            len(set(words)) / len(words)
            if words else 0
        ),
        "avg_sentence_length": (
            len(words) / len(sentences)
            if sentences else 0
        ),
        "word_frequency": word_counter,
        "content_frequency": content_counter,
        "lemma_frequency": lemma_counter,
        "pos_frequency": pos_counter
    }


def normalize_text_for_comparison(text):

    doc = nlp(text)

    return " ".join(
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop
    )


def get_tfidf_keywords(texts, top_n=8):

    normalized = [
        normalize_text_for_comparison(text)
        for text in texts
    ]

    vectorizer = TfidfVectorizer()

    matrix = vectorizer.fit_transform(normalized)

    feature_names = vectorizer.get_feature_names_out()

    results = []

    for row_index in range(matrix.shape[0]):

        scores = matrix[row_index].toarray().flatten()

        top_indices = scores.argsort()[::-1][:top_n]

        keywords = [
            (feature_names[i], scores[i])
            for i in top_indices
            if scores[i] > 0
        ]

        results.append(keywords)

    return results


def calculate_similarity(texts):

    normalized = [
        normalize_text_for_comparison(text)
        for text in texts
    ]

    vectorizer = CountVectorizer()

    matrix = vectorizer.fit_transform(normalized)

    return cosine_similarity(matrix)


def get_shared_words(texts):

    normalized_sets = []

    for text in texts:

        doc = nlp(text)

        words = {
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and not token.is_stop
        }

        normalized_sets.append(words)

    if not normalized_sets:
        return []

    return sorted(
        set.intersection(*normalized_sets)
    )


def get_unique_words(texts):

    normalized_sets = []

    for text in texts:

        doc = nlp(text)

        words = {
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and not token.is_stop
        }

        normalized_sets.append(words)

    unique_results = []

    for index, current_set in enumerate(normalized_sets):

        other_words = set()

        for other_index, other_set in enumerate(normalized_sets):

            if other_index != index:
                other_words.update(other_set)

        unique_results.append(
            sorted(current_set - other_words)
        )

    return unique_results


def vocabulary_overlap(texts):

    normalized_sets = []

    for text in texts:

        doc = nlp(text)

        words = {
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and not token.is_stop
        }

        normalized_sets.append(words)

    results = []

    for i in range(len(normalized_sets)):

        for j in range(i + 1, len(normalized_sets)):

            intersection = (
                normalized_sets[i]
                &
                normalized_sets[j]
            )

            union = (
                normalized_sets[i]
                |
                normalized_sets[j]
            )

            overlap = (
                len(intersection) / len(union)
                if union else 0
            )

            results.append({
                "Text pair": f"Text {i + 1} ↔ Text {j + 1}",
                "Shared vocabulary": len(intersection),
                "Vocabulary overlap": overlap
            })

    return results


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="github-credit">
        by
        <a href="https://github.com/kisarrrrav" target="_blank">
            kisarrrrav (V)
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">Text Analysis Studio ♡</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Explore words, patterns and relationships hidden inside your texts.</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="decorative-line">✦　♡　✿　⌁　✦</div>',
    unsafe_allow_html=True
)


# ============================================================
# INTRO CARDS
# ============================================================

intro_cols = st.columns(3)

intro_cards = [
    (
        "✦",
        "Explore",
        "Break a text down into words, sentences, lemmas and parts of speech."
    ),
    (
        "♡",
        "Compare",
        "Find shared vocabulary, unique words and similarities between texts."
    ),
    (
        "✿",
        "Discover",
        "Use TF-IDF to identify words that are especially characteristic of each text."
    )
]

for col, (icon, title, description) in zip(
    intro_cols,
    intro_cards
):

    with col:

        st.markdown(
            f"""
            <div class="intro-card">
                <div class="intro-icon">{icon}</div>
                <div class="intro-title">{title}</div>
                <div class="intro-text">{description}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# SINGLE TEXT ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">✦ Single Text Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">Paste an English text and explore its structure, vocabulary and linguistic patterns.</div>',
    unsafe_allow_html=True
)

single_text = st.text_area(
    "Text",
    height=210,
    key="single_text",
    placeholder="Enter your text here...",
    label_visibility="collapsed"
)

single_button_left, single_button_right = st.columns(
    [5, 1.3]
)

with single_button_right:

    analyze_button = st.button(
        "✦ Analyze text",
        type="primary",
        key="analyze_single"
    )


if analyze_button:

    if single_text.strip():

        st.session_state.single_analysis_text = single_text

    else:

        st.warning(
            "Please enter some text before analyzing."
        )


# ============================================================
# SINGLE TEXT RESULTS
# ============================================================

if "single_analysis_text" in st.session_state:

    analyzed_text = st.session_state.single_analysis_text

    if analyzed_text.strip():

        results = analyze_text(analyzed_text)

        st.markdown("### Overview")

        metric_cols = st.columns(7)

        metrics = [
            ("Characters", results["characters"]),
            ("Words", results["words"]),
            ("Sentences", results["sentences"]),
            ("Content words", results["content_words"]),
            ("Unique words", results["unique_words"]),
            ("Lexical diversity", f'{results["lexical_diversity"]:.2f}'),
            ("Avg. sentence", f'{results["avg_sentence_length"]:.1f}')
        ]

        for col, (label, value) in zip(
            metric_cols,
            metrics
        ):

            with col:

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("### Frequent content words")

        content_data = pd.DataFrame(
            results["content_frequency"].most_common(10),
            columns=["Word", "Frequency"]
        )

        st.dataframe(
            content_data,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Lemmas")

        lemma_data = pd.DataFrame(
            results["lemma_frequency"].most_common(12),
            columns=["Lemma", "Frequency"]
        )

        st.dataframe(
            lemma_data,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Parts of speech")

        pos_df = pd.DataFrame(
            results["pos_frequency"].most_common(),
            columns=["POS", "Frequency"]
        )

        fig, ax = plt.subplots(
            figsize=(8, 4.5)
        )

        ax.bar(
            pos_df["POS"],
            pos_df["Frequency"],
            color="#C98598"
        )

        ax.set_ylabel("Frequency")
        ax.set_xlabel("Part of speech")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_color("#E1CDD3")
        ax.spines["bottom"].set_color("#E1CDD3")

        ax.tick_params(
            colors="#806B72"
        )

        ax.set_facecolor("#FFFDFC")
        fig.patch.set_facecolor("#FFFDFC")

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-box-title">
                    ♡ Lexical diversity
                </div>

                <div class="result-box-text">
                    Lexical diversity measures how varied the vocabulary is.
                    A value closer to 1 means that a larger proportion of the
                    words are unique.

                    This text has a lexical diversity of
                    <b>{results["lexical_diversity"]:.2f}</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# TEXT COMPARISON
# ============================================================

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">✦ TF-IDF & Text Comparison</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">Compare vocabulary, similarity and characteristic words across multiple texts.</div>',
    unsafe_allow_html=True
)


# ============================================================
# COMPARISON STATE
# ============================================================

if "comparison_ids" not in st.session_state:

    st.session_state.comparison_ids = [1, 2]


if "comparison_texts" not in st.session_state:

    st.session_state.comparison_texts = [
        "",
        ""
    ]


if "comparison_results" not in st.session_state:

    st.session_state.comparison_results = None


# ============================================================
# COMPARISON INPUTS
# ============================================================

texts_to_compare = []

for position, text_id in enumerate(
    st.session_state.comparison_ids
):

    st.markdown(
        f'<div class="text-label">Text {position + 1}</div>',
        unsafe_allow_html=True
    )

    text_value = st.text_area(
        f"Text {position + 1}",
        height=170,
        key=f"comparison_text_{text_id}",
        placeholder="Enter your text here...",
        label_visibility="collapsed"
    )

    texts_to_compare.append(text_value)

    # Remove button for additional texts
    if len(st.session_state.comparison_ids) > 2:

        remove_left, remove_right = st.columns(
            [1.2, 5]
        )

        with remove_left:

            if st.button(
                f"Remove Text {position + 1}",
                key=f"remove_text_{text_id}"
            ):

                st.session_state.comparison_ids.pop(
                    position
                )

                st.session_state.comparison_texts.pop(
                    position
                )

                st.session_state.comparison_results = None

                st.rerun()


st.session_state.comparison_texts = texts_to_compare


# ============================================================
# COMPARISON BUTTONS
# ============================================================

empty_col, add_col, compare_col = st.columns(
    [3.6, 1.3, 1.5]
)

with add_col:

    if len(st.session_state.comparison_ids) < 10:

        if st.button(
            "＋ Add text",
            key="add_comparison_text"
        ):

            new_id = max(
                st.session_state.comparison_ids,
                default=0
            ) + 1

            st.session_state.comparison_ids.append(
                new_id
            )

            st.session_state.comparison_texts.append(
                ""
            )

            st.session_state.comparison_results = None

            st.rerun()


with compare_col:

    compare_button = st.button(
        "✦ Compare texts",
        type="primary",
        key="compare_texts"
    )


# ============================================================
# RUN COMPARISON
# ============================================================

if compare_button:

    valid_texts = [
        text.strip()
        for text in st.session_state.comparison_texts
        if text.strip()
    ]

    if len(valid_texts) >= 2:

        st.session_state.comparison_results = valid_texts

    else:

        st.warning(
            "Please enter at least two texts before comparing them."
        )


# ============================================================
# COMPARISON RESULTS
# ============================================================

if st.session_state.comparison_results:

    texts = st.session_state.comparison_results

    shared_words = get_shared_words(texts)

    unique_words = get_unique_words(texts)

    overlap_data = vocabulary_overlap(texts)


    # ========================================================
    # OVERVIEW
    # ========================================================

    st.markdown("### Comparison overview")

    overview_cols = st.columns(3)

    with overview_cols[0]:

        st.markdown(
            f"""
            <div class="comparison-card">
                <div class="comparison-number">
                    {len(texts)}
                </div>

                <div class="comparison-label">
                    Texts compared
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with overview_cols[1]:

        st.markdown(
            f"""
            <div class="comparison-card">
                <div class="comparison-number">
                    {len(shared_words)}
                </div>

                <div class="comparison-label">
                    Shared words
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with overview_cols[2]:

        average_overlap = (
            sum(
                item["Vocabulary overlap"]
                for item in overlap_data
            )
            / len(overlap_data)
            if overlap_data
            else 0
        )

        st.markdown(
            f"""
            <div class="comparison-card">
                <div class="comparison-number">
                    {average_overlap:.0%}
                </div>

                <div class="comparison-label">
                    Average vocabulary overlap
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # SHARED VOCABULARY
    # ========================================================

    st.markdown("### Shared vocabulary")

    if shared_words:

        shared_html = "".join(
            f'<span class="word-pill">{word}</span>'
            for word in shared_words
        )

        st.markdown(
            shared_html,
            unsafe_allow_html=True
        )

    else:

        st.info(
            "No shared content words were found."
        )


    # ========================================================
    # UNIQUE VOCABULARY
    # ========================================================

    st.markdown("### Unique vocabulary")

    for index, words in enumerate(
        unique_words
    ):

        st.markdown(
            f"**Text {index + 1}**"
        )

        if words:

            unique_html = "".join(
                f'<span class="word-pill">{word}</span>'
                for word in words
            )

            st.markdown(
                unique_html,
                unsafe_allow_html=True
            )

        else:

            st.info(
                "No unique content words."
            )


    # ========================================================
    # VOCABULARY OVERLAP
    # ========================================================

    st.markdown("### Vocabulary overlap")

    overlap_df = pd.DataFrame(
        overlap_data
    )

    if not overlap_df.empty:

        overlap_df["Vocabulary overlap"] = (
            overlap_df["Vocabulary overlap"]
            .map(lambda x: f"{x:.1%}")
        )

        st.dataframe(
            overlap_df,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    st.markdown("### Cosine similarity")

    similarity = calculate_similarity(
        texts
    )

    similarity_labels = [
        f"Text {i + 1}"
        for i in range(len(texts))
    ]

    similarity_df = pd.DataFrame(
        similarity,
        index=similarity_labels,
        columns=similarity_labels
    )

    similarity_df = similarity_df.round(2)

    st.dataframe(
        similarity_df,
        use_container_width=True
    )


    # ========================================================
    # TF-IDF
    # ========================================================

    st.markdown("### TF-IDF keywords")

    tfidf_results = get_tfidf_keywords(
        texts
    )

    for index, keywords in enumerate(
        tfidf_results
    ):

        st.markdown(
            f"**Text {index + 1}**"
        )

        if keywords:

            tfidf_df = pd.DataFrame(
                keywords,
                columns=["Word", "TF-IDF"]
            )

            tfidf_df["TF-IDF"] = (
                tfidf_df["TF-IDF"]
                .round(3)
            )

            st.dataframe(
                tfidf_df,
                use_container_width=True,
                hide_index=True
            )


            # --------------------------------------------
            # TF-IDF CHART
            # --------------------------------------------

            chart_df = pd.DataFrame(
                keywords,
                columns=["Word", "TF-IDF"]
            )

            chart_df = chart_df.sort_values(
                "TF-IDF",
                ascending=True
            )

            fig, ax = plt.subplots(
                figsize=(8, 4.5)
            )

            ax.barh(
                chart_df["Word"],
                chart_df["TF-IDF"],
                color="#C98598"
            )

            ax.set_xlabel(
                "TF-IDF score"
            )

            ax.set_title(
                f"Text {index + 1}: characteristic words",
                color="#4B3038",
                fontweight="600"
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.spines["left"].set_color("#E1CDD3")
            ax.spines["bottom"].set_color("#E1CDD3")

            ax.tick_params(
                colors="#806B72"
            )

            ax.set_facecolor("#FFFDFC")
            fig.patch.set_facecolor("#FFFDFC")

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


# ============================================================
# FOOTER
# ============================================================

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="footer">
        <div class="footer-symbols">
            ✦　♡　✿　⌁　✦
        </div>
        <div>
            Made with curiosity ✦
        </div>
    </div>
    """,
    unsafe_allow_html=True
)