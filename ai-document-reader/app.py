"""AI Document Reader - Part 1 (TF-IDF baseline). Run: python -m streamlit run app.py

Model evaluation is done in code (training/train_language_detection.py -> results/),
not in this interface.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from src.extraction import SUPPORTED_TYPES, UnsupportedFileTypeError, extract_text
from src.language_detection import LanguageDetector, ModelNotTrainedError
from src.preprocessing import preprocess
from src.question_answering import DocumentRetriever
from src.summarization import (
    ABSTRACTIVE_MODEL_NAME, AbstractiveUnavailableError, abstractive_summary, extractive_summary,
)
from src.tts import ENGINE_LABELS, TTSUnavailableError, engines_for, synthesize

st.set_page_config(page_title="AI Document Reader", page_icon="📄", layout="centered")

LANGUAGES = ["english", "french", "kinyarwanda"]
SPEEDS = {"0.75x": 0.75, "1x": 1.0, "1.25x": 1.25, "1.5x": 1.5}
READ_EXCERPT_CHARS = 3000

st.markdown(
    """
    <style>
      [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }
      .block-container { max-width: 960px; padding-top: 3rem; }
      .app-title { text-align: center; font-size: 2.1rem; font-weight: 800;
                   letter-spacing: 0.12em; margin: 0; }
      .app-subtitle { text-align: center; color: #6b6b6b; margin: 0.4rem 0 0.2rem; }
      .info-label { color: #6b6b6b; font-size: 0.82rem; margin-bottom: 0.25rem; }
      .info-value { font-weight: 600; font-size: 1.02rem; overflow-wrap: anywhere; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------- cached ----
@st.cache_resource
def load_detector() -> LanguageDetector:
    return LanguageDetector.load()


@st.cache_data(show_spinner=False)
def cached_extract(data: bytes, filename: str):
    return extract_text(data, filename)


@st.cache_resource(show_spinner=False)
def cached_retriever(text: str) -> DocumentRetriever:
    return DocumentRetriever(text)


def human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 ** 2:.2f} MB"


def info_cell(column, label: str, value: str) -> None:
    column.markdown(
        f'<div class="info-label">{html.escape(label)}</div>'
        f'<div class="info-value">{html.escape(value)}</div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------- header ----
st.markdown('<div class="app-title">AI DOCUMENT READER</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Upload · Understand · Summarize · Ask · Listen</div>',
            unsafe_allow_html=True)
st.divider()

st.subheader("Upload your document")
uploaded = st.file_uploader(
    "Drag & Drop PDF / DOCX / TXT", type=list(SUPPORTED_TYPES),
    help="PDF is read with PyMuPDF, DOCX with python-docx, TXT as UTF-8 text.",
)

try:
    detector = load_detector()
except ModelNotTrainedError as exc:
    st.error(f"**Language detection model not found.**\n\n```\n{exc}\n```")
    st.info("Build the dataset and train the model first:\n\n"
            "```bash\npython training/prepare_dataset.py\npython training/train_language_detection.py\n```")
    st.stop()

if uploaded is None:
    st.stop()

# A new upload must not show the previous document's summary or audio.
if st.session_state.get("file_id") != uploaded.file_id:
    st.session_state["file_id"] = uploaded.file_id
    st.session_state.pop("summary", None)
    st.session_state.pop("speech", None)

# --------------------------------------------------------- processing ----
with st.status("Processing document...", expanded=False) as status:
    try:
        doc = cached_extract(uploaded.getvalue(), uploaded.name)
    except UnsupportedFileTypeError as exc:
        status.update(label="Unsupported file", state="error")
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # corrupted files, missing libraries, ...
        status.update(label="Could not read the document", state="error")
        st.error(f"Could not read **{uploaded.name}**: {exc}")
        st.stop()
    st.write(f"Text extracted: {len(doc.text):,} characters")
    if doc.is_empty:
        status.update(label="No text found", state="error")
        for warning in doc.warnings:
            st.warning(warning)
        st.stop()

    cleaned, tokens, stats = preprocess(doc.text)
    st.write(f"Text cleaned and tokenized: {stats.tokens:,} tokens")
    prediction = detector.predict(doc.text)
    language = prediction.language
    st.write(f"Language detected: {language.capitalize()}")
    status.update(label="Ready", state="complete")

for warning in doc.warnings:
    st.warning(warning)

# --------------------------------------------------- document info card ----
with st.container(border=True):
    st.markdown("**Document Information**")
    cols = st.columns([3.2, 0.8, 1, 0.8, 1.2, 1.2])
    info_cell(cols[0], "File", doc.filename)
    info_cell(cols[1], "Type", doc.file_type.upper())
    info_cell(cols[2], "Size", human_size(doc.size_bytes))
    info_cell(cols[3], "Pages", str(doc.num_pages) if doc.num_pages is not None else "n/a")
    info_cell(cols[4], "Language", language.capitalize())
    info_cell(cols[5], "Confidence",
              f"{prediction.confidence:.0%}" if prediction.confidence is not None else "n/a")

tab_doc, tab_sum, tab_qa, tab_read = st.tabs(["📄 Document", "📝 Summary", "❓ Ask AI", "🔊 Read"])

# ----------------------------------------------------------- Document ----
with tab_doc:
    st.text_area("Extracted text", doc.text, height=380)

    left, right = st.columns(2)
    with left:
        st.markdown("**Language detection**")
        st.markdown(
            f"Detected language: **{language.capitalize()}**  \n"
            f"Model: TF-IDF ({detector.metadata['feature_config']}) + {prediction.model_name}  \n"
            f"Characters analysed: {prediction.characters_used:,}"
        )
        if prediction.probabilities:
            probs = pd.Series(prediction.probabilities).rename(index=str.capitalize)
            st.dataframe(probs.to_frame("probability").style.format("{:.1%}"), width="stretch")
        with st.expander("What does confidence mean?"):
            st.markdown(
                "**Confidence** is the probability the model gives to the predicted language "
                "for *this document*. It is **not** the model's accuracy: accuracy is measured "
                "once, on a held-out test set, during training. A high confidence does not "
                "guarantee a correct answer, and the model can only choose between English, "
                "French and Kinyarwanda."
            )
    with right:
        st.markdown("**Preprocessing statistics**")
        st.dataframe(pd.DataFrame(
            {"value": [stats.original_characters, stats.cleaned_characters, stats.removed_characters,
                       stats.tokens, stats.unique_tokens]},
            index=["Original characters", "Cleaned characters", "Removed characters",
                   "Tokens", "Unique tokens"]), width="stretch")

# ------------------------------------------------------------ Summary ----
with tab_sum:
    methods = ["Extractive (all languages)"]
    if language == "english":
        methods.append("Abstractive (English, pretrained)")
    c1, c2 = st.columns([2, 1])
    method = c1.radio("Method", methods, horizontal=True)
    n_sent = c2.selectbox("Length (sentences)", [3, 5, 7, 10], index=1,
                          disabled=method.startswith("Abstractive"))
    if st.button("Generate Summary", type="primary"):
        result = None
        if method.startswith("Abstractive"):
            try:
                with st.spinner("Summarising (the first run downloads the model)..."):
                    result = abstractive_summary(doc.text, language)
            except AbstractiveUnavailableError as exc:
                st.warning(f"Abstractive model unavailable ({exc}). Showing the extractive summary instead.")
        if result is None:
            result = extractive_summary(doc.text, n_sent, language)
        st.session_state["summary"] = result

    result = st.session_state.get("summary")
    if result is not None:
        if not result.summary:
            st.warning("The document is too short to summarise.")
        else:
            with st.container(border=True):
                if result.method == "extractive":
                    for sentence in result.sentences:
                        st.markdown(f"- {sentence}")
                else:
                    st.write(result.summary)
            if result.method == "extractive":
                st.caption("Extractive summary: the most representative sentences, copied word-for-word "
                           "from the document (frequency-based sentence scoring).")
            else:
                st.caption(f"Abstractive summary generated by the pretrained model {ABSTRACTIVE_MODEL_NAME}.")

# ------------------------------------------------------------- Ask AI ----
with tab_qa:
    retriever = cached_retriever(doc.text)
    question = st.text_input("Ask a question about the document",
                             placeholder="e.g. What is this document about?")
    if question:
        answer = retriever.answer(question)
        if answer.answer is None:
            st.warning("I could not find a passage in the document related to this question. "
                       "Try using words that appear in the document.")
        else:
            st.success(answer.answer)
        if answer.passages:
            best = answer.passages[0]
            with st.expander(f"Source passage (similarity {best.score:.2f})", expanded=answer.answer is not None):
                st.write(best.text)
            if len(answer.passages) > 1:
                with st.expander("Other related passages"):
                    for passage in answer.passages[1:]:
                        st.caption(f"Similarity {passage.score:.2f}")
                        st.write(passage.text)
    st.caption("Answers are sentences retrieved from the document (TF-IDF passage search), not generated text.")

# --------------------------------------------------------------- Read ----
with tab_read:
    c1, c2 = st.columns([1, 2])
    speed_label = c1.selectbox("Reading Speed", list(SPEEDS), index=1)
    source = c2.radio("Read", ["Full document (excerpt)", "Summary (if generated)", "Custom text"],
                      horizontal=True)

    if source.startswith("Full"):
        text_to_read = doc.text[:READ_EXCERPT_CHARS]
    elif source.startswith("Summary"):
        summary = st.session_state.get("summary")
        text_to_read = summary.summary if summary is not None else ""
        if not text_to_read:
            st.info("Generate a summary in the Summary tab first.")
    else:
        text_to_read = st.text_area("Text to read", placeholder="Type or paste text here")

    with st.expander("Voice options"):
        tts_language = st.selectbox("Reading language", LANGUAGES, index=LANGUAGES.index(language),
                                    format_func=str.capitalize)
        engine = st.selectbox("Voice engine", engines_for(tts_language), format_func=ENGINE_LABELS.get)
    if tts_language == "kinyarwanda":
        st.caption("ℹ️ Kinyarwanda speech support is limited: it uses Meta's pretrained "
                   "facebook/mms-tts-kin model (downloaded on first use). No voice was trained in this project.")

    if st.button("► Generate & Play Audio", type="primary", disabled=not text_to_read.strip()):
        try:
            with st.spinner("Generating audio..."):
                st.session_state["speech"] = synthesize(text_to_read, tts_language, engine, SPEEDS[speed_label])
        except TTSUnavailableError as exc:
            st.session_state.pop("speech", None)
            st.error(str(exc))

    speech = st.session_state.get("speech")
    if speech is not None:
        st.audio(speech.audio, format=speech.mime, autoplay=True)
        if speech.note:
            st.caption(speech.note)
    st.caption("Streamlit's audio player supports play / pause / seek. Choose a speed, then generate audio.")
