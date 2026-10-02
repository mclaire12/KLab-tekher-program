"""AI Document Reader - Part 1 (TF-IDF baseline). Run: streamlit run app.py"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import RESULTS_DIR
from src.extraction import SUPPORTED_TYPES, UnsupportedFileTypeError, extract_text
from src.language_detection import LanguageDetector, ModelNotTrainedError
from src.preprocessing import preprocess
from src.question_answering import DocumentRetriever
from src.summarization import (
    ABSTRACTIVE_MODEL_NAME, AbstractiveUnavailableError, abstractive_summary, extractive_summary,
)
from src.tts import ENGINE_LABELS, TTSUnavailableError, engines_for, synthesize

st.set_page_config(page_title="AI Document Reader", page_icon="📄", layout="wide")


# ------------------------------------------------------------- cached ----
@st.cache_resource
def load_detector() -> LanguageDetector:
    return LanguageDetector.load()


@st.cache_data(show_spinner="Extracting text...")
def cached_extract(data: bytes, filename: str):
    return extract_text(data, filename)


@st.cache_resource(show_spinner="Indexing document for questions...")
def cached_retriever(text: str) -> DocumentRetriever:
    return DocumentRetriever(text)


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


# ------------------------------------------------------------- header ----
st.title("📄 AI Document Reader")
st.caption("Part 1 — TF-IDF baseline · upload a document to extract, analyse, summarise, query and listen to it.")

try:
    detector = load_detector()
except ModelNotTrainedError as exc:
    st.error(f"**Language detection model not found.**\n\n```\n{exc}\n```")
    st.info("Build the dataset and train the model first:\n\n"
            "```bash\npython training/prepare_dataset.py\npython training/train_language_detection.py\n```")
    st.stop()

meta = detector.metadata

with st.sidebar:
    st.header("Upload Document")
    uploaded = st.file_uploader("PDF, DOCX or TXT", type=list(SUPPORTED_TYPES))
    st.divider()
    st.subheader("Language model")
    st.markdown(
        f"- **Representation:** {meta['feature_config']}\n"
        f"- **Classifier:** {meta['selected_classifier']}\n"
        f"- **Languages:** {', '.join(l.capitalize() for l in meta['supported_languages'])}\n"
        f"- **Test macro F1:** {meta['test_metrics']['f1_macro']:.4f}\n"
        f"- **Version:** {meta['model_version']}"
    )

if uploaded is None:
    st.info("⬅️ Upload a PDF, DOCX or TXT file to begin.")
    tab_model, = st.tabs(["📊 Model evaluation"])
    show_doc = False
else:
    show_doc = True

# ----------------------------------------------------------- document ----
if show_doc:
    try:
        doc = cached_extract(uploaded.getvalue(), uploaded.name)
    except UnsupportedFileTypeError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # corrupted files etc.
        st.error(f"Could not read **{uploaded.name}**: {exc}")
        st.stop()

    for warning in doc.warnings:
        st.warning(warning)
    if doc.is_empty:
        st.stop()

    cleaned, tokens, stats = preprocess(doc.text)
    try:
        prediction = detector.predict(doc.text)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    language = prediction.language

    st.subheader("Document Information")
    c = st.columns(6)
    c[0].metric("File", doc.filename if len(doc.filename) < 22 else doc.filename[:19] + "…")
    c[1].metric("Type", doc.file_type.upper())
    c[2].metric("Size", human_size(doc.size_bytes))
    c[3].metric("Pages", doc.num_pages if doc.num_pages is not None else "n/a")
    c[4].metric("Characters", f"{len(doc.text):,}")
    c[5].metric("Detected language", language.capitalize())
    if prediction.confidence is not None:
        st.caption(f"Model confidence for this document: **{prediction.confidence:.1%}** "
                   "(see the Document tab for what this means).")
    else:
        st.caption(f"{prediction.model_name} does not output probabilities; decision scores are shown in the Document tab.")

    tab_doc, tab_sum, tab_qa, tab_read, tab_model = st.tabs(
        ["📄 Document", "📝 Summary", "💬 Ask AI", "🔊 Read", "📊 Model evaluation"]
    )

    # ------------------------------------------------------- Document ----
    with tab_doc:
        left, right = st.columns([3, 2])
        with left:
            st.markdown("#### Extracted text")
            st.text_area("Extracted text", doc.text, height=480, label_visibility="collapsed")
        with right:
            st.markdown("#### Language detection")
            st.markdown(
                f"**Detected language:** {language.capitalize()}  \n"
                f"**Model:** TF-IDF ({meta['feature_config']}) + {prediction.model_name}  \n"
                f"**Characters analysed:** {prediction.characters_used:,}"
            )
            if prediction.probabilities:
                st.markdown(f"**Confidence:** {prediction.confidence:.1%}")
                st.bar_chart(pd.Series(prediction.probabilities, name="probability"), horizontal=True)
            elif prediction.decision_scores:
                st.markdown("**Confidence:** not available — this classifier does not estimate probabilities.")
                st.dataframe(
                    pd.DataFrame({"decision score": prediction.decision_scores}).round(3),
                    width="stretch",
                )
                st.caption("Decision scores are signed distances from each class's separating "
                           "hyperplane. The highest score wins. They are not probabilities.")
            with st.expander("Confidence vs. accuracy — what is the difference?"):
                st.markdown(
                    "- **Confidence** (or decision score) describes how strongly the model "
                    "prefers one language for *this particular document*.\n"
                    "- **Accuracy / F1** describe how often the model was right on a *held-out test "
                    f"set* of {meta['dataset']['test_rows']:,} labelled sentences "
                    f"(test macro F1 = {meta['test_metrics']['f1_macro']:.4f}).\n"
                    "- A high confidence does not guarantee a correct answer, and the model can only "
                    "choose between English, French and Kinyarwanda — a document in any other "
                    "language will still be assigned one of these three."
                )

            st.markdown("#### Preprocessing statistics")
            st.dataframe(pd.DataFrame(
                {"value": [stats.original_characters, stats.cleaned_characters, stats.removed_characters,
                           stats.tokens, stats.unique_tokens]},
                index=["original characters", "cleaned characters", "removed characters",
                       "tokens", "unique tokens"]), width="stretch")
            with st.expander("Cleaned text sent to the model (first 1,000 characters)"):
                st.code(cleaned[:1000] or "(empty)", language=None)

    # -------------------------------------------------------- Summary ----
    with tab_sum:
        st.markdown("#### Summary")
        st.caption("Summarisation is a separate task from language detection: it does **not** use the "
                   "trained TF-IDF language classifier.")
        methods = ["Extractive (frequency-based, all languages)"]
        if language == "english":
            methods.append(f"Abstractive (pretrained {ABSTRACTIVE_MODEL_NAME}, English only)")
        method = st.radio("Method", methods, horizontal=True)
        n_sent = st.slider("Sentences in extractive summary", 2, 12, 5)
        if st.button("Generate summary", type="primary"):
            result = None
            if method.startswith("Abstractive"):
                try:
                    with st.spinner(f"Running {ABSTRACTIVE_MODEL_NAME} (first run downloads the model)..."):
                        result = abstractive_summary(doc.text, language)
                except AbstractiveUnavailableError as exc:
                    st.warning(f"Abstractive model unavailable: {exc}\n\n"
                               "**Falling back to the extractive summary below.**")
            if result is None:
                result = extractive_summary(doc.text, n_sent, language)
            st.session_state["summary"] = result

        result = st.session_state.get("summary")
        if result is not None:
            label = ("EXTRACTIVE summary — sentences copied verbatim from the document"
                     if result.method == "extractive"
                     else f"ABSTRACTIVE summary — generated by pretrained model {result.model_name}")
            st.info(f"**{label}**  \nModel/method: {result.model_name}")
            if not result.summary:
                st.warning("The document is too short to summarise.")
            elif result.method == "extractive":
                for i, s in zip(result.sentence_indices, result.sentences):
                    st.markdown(f"- {s}  \n  <small>sentence #{i + 1}</small>", unsafe_allow_html=True)
            else:
                st.write(result.summary)

    # --------------------------------------------------------- Ask AI ----
    with tab_qa:
        st.markdown("#### Ask a question about the document")
        st.caption("Method: **retrieval-based, extractive** question answering. The document is split into "
                   "overlapping passages, indexed with TF-IDF, and the passage most similar to your "
                   "question (cosine similarity) is returned. The answer is a sentence copied from the "
                   "document — this is not a generative AI model.")
        retriever = cached_retriever(doc.text)
        question = st.text_input("Question", placeholder="e.g. Who is responsible for the project?")
        if question:
            answer = retriever.answer(question)
            if answer.answer is None:
                st.warning("No passage in the document is sufficiently related to this question "
                           f"(best similarity {answer.score:.3f}). Try different keywords.")
            else:
                st.success(f"**Answer:** {answer.answer}")
                st.caption(f"Similarity score of the source passage: {answer.score:.3f} (0 = unrelated, 1 = identical wording)")
            for rank, passage in enumerate(answer.passages, 1):
                with st.expander(f"Source passage {rank} · similarity {passage.score:.3f} · chunk #{passage.chunk_index + 1}",
                                 expanded=rank == 1 and answer.answer is not None):
                    st.write(passage.text)

    # ----------------------------------------------------------- Read ----
    with tab_read:
        st.markdown("#### Read the document aloud")
        tts_language = st.selectbox(
            "Reading language", ["english", "french", "kinyarwanda"],
            index=["english", "french", "kinyarwanda"].index(language),
            format_func=str.capitalize, help="Defaults to the detected language.",
        )
        if tts_language == "kinyarwanda":
            st.warning("**Kinyarwanda TTS support is limited.** Google TTS and standard operating-system "
                       "voices have no Kinyarwanda voice. This app uses Meta's *pretrained* MMS model "
                       "`facebook/mms-tts-kin` (needs `transformers` + `torch`, downloaded on first use). "
                       "No Kinyarwanda TTS model was trained in this project.")
        engines = engines_for(tts_language)
        engine = st.selectbox("Voice engine", engines, format_func=ENGINE_LABELS.get)
        speed = st.slider("Reading speed", 0.5, 2.0, 1.0, 0.1,
                          help="gTTS supports only normal or slow (< 1.0). pyttsx3 and MMS support the full range.")
        max_chars = min(len(doc.text), 5000)
        start, end = st.slider("Characters to read", 0, len(doc.text), (0, max_chars), step=50,
                               help="Long passages take longer to synthesise.")
        if st.button("🔊 Generate audio", type="primary"):
            try:
                with st.spinner(f"Synthesising speech with {ENGINE_LABELS[engine]}..."):
                    speech = synthesize(doc.text[start:end], tts_language, engine, speed)
                st.session_state["speech"] = speech
            except TTSUnavailableError as exc:
                st.session_state.pop("speech", None)
                st.error(str(exc))
        speech = st.session_state.get("speech")
        if speech is not None:
            st.audio(speech.audio, format=speech.mime)
            st.caption(f"Engine: {ENGINE_LABELS[speech.engine]}" + (f" · {speech.note}" if speech.note else ""))

# ------------------------------------------------------ Model evaluation ----
with tab_model:
    st.markdown("#### Language detection — evaluation results")
    st.caption("All numbers below are computed by `training/train_language_detection.py` on the real dataset. "
               "Model selection used the validation split; the test split was only used for reporting.")
    results_csv = RESULTS_DIR / "language_detection_results.csv"
    if not results_csv.exists():
        st.warning("No results yet. Run `python training/train_language_detection.py`.")
    else:
        res = pd.read_csv(results_csv)
        table = res.pivot_table(index=["features", "classifier"], columns="split",
                                values=["accuracy", "precision_macro", "recall_macro", "f1_macro"])
        table.columns = [f"{split} {metric}" for metric, split in table.columns]
        st.dataframe(table.sort_values("validation f1_macro", ascending=False).round(4), width="stretch")
        st.markdown(f"**Selected:** {meta['selected_classifier']} + {meta['feature_config']} "
                    f"— selection rule: {meta['selection_metric']}.")
        for image, caption in [("model_comparison.png", "Validation macro F1 by representation and classifier"),
                               ("short_text_robustness.png", "Robustness to short inputs (test set)"),
                               ("confusion_matrix.png", "Confusion matrix of the selected model (test set)")]:
            if (RESULTS_DIR / image).exists():
                st.image(str(RESULTS_DIR / image), caption=caption)
