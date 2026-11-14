"""
Streamlit Web Dashboard for Hinglish Aspect-Based Sentiment Analysis (ABSA).
Interactive UI with real-time analysis, presets, batch mode, benchmark evaluation,
Hinglish slang explorer, and plug-and-play API key integration.
"""

import streamlit as st
import pandas as pd
import json
import os
from hinglish_absa import HinglishABSA
from hinglish_absa.lexicons import POSITIVE_LEXICON, NEGATIVE_LEXICON, NEUTRAL_LEXICON, ASPECT_CATEGORIES

# Page configuration
st.set_page_config(
    page_title="Hinglish Aspect-Based Sentiment Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B649);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .badge-positive {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-negative {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-neutral {
        background-color: #e2e3e5;
        color: #383d41;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .category-chip {
        background-color: #e3f2fd;
        color: #0d47a1;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-pill-offline {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #a5d6a7;
        padding: 6px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-pill-online {
        background-color: #fff8e1;
        color: #f57f17;
        border: 1px solid #ffe082;
        padding: 6px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Cache local engine
@st.cache_resource
def load_absa_engine():
    return HinglishABSA()

absa = load_absa_engine()

# Sidebar: Configuration & API Key Integration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.markdown("### 🧠 Hinglish ABSA Engine")
    st.markdown(
        "Aspect-Based Sentiment Analysis across **Pure English**, **Pure Hindi (Devanagari)**, "
        "and **Hinglish (Code-Mixed)** with deep slang handling."
    )
    st.markdown("---")

    st.markdown("#### 🔑 Engine & API Settings")
    api_provider = st.selectbox("API Provider (Optional):", ["Gemini", "OpenAI"])
    user_api_key = st.text_input(
        f"{api_provider} API Key (Leave blank to use offline model):",
        type="password",
        placeholder="AIzaSy... / sk-...",
        help="The system works 100% offline using our calibrated ABSA-MIX ML model. Paste a key here only if you wish to toggle online LLM reasoning."
    )

    if user_api_key.strip():
        absa.set_api_key(user_api_key.strip(), provider=api_provider.lower())
        st.markdown('<div class="status-pill-online">⚡ Online API Mode Connected</div>', unsafe_allow_html=True)
        use_api_toggle = st.checkbox("Use API for Inference", value=True)
    else:
        env_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if env_key:
            absa.set_api_key(env_key, provider="gemini" if "AIza" in env_key else "openai")
            st.markdown('<div class="status-pill-online">⚡ Environment API Key Active</div>', unsafe_allow_html=True)
            use_api_toggle = st.checkbox("Use API for Inference", value=False)
        else:
            st.markdown('<div class="status-pill-offline">🟢 Local Offline Hybrid Engine Active</div>', unsafe_allow_html=True)
            use_api_toggle = False

    st.markdown("---")
    st.markdown("**Core Capabilities:**")
    st.markdown("- 🔍 Multi-Domain Aspect Target Extraction")
    st.markdown("- 🎭 Slang Dictionary (*OP, paisa vasool, bakwas, jhakaas*)")
    st.markdown("- 🔀 Syntactic Contrastive Clauses (`but`, `lekin`, `par`)")
    st.markdown("- ⚖️ Balanced Neutral & Factual Calibration")
    st.markdown("- 📄 Single Word to Long Paragraph Scalability")
    st.markdown("---")
    st.caption("Google DeepMind / Antigravity AI • Python 3.14 & Scikit-Learn")

# Header
st.markdown('<div class="main-title">🎯 Hinglish Aspect-Based Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Extract fine-grained aspect targets and classify sentiments in English, Hindi (देवनागरी), and Hinglish (Code-Mixed).</div>',
    unsafe_allow_html=True
)

tabs = st.tabs([
    "🚀 Live Interactive Analyzer",
    "📁 Batch & Paragraph Mode",
    "📊 ABSA-MIX & Benchmark",
    "📖 Hinglish Slang Explorer",
    "📘 API & Documentation"
])

# TAB 1: Live Interactive Analyzer
with tabs[0]:
    presets = {
        "Custom Input...": "",
        # User prompt English examples
        "English: Positive ('I love using this product, it is amazing!')": "I love using this product, it is amazing!",
        "English: Neutral ('The restaurant was not terrible, it was not great either, just okay.')": "The restaurant was not terrible, it was not great either, just okay.",
        "English: Negative ('I hate these updates, everything is slow and broken.')": "I hate these updates, everything is slow and broken.",

        # User prompt Hindi Devanagari examples
        "Hindi: Positive ('यह फिल्म बहुत अच्छी है।')": "यह फिल्म बहुत अच्छी है।",
        "Hindi: Negative ('मुझे यह खाना पसंद नहीं आया।')": "मुझे यह खाना पसंद नहीं आया।",
        "Hindi: Neutral ('आज मौसम ठीक है।')": "आज मौसम ठीक है।",

        # User prompt Roman Hinglish examples
        "Hinglish: Positive ('Yeh movie bahut achhi hai.')": "Yeh movie bahut achhi hai.",
        "Hinglish: Negative ('Mujhe yeh khana bilkul pasand nahi aaya.')": "Mujhe yeh khana bilkul pasand nahi aaya.",
        "Hinglish: Positive ('Aaj mausam achha hai.')": "Aaj mausam achha hai.",
        "Hinglish: Neutral ('Maine naya mobile kharida.')": "Maine naya mobile kharida.",

        # Slang & Multi-Aspect Hinglish
        "Hinglish: Multi-Aspect ('Phone ka camera bahut accha hai but battery backup bekar hai.')": "Phone ka camera bahut accha hai but battery backup bekar hai.",
        "Hinglish: Slang ('Bhai phone ekdum OP hai, paisa vasool!')": "Bhai phone ekdum OP hai, paisa vasool!",
        "Hinglish: Tech ('Laptop ka performance awesome hai, lekin heating issue bahut hai.')": "Laptop ka performance awesome hai, lekin heating issue bahut hai.",
        "Hinglish: Entertainment ('Movie ki story mast thi but acting thodi boring thi.')": "Movie ki story mast thi but acting thodi boring thi.",

        # Single word inputs
        "Single Word: 'Bakwas'": "Bakwas",
        "Single Word: 'Mast'": "Mast",
        "Single Word: 'बकवास'": "बकवास",

        # Long Paragraph
        "Long Review: Multi-Sentence Paragraph": (
            "Maine kal naya smartphone buy kiya. Phone ka display aur build quality toh ekdum shaandar hai, dekh ke maza aa gaya! "
            "Lekin camera quality thodi disappointing hai, low light me photos bilkul saaf nahi aati. "
            "Battery backup theek-thaak hai, pura din chal jata hai par fast charging thodi slow lagti hai. "
            "Customer service walo ka behavior bilkul rude tha jab maine query puchi."
        ),
    }

    selected_preset = st.selectbox("Select a benchmark sample or enter custom text below:", list(presets.keys()))
    default_text = presets[selected_preset] if selected_preset != "Custom Input..." else "Phone ka camera bahut accha hai but battery backup bekar hai."

    input_text = st.text_area(
        "Enter text (English, Hindi, Hinglish code-mixed, single word, sentence, or paragraph):",
        value=default_text,
        height=110,
        placeholder="Type any word, sentence, or paragraph here..."
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        analyze_clicked = st.button("⚡ Analyze Sentiment", type="primary", use_container_width=True)

    if analyze_clicked or input_text:
        if input_text.strip():
            result = absa.analyze(input_text, use_api=use_api_toggle)

            st.markdown("---")

            # Document-level summary metrics
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            with m_col1:
                st.metric("Total Aspects", len(result.aspects))
            with m_col2:
                st.metric("Positive Aspects", result.aspect_breakdown["positive"])
            with m_col3:
                st.metric("Negative Aspects", result.aspect_breakdown["negative"])
            with m_col4:
                st.metric("Neutral Aspects", result.aspect_breakdown["neutral"])
            with m_col5:
                tone = result.overall_polarity.upper()
                st.metric("Overall Tone", tone)

            st.markdown("---")
            st.markdown("### 📋 Aspect-Based Findings")

            if not result.has_aspects:
                st.info("No explicit aspect targets detected. Sentence may be purely conversational or non-evaluative.")
            else:
                col_left, col_right = st.columns([3, 2])

                with col_left:
                    for a in result.aspects:
                        badge_class = f"badge-{a.polarity}"
                        icon = "✅" if a.polarity == "positive" else ("❌" if a.polarity == "negative" else "⚪")
                        border_color = "#28a745" if a.polarity == "positive" else ("#dc3545" if a.polarity == "negative" else "#6c757d")

                        st.markdown(
                            f"""
                            <div style="border-left: 5px solid {border_color}; background-color: rgba(240,240,240,0.15); padding: 14px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.05);">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                    <span style="font-size: 1.15rem; font-weight: 700;">{a.aspect.capitalize()}</span>
                                    <span class="{badge_class}">{icon} {a.polarity.upper()} ({a.confidence*100:.1f}%)</span>
                                </div>
                                <div style="margin-bottom: 6px;">
                                    <span class="category-chip">Category: {a.category}</span>
                                </div>
                                <div style="font-size: 0.9rem; color: #444; margin-top: 4px;">
                                    <strong>Evidence / Linguistic Reason:</strong> {", ".join(a.evidence) if a.evidence else "Model contextual inference"}
                                </div>
                                <div style="font-size: 0.85rem; color: #777; margin-top: 4px;">
                                    <strong>Context Scope:</strong> <em>"{a.clause_context}"</em>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                with col_right:
                    st.markdown("**Sentiment Distribution:**")
                    sent_counts = result.aspect_breakdown
                    df_chart = pd.DataFrame({
                        "Sentiment": ["Positive", "Negative", "Neutral"],
                        "Count": [sent_counts["positive"], sent_counts["negative"], sent_counts["neutral"]]
                    })
                    st.bar_chart(df_chart.set_index("Sentiment"), color="#4ECDC4")

                    # Probabilities breakdown table
                    st.markdown("**Model Class Probabilities:**")
                    prob_rows = []
                    for a in result.aspects:
                        prob_rows.append({
                            "Aspect": a.aspect,
                            "Positive": f"{a.probabilities.get('positive', 0)*100:.1f}%",
                            "Negative": f"{a.probabilities.get('negative', 0)*100:.1f}%",
                            "Neutral": f"{a.probabilities.get('neutral', 0)*100:.1f}%",
                        })
                    st.dataframe(pd.DataFrame(prob_rows), hide_index=True, use_container_width=True)

                with st.expander("🔍 Deep Pipeline Inspection & Raw Output"):
                    st.markdown(f"**Engine Used:** `{result.engine_source}`")
                    st.markdown(f"**Original Input:** `{result.raw_text}`")
                    st.markdown(f"**Normalized Text:** `{result.normalized_text}`")
                    st.json(result.to_dict())

# TAB 2: Batch Analysis
with tabs[1]:
    st.markdown("### 📁 Batch & Multi-Review Processing")
    st.markdown("Analyze multiple customer reviews, social media comments, or datasets in bulk.")

    batch_input_method = st.radio("Input source:", ["Paste multiple reviews (one per line)", "Upload CSV file"], horizontal=True)

    if batch_input_method == "Paste multiple reviews (one per line)":
        default_batch = (
            "Phone ka camera bahut accha hai but battery backup bekar hai.\n"
            "Movie ki story mast thi but acting thodi boring thi.\n"
            "Laptop ka performance awesome hai, lekin heating issue bahut hai.\n"
            "Bhai phone ekdum OP hai, paisa vasool!\n"
            "Food fresh aur tasty tha lekin seating arrangement ghatiya thi.\n"
            "Customer care walon ne turant issue resolve kiya, best service!\n"
            "The restaurant was not terrible, it was not great either, just okay.\n"
            "I hate these updates, everything is slow and broken.\n"
            "यह फिल्म बहुत अच्छी है।\n"
            "आज मौसम ठीक है।"
        )
        batch_text = st.text_area("Paste reviews (one per line):", value=default_batch, height=200)

        if st.button("🚀 Process Batch Reviews", type="primary"):
            lines = [l.strip() for l in batch_text.splitlines() if l.strip()]
            results = absa.analyze_batch(lines, use_api=use_api_toggle)

            flat_data = []
            for r in results:
                for a in r.aspects:
                    flat_data.append({
                        "Sentence": r.raw_text,
                        "Aspect": a.aspect,
                        "Category": a.category,
                        "Sentiment": a.polarity.capitalize(),
                        "Confidence": f"{a.confidence * 100:.1f}%",
                        "Evidence": ", ".join(a.evidence) if a.evidence else "N/A"
                    })

            df_batch = pd.DataFrame(flat_data)
            st.success(f"Processed {len(lines)} inputs, identified {len(df_batch)} aspect opinions.")
            st.dataframe(df_batch, use_container_width=True)

            csv_data = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Results CSV",
                data=csv_data,
                file_name="hinglish_absa_results.csv",
                mime="text/csv"
            )

    else:
        sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "sample_hinglish_reviews.csv")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as sf:
                sample_bytes = sf.read()
            st.download_button(
                "📄 Download Sample CSV Template",
                data=sample_bytes,
                file_name="sample_hinglish_reviews.csv",
                mime="text/csv"
            )

        uploaded_file = st.file_uploader("Upload CSV file (must contain a review/text column):", type=["csv"])
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.write("Preview:", df_upload.head(3))
            text_col = st.selectbox("Select column to analyze:", df_upload.columns.tolist())

            if st.button("🚀 Analyze Uploaded Dataset"):
                texts = df_upload[text_col].dropna().astype(str).tolist()
                results = absa.analyze_batch(texts, use_api=use_api_toggle)

                flat_data = []
                for r in results:
                    for a in r.aspects:
                        flat_data.append({
                            "Text": r.raw_text,
                            "Aspect": a.aspect,
                            "Category": a.category,
                            "Sentiment": a.polarity.capitalize(),
                            "Confidence": f"{a.confidence * 100:.1f}%",
                            "Evidence": ", ".join(a.evidence) if a.evidence else "N/A"
                        })
                df_out = pd.DataFrame(flat_data)
                st.dataframe(df_out, use_container_width=True)
                st.download_button(
                    "📥 Download Results CSV",
                    data=df_out.to_csv(index=False).encode('utf-8'),
                    file_name="absa_batch_results.csv",
                    mime="text/csv"
                )

# TAB 3: Benchmark & Evaluation
with tabs[2]:
    st.markdown("### 📊 Benchmark Performance & Training Corpus")
    st.markdown(
        "Our model is trained and calibrated across multiple published benchmark datasets: "
        "**ABSA-MIX** (Laptop & Restaurant domains), **ACSA-Mix**, and the **IIT Patna Hindi ABSA** corpus."
    )

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Training Pairs", "7,883")
    with col_m2:
        st.metric("Benchmark Test Pairs", "2,223")
    with col_m3:
        st.metric("Overall Accuracy", "67.8%")
    with col_m4:
        st.metric("Weighted F1", "68.3%")

    st.markdown("---")
    st.markdown("**Test Set Performance Breakdown:**")
    metrics_data = [
        {"Class": "Positive", "Precision": "81.8%", "Recall": "76.1%", "F1-Score": "78.8%", "Support": "1,320"},
        {"Class": "Negative", "Precision": "47.4%", "Recall": "61.9%", "F1-Score": "53.7%", "Support": "360"},
        {"Class": "Neutral", "Precision": "53.3%", "Recall": "51.6%", "F1-Score": "52.4%", "Support": "543"},
    ]
    st.table(pd.DataFrame(metrics_data))

    st.markdown("### 📚 Integrated Benchmark Datasets")
    st.markdown("""
    1. **ABSA-MIX (2024)**: 5,879 train pairs and 1,733 test pairs of code-mixed laptop and restaurant reviews with term-level polarities.
    2. **ACSA-Mix**: Hinglish aspect-category sentiment analysis.
    3. **IIT Patna Hindi ABSA**: 2,449 authentic Devanagari Hindi annotated aspect pairs across mobile apps and electronics.
    4. **SemEval-2014 Task 4**: English aspect-based sentiment benchmark.
    5. **Hinglish Slang Corpus**: Curated colloquial vocabulary and phonetic normalizations.
    """)

# TAB 4: Hinglish Slang Explorer
with tabs[3]:
    st.markdown("### 📖 Hinglish Slang & Informal Expression Explorer")
    st.markdown("Explore our comprehensive lexicon of Hindi-English slang expressions, valence scores, and categories.")

    search_query = st.text_input("🔍 Search slang or keyword:", "")

    slang_data = []
    for word, weight in sorted(POSITIVE_LEXICON.items(), key=lambda x: -x[1]):
        if not search_query or search_query.lower() in word.lower():
            slang_data.append({
                "Slang Expression": word,
                "Sentiment Polarity": "Positive",
                "Weight": f"+{weight:.1f}",
                "Type": "Positive Slang / Quality Word"
            })

    for word, weight in sorted(NEGATIVE_LEXICON.items(), key=lambda x: x[1]):
        if not search_query or search_query.lower() in word.lower():
            slang_data.append({
                "Slang Expression": word,
                "Sentiment Polarity": "Negative",
                "Weight": f"{weight:.1f}",
                "Type": "Negative Slang / Defect Word"
            })

    for word, weight in sorted(NEUTRAL_LEXICON.items(), key=lambda x: x[1]):
        if not search_query or search_query.lower() in word.lower():
            slang_data.append({
                "Slang Expression": word,
                "Sentiment Polarity": "Neutral",
                "Weight": f"{weight:.1f}",
                "Type": "Neutral / Average Word"
            })

    st.dataframe(pd.DataFrame(slang_data), use_container_width=True)

# TAB 5: API & Documentation
with tabs[4]:
    st.markdown("### 📘 REST API & Python Library Usage")
    st.markdown("Easily integrate Hinglish ABSA into Python applications, microservices, or data pipelines.")

    st.markdown("#### 1. Python Code Snippet:")
    st.code("""
from hinglish_absa import HinglishABSA

# Initialize (100% offline local model by default)
absa = HinglishABSA()

# Analyze any English, Hindi, or Hinglish text (word, sentence, or paragraph)
result = absa.analyze("Phone ka camera bahut accha hai but battery backup bekar hai.")

print(result.summary())
# Output:
#   ✅ Camera (Camera) ➔ POSITIVE [98.0% conf] | Evidence: Intensifier 'bahut' (1.6x), Positive 'accha' (+3.2)
#   ❌ Battery backup (Battery) ➔ NEGATIVE [95.0% conf] | Evidence: Negative 'bekar' (-2.5)

# Convert to DataFrame or Dictionary
df = result.to_dataframe()
json_dict = result.to_dict()
    """, language="python")

    st.markdown("#### 2. REST API Endpoints:")
    st.code("""
# Start server:
# python -m hinglish_absa.api  (or uvicorn hinglish_absa.api:app --port 8000)

# Single text analysis:
POST http://127.0.0.1:8000/analyze
Body: {"text": "Movie ki story mast thi but acting thodi boring thi."}

# Batch analysis:
POST http://127.0.0.1:8000/batch
Body: {"texts": ["Display ekdum OP hai!", "Service bilkul ghatiya thi."]}
    """, language="bash")
