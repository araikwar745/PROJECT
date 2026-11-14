"""
Streamlit Web Dashboard for Hinglish Aspect-Based Sentiment Analysis (ABSA).
Interactive UI with real-time analysis, presets, batch mode, benchmark evaluation,
and slang dictionary explorer.
"""

import streamlit as st
import pandas as pd
import json
from hinglish_absa import HinglishABSA
from hinglish_absa.lexicons import POSITIVE_LEXICON, NEGATIVE_LEXICON, ASPECT_CATEGORIES

# Page configuration
st.set_page_config(
    page_title="Hinglish Aspect-Based Sentiment Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
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
    .aspect-card {
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
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
</style>
""", unsafe_allow_html=True)

# Cache engine
@st.cache_resource
def load_absa_engine():
    return HinglishABSA()

absa = load_absa_engine()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.markdown("### 🧠 Hinglish ABSA System")
    st.markdown(
        "A hybrid NLP architecture for **Aspect-Based Sentiment Analysis** in Hindi-English "
        "code-mixed language with full support for informal expressions, slang, and emojis."
    )
    st.markdown("---")
    st.markdown("**Core Capabilities:**")
    st.markdown("- 🔍 Fine-Grained Aspect Target Extraction")
    st.markdown("- 🎭 Slang & Informal Expression Support (OP, paisa vasool, etc.)")
    st.markdown("- 🔀 Clause-Aware Contrastive Parsing (`but`, `lekin`, `par`)")
    st.markdown("- 🛡️ Negation & Intensifier Scaling")
    st.markdown("- 🤖 Calibrated ABSA-MIX Benchmark Model")
    st.markdown("---")
    st.caption("Antigravity AI • Python 3.14 & Scikit-Learn")

# Header
st.markdown('<div class="main-title">🎯 Hinglish Aspect-Based Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Identify specific aspects in Hindi-English code-mixed reviews and predict their individual sentiments with high-precision slang handling.</div>',
    unsafe_allow_html=True
)

tabs = st.tabs(["🚀 Live Interactive Analyzer", "📁 Batch Analysis", "📊 ABSA-MIX Benchmark", "📖 Hinglish Slang Explorer"])

# TAB 1: Live Interactive Analyzer
with tabs[0]:
    presets = {
        "Custom Input...": "",
        "Example 1 (Mobile - Camera vs Battery)": "Phone ka camera bahut accha hai but battery backup bekar hai.",
        "Example 2 (Entertainment - Story vs Acting)": "Movie ki story mast thi but acting thodi boring thi.",
        "Example 3 (Laptop - Performance vs Thermals)": "Laptop ka performance awesome hai, lekin heating issue bahut hai.",
        "Example 4 (Slang - Phone Quality & Value)": "Bhai phone ekdum OP hai, paisa vasool!",
        "Example 5 (Restaurant - Food vs Service)": "Food fresh aur swadist tha par waiter ka behavior bhot rude tha.",
        "Example 6 (Emojis + Slang)": "Camera 🔥 picture quality ekdum kadak but battery backup 🤮",
        "Example 7 (Negation Handling)": "Screen acchi nahi hai aur speaker se awaz bilkul dhang se nahi aati.",
    }

    selected_preset = st.selectbox("Select a benchmark sample or choose custom input:", list(presets.keys()))
    default_text = presets[selected_preset] if selected_preset != "Custom Input..." else "Phone ka camera bahut accha hai but battery backup bekar hai."

    input_text = st.text_area(
        "Enter Hinglish sentence, social media review, or comment:",
        value=default_text,
        height=90,
        placeholder="e.g. Phone ka camera mast hai but battery bekar hai..."
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        analyze_clicked = st.button("⚡ Analyze Sentiment", type="primary", use_container_width=True)

    if analyze_clicked or input_text:
        if input_text.strip():
            result = absa.analyze(input_text)

            st.markdown("---")
            st.markdown("### 📋 Analysis Results")

            if not result.has_aspects:
                st.warning("No explicit aspect targets were detected. Try mentioning a specific product or feature (e.g. camera, battery, story, food, phone).")
            else:
                col_left, col_right = st.columns([3, 2])

                with col_left:
                    st.markdown(f"**Detected Aspects ({len(result.aspects)}):**")
                    for a in result.aspects:
                        badge_class = f"badge-{a.polarity}"
                        icon = "✅" if a.polarity == "positive" else ("❌" if a.polarity == "negative" else "⚪")
                        border_color = "#28a745" if a.polarity == "positive" else ("#dc3545" if a.polarity == "negative" else "#6c757d")

                        st.markdown(
                            f"""
                            <div style="border-left: 5px solid {border_color}; background-color: rgba(240,240,240,0.15); padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                    <span style="font-size: 1.15rem; font-weight: 700;">{a.aspect.capitalize()}</span>
                                    <span class="{badge_class}">{icon} {a.polarity.upper()} ({a.confidence*100:.1f}%)</span>
                                </div>
                                <div style="margin-bottom: 4px;">
                                    <span class="category-chip">Category: {a.category}</span>
                                </div>
                                <div style="font-size: 0.9rem; color: #555; margin-top: 6px;">
                                    <strong>Evidence Words:</strong> {", ".join(a.evidence) if a.evidence else "Model contextual inference"}
                                </div>
                                <div style="font-size: 0.85rem; color: #777; margin-top: 2px;">
                                    <strong>Clause Scope:</strong> <em>"{a.clause_context}"</em>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                with col_right:
                    st.markdown("**Sentiment Distribution:**")
                    sent_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
                    for a in result.aspects:
                        sent_counts[a.polarity.capitalize()] += 1

                    df_chart = pd.DataFrame({
                        "Sentiment": list(sent_counts.keys()),
                        "Aspect Count": list(sent_counts.values())
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

                with st.expander("🔍 Deep Pipeline Inspection (Preprocessing & NLP Traces)"):
                    st.markdown(f"**Original Input:** `{result.raw_text}`")
                    st.markdown(f"**Phonetic Normalized:** `{result.normalized_text}`")
                    st.json(result.to_dict())

# TAB 2: Batch Analysis
with tabs[1]:
    st.markdown("### 📁 Batch Processing for Hinglish Text")
    st.markdown("Upload a CSV file containing reviews or paste multiple sentences to analyze in bulk.")

    batch_input_method = st.radio("Input format:", ["Paste multiple reviews (one per line)", "Upload CSV file"], horizontal=True)

    if batch_input_method == "Paste multiple reviews (one per line)":
        default_batch = (
            "Phone ka camera bahut accha hai but battery backup bekar hai.\n"
            "Movie ki story mast thi but acting thodi boring thi.\n"
            "Laptop ka performance awesome hai, lekin heating issue bahut hai.\n"
            "Bhai phone ekdum OP hai, paisa vasool!\n"
            "Food fresh aur tasty tha lekin seating arrangement ghatiya thi.\n"
            "Customer care walon ne turant issue resolve kiya, best service!"
        )
        batch_text = st.text_area("Paste reviews (one per line):", value=default_batch, height=180)

        if st.button("🚀 Process Batch Reviews", type="primary"):
            lines = [l.strip() for l in batch_text.splitlines() if l.strip()]
            results = absa.analyze_batch(lines)

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
            st.success(f"Processed {len(lines)} sentences, identified {len(df_batch)} aspect opinions.")
            st.dataframe(df_batch, use_container_width=True)

            csv_data = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Results CSV",
                data=csv_data,
                file_name="hinglish_absa_results.csv",
                mime="text/csv"
            )

    else:
        import os
        sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "sample_hinglish_reviews.csv")
        sample_df = None
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as sf:
                sample_bytes = sf.read()
            st.download_button(
                "📄 Download Sample CSV Template",
                data=sample_bytes,
                file_name="sample_hinglish_reviews.csv",
                mime="text/csv",
                help="Download a ready-to-use CSV template with mixed Hindi-English reviews"
            )

        use_sample = st.checkbox("💡 Or load built-in sample dataset (15 reviews)", value=False)
        uploaded_file = st.file_uploader("Upload CSV file (must contain a text column):", type=["csv"])

        df_upload = None
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
        elif use_sample and os.path.exists(sample_path):
            df_upload = pd.read_csv(sample_path)
            st.info("Loaded built-in sample reviews dataset.")

        if df_upload is not None:
            st.write("Preview of data:", df_upload.head(3))
            text_col = st.selectbox("Select text column to analyze:", df_upload.columns.tolist())

            if st.button("🚀 Analyze Dataset"):
                texts = df_upload[text_col].dropna().astype(str).tolist()
                results = absa.analyze_batch(texts)

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
    st.markdown("### 📊 ABSA-MIX Benchmark Performance")
    st.markdown(
        "Evaluation on the official test sets from the research paper *"
        "Quality achhi hai (is good), satisfied! Towards aspect based sentiment analysis in code-mixed language* (CSL 2024)."
    )

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Test Pairs Evaluated", "1,733")
    with col_m2:
        st.metric("Overall Accuracy", "69.30%")
    with col_m3:
        st.metric("Weighted F1", "68.68%")
    with col_m4:
        st.metric("Positive F1", "81.81%")

    st.markdown("---")
    st.markdown("**Per-Class Classification Metrics on Test Set:**")
    metrics_data = [
        {"Class": "Positive", "Precision": "81.08%", "Recall": "82.54%", "F1-Score": "81.81%", "Support": "1,054"},
        {"Class": "Negative", "Precision": "49.64%", "Recall": "64.29%", "F1-Score": "56.02%", "Support": "322"},
        {"Class": "Neutral", "Precision": "51.03%", "Recall": "34.73%", "F1-Score": "41.33%", "Support": "357"},
    ]
    st.table(pd.DataFrame(metrics_data))

    st.markdown("### 🎯 Benchmark Highlights")
    st.markdown("- **Dual-Domain Coverage:** Trained on both Restaurant and Laptop domains from ABSA-MIX (5,879 train pairs).")
    st.markdown("- **Code-Mixed Generalization:** Seamlessly processes Roman script, Devanagari script, and code-mixed hybrid sentences.")
    st.markdown("- **Hybrid Precision:** High precision on authentic Indian social media Hinglish slang words like *OP, paisa vasool, mast, dhansu, jhakaas, bekar, kachra*.")

# TAB 4: Hinglish Slang Explorer
with tabs[3]:
    st.markdown("### 📖 Hinglish Slang & Informal Expression Explorer")
    st.markdown("Explore the curated lexicon of Hindi-English informal slang and their sentiment valence.")

    search_query = st.text_input("🔍 Search slang or keyword:", "")

    slang_data = []
    for word, weight in sorted(POSITIVE_LEXICON.items(), key=lambda x: -x[1]):
        if not search_query or search_query.lower() in word.lower():
            slang_data.append({
                "Slang Expression": word,
                "Sentiment Polarity": "Positive",
                "Intensity Weight": f"+{weight:.1f}",
                "Type": "Positive Slang / Quality Word"
            })

    for word, weight in sorted(NEGATIVE_LEXICON.items(), key=lambda x: x[1]):
        if not search_query or search_query.lower() in word.lower():
            slang_data.append({
                "Slang Expression": word,
                "Sentiment Polarity": "Negative",
                "Intensity Weight": f"{weight:.1f}",
                "Type": "Negative Slang / Defect Word"
            })

    st.dataframe(pd.DataFrame(slang_data), use_container_width=True)
