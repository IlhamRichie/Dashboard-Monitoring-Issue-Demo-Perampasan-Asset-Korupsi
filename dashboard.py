# Nama file: dashboard.py (WordCloud Telah Diintegrasikan)

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import re
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from wordcloud import WordCloud  # <-- Import di bagian atas

# =======================================================================
# PENGATURAN HALAMAN
# =======================================================================

st.set_page_config(
    page_title="Dashboard Analisis Sentimen Publik",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =======================================================================
# DATA & UTIL
# =======================================================================

@st.cache_data
def load_data_from_db(db_name="youtube_data.db"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, db_name)
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM comments", conn)
    conn.close()
    df.dropna(subset=["text", "sentimen"], inplace=True)
    df["text"] = df["text"].astype(str)
    return df

def _clean_text(txt: str) -> str:
    txt = txt.lower()
    txt = re.sub(r"http\S+", " ", txt)
    txt = re.sub(r"[^a-z\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

@st.cache_data
def extract_top_phrases(_df_negative: pd.DataFrame, topn: int = 15) -> pd.DataFrame:
    cleaned = [_clean_text(t) for t in _df_negative["text"]]
    stopword_factory = StopWordRemoverFactory()
    stopwords = set(stopword_factory.get_stop_words())
    vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words=list(stopwords))
    X = vectorizer.fit_transform(cleaned)
    phrases = vectorizer.get_feature_names_out()
    counts = np.asarray(X.sum(axis=0)).ravel()
    df_phr = pd.DataFrame({"frasa": phrases, "jumlah": counts})
    df_phr = df_phr.sort_values("jumlah", ascending=False).head(topn)
    return df_phr

# --- KODE WORDCLOUD BARU DIINTEGRASIKAN DI SINI ---
# Mendefinisikan stopwords di luar fungsi agar bisa di-cache dan lebih efisien
stopword_factory = StopWordRemoverFactory()
base_stopwords = set(stopword_factory.get_stop_words())
extra_stopwords = {
    "yang", "ini", "itu", "ada", "dari", "untuk", "dengan", "juga",
    "pada", "sudah", "akan", "kami", "kita", "saya", "nya", "agar",
    "bisa", "tidak", "tak", "tdk", "ga", "gak", "aja", "lagi", "kok", "apa","yang","yg","lu","gk",
    "yg", "dr", "dlm", "trs", "jd", "lah", "si", "nih", "loh", "deh","jangan","klu","kalo","apa","semua","ayo","dgn"
}
all_stopwords = base_stopwords.union(extra_stopwords)

@st.cache_data
def create_wordcloud_text(_df_negative: pd.DataFrame) -> str:
    texts = []
    for t in _df_negative["text"]:
        t = t.lower()
        t = re.sub(r"http\S+", " ", t)
        t = re.sub(r"[^a-z\s]", " ", t)
        tokens = t.split()
        tokens = [tok for tok in tokens if tok not in all_stopwords] # Gunakan daftar stopwords yang sudah lengkap
        texts.append(" ".join(tokens))
    return " ".join(texts)
# ----------------------------------------------------

# =======================================================================
# UI
# =======================================================================

st.title("⚡ Dashboard Analisis Sentimen Publik")
st.markdown("Monitoring Isu Demo & RUU Perampasan Aset di YouTube Indonesia")
st.markdown(f"**Data dianalisis per:** `{datetime.now().strftime('%d %B %Y, %H:%M WIB')}`")

df_analyzed = load_data_from_db()

if df_analyzed is None:
    st.warning("⚠️ Menunggu data dari crawler otomatis. Jalankan crawler kemudian refresh halaman ini.")
    st.stop()

sentiment_options = list(df_analyzed["sentimen"].dropna().unique())
sentiment_filter = st.multiselect(
    "🔍 Pilih Sentimen untuk Ditampilkan:",
    options=sentiment_options,
    default=sentiment_options,
    help="Filter data berdasarkan sentimen"
)
df_selection = df_analyzed[df_analyzed["sentimen"].isin(sentiment_filter)]

# =======================================================================
# METRIK
# =======================================================================

st.markdown("---")
total_comments = len(df_selection)
sentiment_order = ["Negatif", "Netral", "Positif"]
sentiment_counts = df_selection["sentimen"].value_counts().reindex(sentiment_order, fill_value=0)
neg_pct = (sentiment_counts["Negatif"] / total_comments) * 100 if total_comments else 0
pos_pct = (sentiment_counts["Positif"] / total_comments) * 100 if total_comments else 0

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total Komentar (Filtered)", f"{total_comments:,}")
with c2:
    st.metric("Sentimen Negatif", f"{neg_pct:.1f}%")
with c3:
    st.metric("Sentimen Positif", f"{pos_pct:.1f}%")

st.markdown("---")

# =======================================================================
# VISUALISASI
# =======================================================================

st.header("📊 Visualisasi Interaktif")
vc1, vc2 = st.columns(2)

with vc1:
    st.subheader("Distribusi Sentimen")
    if total_comments == 0 or sentiment_counts.sum() == 0:
        st.warning("Tidak ada data untuk sentimen yang dipilih.")
    else:
        labels = sentiment_counts.index.tolist()
        values = sentiment_counts.values.tolist()
        colors = ["#FF6F61", "#FFD700", "#2ECC71"]
        fig_pie, ax = plt.subplots(figsize=(6, 6), dpi=160)
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct="%.1f%%", startangle=90,
            colors=colors, pctdistance=0.7, labeldistance=1.08,
            wedgeprops=dict(linewidth=2, edgecolor="white")
        )
        for t in texts:
            t.set_fontweight("bold"); t.set_fontsize(12)
        for at in autotexts:
            at.set_fontsize(12); at.set_color("black")
        ax.axis("equal")
        st.pyplot(fig_pie)

with vc2:
    st.subheader("Topik di Komentar Negatif")
    df_negative = df_selection[df_selection["sentimen"] == "Negatif"]
    if df_negative.empty:
        st.warning("Tidak ditemukan komentar negatif.")
    else:
        df_phrases = extract_top_phrases(df_negative, topn=15)
        df_plot = df_phrases.sort_values("jumlah", ascending=True)
        fig_bar, axb = plt.subplots(figsize=(8, 5), dpi=150)
        axb.barh(df_plot["frasa"], df_plot["jumlah"], color="#FF6F61")
        axb.set_xlabel("jumlah")
        axb.set_ylabel("")
        axb.tick_params(axis='y', labelsize=10)
        axb.margins(y=0.02)
        plt.tight_layout()
        st.pyplot(fig_bar)

# =======================================================================
# WAWASAN KUALITATIF
# =======================================================================

st.markdown("---")
st.header("💬 Wawasan Kualitatif")
w1, w2 = st.columns(2)

with w1:
    # --- KODE WORDCLOUD BARU DIGUNAKAN DI SINI ---
    st.subheader("Word Cloud Komentar Negatif")
    # Definisikan df_negative di sini juga untuk memastikan variabelnya ada
    df_negative = df_selection[df_selection["sentimen"] == "Negatif"]
    if df_negative.empty:
        st.warning("Tidak ada data untuk Word Cloud.")
    else:
        wc_text = create_wordcloud_text(df_negative) # Panggil fungsi yang sudah disempurnakan
        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            colormap="Reds",
            collocations=False
        ).generate(wc_text)
        fig_wc, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc)
    # -----------------------------------------------

with w2:
    st.subheader("Contoh Komentar")
    df_positive = df_selection[df_selection["sentimen"] == "Positif"]
    with st.expander("Lihat Contoh Komentar Positif ✅"):
        if not df_positive.empty:
            st.dataframe(df_positive[["author", "text"]].sample(min(5, len(df_positive))))
        else:
            st.write("Tidak ada komentar positif.")
    with st.expander("Lihat Contoh Komentar Negatif ❌"):
        # Gunakan df_negative yang sudah kita definisikan di atas
        if not df_negative.empty:
            st.dataframe(df_negative[["author", "text"]].sample(min(5, len(df_negative))))
        else:
            st.write("Tidak ada komentar negatif.")

st.markdown("---")
st.header("⬇️ Data Lengkap (Filtered)")
st.dataframe(df_selection)