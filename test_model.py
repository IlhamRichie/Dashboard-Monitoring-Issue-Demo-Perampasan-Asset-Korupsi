# Nama file: tes_model.py

import transformers

print("🔬 Memuat model AI untuk tes...")
sentiment_analyzer = transformers.pipeline(
    "sentiment-analysis",
    model="mdhugol/indonesia-bert-sentiment-classification"
)
print("✅ Model siap.")

# Ambil salah satu komentar yang kamu yakin 100% negatif
komentar_negatif = "Dpr pengecut , pemerintah biarkan .saatnya rakyat melawan"

print(f"\n💬 Menganalisis komentar: '{komentar_negatif}'")

# Jalankan analisis dan lihat output mentahnya
hasil_analisis = sentiment_analyzer(komentar_negatif)

print("\n--- HASIL MENTAH DARI MODEL ---")
print(hasil_analisis)
print("---------------------------------")