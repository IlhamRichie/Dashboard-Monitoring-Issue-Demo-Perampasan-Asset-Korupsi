import os
import sys
import sqlite3
import traceback
import pandas as pd
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import transformers

# =======================================================================
# KONFIGURASI UTAMA
# =======================================================================
API_KEY = 'apikey'
DB_NAME = "youtube_data.db"
TARGET_TOTAL_COMMENTS = 20000

# =======================================================================
# KONFIGURASI PENCARIAN VIDEO - INI BAGIAN BARUNYA!
# =======================================================================
# Gunakan kata kunci yang spesifik. Gunakan | untuk ATAU, "" untuk frasa pasti.
SEARCH_QUERY = '"demo mahasiswa" | "perampasan asset" | "tolak RUU"'
# Berapa banyak video teratas yang ingin diambil ID-nya dari hasil pencarian?
MAX_SEARCH_RESULTS = 15 
# Cari video yang diupload dalam 7 hari terakhir untuk menjaga relevansi
SEARCH_PERIOD_DAYS = 7
# =======================================================================

# --- BAGIAN BARU: SETUP LOGGING ---
# Arahkan semua output (print) ke file log
script_dir_for_log = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir_for_log, 'crawler.log')
sys.stdout = open(log_file_path, 'w', encoding='utf-8')
sys.stderr = sys.stdout
# ------------------------------------


def analyze_sentiment_in_batch(df):
    """
    Fungsi untuk menjalankan analisis sentimen pada DataFrame. (VERSI FINAL)
    """
    print("🧠 Memulai analisis sentimen pada data yang di-crawl...")
    try:
        sentiment_analyzer = transformers.pipeline(
            "sentiment-analysis",
            model="mdhugol/indonesia-bert-sentiment-classification"
        )
        print("   Model AI berhasil dimuat.")
    except Exception as e:
        print("[FATAL ERROR] Gagal memuat model AI. Proses analisis dibatalkan.")
        traceback.print_exc(file=sys.stdout)
        df['sentimen'] = 'Error'
        return df

    def get_sentiment(text):
        try:
            result = sentiment_analyzer(str(text)[:512])
            label = result[0]['label']
            
            # --- PERBAIKAN UTAMA DI SINI ---
            if label == 'LABEL_2':
                return "Negatif"
            elif label == 'LABEL_0':
                return "Positif"
            else: # Anggap sisanya (LABEL_1) sebagai Netral
                return "Netral"
            # --------------------------------

        except Exception:
            print(f"\n--- Gagal menganalisis teks: {str(text)[:50]}... ---")
            traceback.print_exc(file=sys.stdout)
            print("--------------------------------------------------\n")
            return "Error"
    
    df.dropna(subset=['text'], inplace=True)
    df['text'] = df['text'].astype(str)
    
    df['sentimen'] = df['text'].apply(get_sentiment)
    print("✅ Analisis sentimen selesai.")
    return df

def search_videos(api_key, query, max_results, period_days):
    print(f"🔎 Mencari video dengan kata kunci: '{query}'...")
    youtube = build('youtube', 'v3', developerKey=api_key)
    search_after_date = (datetime.now() - timedelta(days=period_days)).strftime('%Y-%m-%dT%H:%M:%SZ')
    try:
        request = youtube.search().list(part="snippet", q=query, type="video", order="relevance", maxResults=max_results, regionCode="ID", relevanceLanguage="id", publishedAfter=search_after_date)
        response = request.execute()
        video_ids = [item['id']['videoId'] for item in response.get('items', [])]
        if video_ids: print(f"✅ Berhasil menemukan {len(video_ids)} ID video.")
        else: print("⚠️ Tidak ada video yang ditemukan.")
        return video_ids
    except Exception as e:
        print(f"[ERROR] Gagal mencari video: {e}")
        traceback.print_exc(file=sys.stdout)
        return []

def scrape_youtube_comments(api_key, video_ids, target_count):
    comments_list = []
    youtube = build('youtube', 'v3', developerKey=api_key)
    for video_id in video_ids:
        if len(comments_list) >= target_count: break
        print(f"   -> Mengambil komentar dari video ID: {video_id}")
        try:
            request = youtube.commentThreads().list(part='snippet', videoId=video_id, maxResults=100, textFormat='plainText')
            response = request.execute()
            while response:
                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    comments_list.append({'author': comment.get('authorDisplayName'),'text': comment.get('textDisplay'),'published_at': comment.get('publishedAt'),'like_count': comment.get('likeCount', 0)})
                if 'nextPageToken' in response and len(comments_list) < target_count:
                    request = youtube.commentThreads().list(part='snippet', videoId=video_id, maxResults=100, textFormat='plainText', pageToken=response['nextPageToken'])
                    response = request.execute()
                else: break
        except Exception as e:
            print(f"      [ERROR] Gagal mengambil dari video {video_id}: {e}")
            traceback.print_exc(file=sys.stdout)
            continue
    return pd.DataFrame(comments_list)

def save_to_sqlite(df, db_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, db_name)
    try:
        conn = sqlite3.connect(db_path)
        print(f"Menghapus tabel 'comments' lama dan menyimpan {len(df)} data baru...")
        df.to_sql("comments", conn, if_exists='replace', index=False)
        conn.commit()
        print(f"✅ Data berhasil disimpan di '{db_path}'")
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan ke database: {e}")
        traceback.print_exc(file=sys.stdout)
    finally:
        if 'conn' in locals() and conn: conn.close()

def main():
    print(f"🚀 Memulai proses monitoring & analisis otomatis pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    video_ids_to_scrape = search_videos(API_KEY, SEARCH_QUERY, MAX_SEARCH_RESULTS, SEARCH_PERIOD_DAYS)
    
    if video_ids_to_scrape:
        df_comments = scrape_youtube_comments(API_KEY, video_ids_to_scrape, TARGET_TOTAL_COMMENTS)
        
        if not df_comments.empty:
            print(f"Berhasil mengumpulkan total {len(df_comments)} komentar.")
            df_analyzed = analyze_sentiment_in_batch(df_comments)
            save_to_sqlite(df_analyzed, DB_NAME)
        else:
            print("Tidak ada komentar yang berhasil di-crawl.")
    else:
        print("Proses dihentikan karena tidak ada video yang ditemukan.")
    print("✨ Proses monitoring & analisis selesai.")

if __name__ == "__main__":
    main()

sys.stdout.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__