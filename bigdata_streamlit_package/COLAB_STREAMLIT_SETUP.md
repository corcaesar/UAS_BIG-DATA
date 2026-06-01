# Menjalankan Dashboard Streamlit dari Google Colab

Gunakan langkah ini setelah file `streamlit_app.py` dan `requirements.txt` sudah ada di repository GitHub atau sudah di-upload ke Colab.

## Opsi 1 — Jalankan dari repository GitHub

```python
!git clone https://github.com/depmkeu/big-data-sentiment-analysis.git
%cd big-data-sentiment-analysis
!pip install -r requirements.txt
```

Jika file dashboard belum ada di repo, upload `streamlit_app.py` dan `requirements.txt` terlebih dahulu ke repo.

```python
!npm install -g localtunnel
!streamlit run streamlit_app.py --server.port 8501 & npx localtunnel --port 8501
```

Colab akan menampilkan URL sementara dari localtunnel. Buka URL tersebut untuk melihat dashboard.

## Opsi 2 — Ekspor hasil prediksi dari notebook Colab

Jalankan cell ini setelah variabel `predictions` terbentuk di notebook.

```python
from pyspark.sql.functions import when, col

predictions_export = predictions.withColumn(
    "predicted_sentiment",
    when(col("prediction") == 0.0, "positive")
    .when(col("prediction") == 1.0, "negative")
    .when(col("prediction") == 2.0, "neutral")
    .otherwise("unknown")
)

predictions_export.select(
    "id", "keyword", "subdomain", "final label", "prediction", "predicted_sentiment"
).coalesce(1).write.mode("overwrite").option("header", True).csv("/content/sentiment_results_export")
```

Setelah itu, download file CSV hasil ekspor dari folder `/content/sentiment_results_export/` dan rename menjadi `sentiment_results.csv`. File ini bisa di-upload ke dashboard atau diletakkan di folder `data/` pada repository.

## Opsi 3 — Deploy permanen ke Streamlit Community Cloud

1. Pastikan repository berisi minimal file berikut:
   - `streamlit_app.py`
   - `requirements.txt`
   - `README.md`
   - opsional: `data/sentiment_results.csv`
2. Push semua file ke branch `main`.
3. Buka Streamlit Community Cloud, pilih repository, branch `main`, dan entrypoint `streamlit_app.py`.
4. Klik deploy dan tunggu proses build selesai.

Jika dataset tidak dimasukkan ke repo, dashboard tetap bisa dibuka menggunakan sample data dan pengguna dapat upload CSV secara manual dari sidebar.
