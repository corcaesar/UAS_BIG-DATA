# Big Data Sentiment Analysis — Environmental Management

Dashboard dan notebook ini dibuat untuk proyek UAS Big Data C dengan topik **Analisis Sentimen Publik terhadap Isu Lingkungan Berdasarkan Dataset Biodiversity Menggunakan Big Data Analytics**.

Project ini menggunakan pipeline pengolahan data berbasis **PySpark** untuk membaca dataset, melakukan preprocessing, mengekstraksi fitur teks menggunakan **TF-IDF**, melakukan klasifikasi sentimen menggunakan **Logistic Regression**, menghitung agregasi persentase sentimen, dan menampilkan hasilnya melalui dashboard **Streamlit**.

## Ringkasan Hasil

Berdasarkan notebook yang digunakan dalam laporan:

| Metrik | Hasil |
|---|---:|
| Total data pada `biodiversity_for_modelling.csv` | 13.435 data |
| Data subdomain `environmental management` | 4.316 data |
| Data training | 3.502 data |
| Data testing | 814 data |
| Accuracy Logistic Regression | 52,33% |
| Prediksi positif | 495 data / 60,81% |
| Prediksi negatif | 306 data / 37,59% |
| Prediksi netral | 13 data / 1,60% |

Mapping label prediksi yang digunakan pada dashboard:

| Kode Prediksi | Sentimen |
|---:|---|
| 0.0 | positive |
| 1.0 | negative |
| 2.0 | neutral |

## Struktur Project

```text
big-data-sentiment-analysis/
├── bigdata_sentiment_analysis.ipynb
├── streamlit_app.py
├── requirements.txt
├── README.md
├── COLAB_STREAMLIT_SETUP.md
├── .streamlit/
│   └── config.toml
└── data/
    ├── sentiment_summary_sample.csv
    └── sentiment_results.csv              # opsional, hasil ekspor Colab
```

## Tools dan Teknologi

- Python
- Google Colab
- PySpark
- Spark SQL
- TF-IDF
- Logistic Regression
- Pandas
- Plotly
- Streamlit

## Alur Analisis

1. **Data ingestion**  
   Dataset `biodiversity_for_modelling.csv` dibaca menggunakan PySpark DataFrame.

2. **Data cleaning**  
   Data dibersihkan menggunakan proses `dropna()` dan `dropDuplicates()`.

3. **Filtering subdomain**  
   Analisis difokuskan pada subdomain `environmental management`.

4. **Text preparation**  
   Kolom `keyword` dan `subdomain` digabung menjadi kolom `text`.

5. **Feature extraction**  
   Data teks diubah menjadi fitur numerik menggunakan Tokenizer, StopWordsRemover, CountVectorizer, dan IDF.

6. **Modeling**  
   Model Logistic Regression digunakan untuk klasifikasi sentimen.

7. **Evaluation**  
   Evaluasi dilakukan menggunakan metrik accuracy.

8. **Aggregation**  
   Hasil prediksi diagregasikan menggunakan PySpark untuk menghitung jumlah dan persentase tiap sentimen.

9. **Dashboard**  
   Hasil analisis divisualisasikan menggunakan Streamlit dalam bentuk KPI, bar chart, pie chart, tabel ringkasan, top keyword, dan preview data.

## Cara Menjalankan Dashboard Secara Lokal

Install dependency:

```bash
pip install -r requirements.txt
```

Jalankan Streamlit:

```bash
streamlit run streamlit_app.py
```

Buka URL lokal yang muncul di terminal, biasanya:

```text
http://localhost:8501
```

## Cara Menjalankan di Google Colab

Buka file `COLAB_STREAMLIT_SETUP.md`, lalu jalankan cell yang tersedia pada Google Colab.

Inti perintahnya:

```python
!pip install -r requirements.txt
!npm install -g localtunnel
!streamlit run streamlit_app.py --server.port 8501 & npx localtunnel --port 8501
```

## Cara Export CSV Hasil Prediksi dari Colab

Setelah variabel `predictions` terbentuk di notebook, jalankan:

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

Download file CSV part hasil ekspor, rename menjadi `sentiment_results.csv`, lalu letakkan pada folder `data/` atau upload langsung melalui sidebar dashboard.

## Deploy ke Streamlit Community Cloud

1. Push file berikut ke repository GitHub:
   - `streamlit_app.py`
   - `requirements.txt`
   - `README.md`
   - `.streamlit/config.toml`
   - `data/sentiment_results.csv` jika ingin dashboard langsung memakai data final
2. Buka Streamlit Community Cloud.
3. Pilih repository `depmkeu/big-data-sentiment-analysis`.
4. Pilih branch `main`.
5. Isi entrypoint file dengan:

```text
streamlit_app.py
```

6. Klik **Deploy**.

## Catatan Interpretasi

Nilai accuracy 52,33% menunjukkan bahwa pipeline klasifikasi sudah berjalan, tetapi performanya masih moderat. Hal ini wajar karena fitur yang digunakan hanya berasal dari kombinasi `keyword` dan `subdomain`, bukan teks lengkap media sosial. Untuk hasil yang lebih optimal, penelitian lanjutan dapat menggunakan teks lengkap, memperbanyak fitur, menyeimbangkan kelas, dan membandingkan model lain seperti Naive Bayes, Support Vector Machine, Random Forest, atau model berbasis transformer.

## Anggota Kelompok

- Ni Made Adelia Wirasanti — 2405551010
- Ni Putu Candradevi Davantari — 2405551035
- Cornelius Caesar Hendrik Pasaribu — 2405551059
- Putu Della Pradnyaswari Cipta Dewi — 2405551063
- Ida Ayu Ratih Widya Paramitha — 2405551111
