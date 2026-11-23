# Penjelasan Pipeline Notebook `pci-ml-26.ipynb`

Dokumen ini menjelaskan alur **pipeline prediksi Pavement Condition Index (PCI)** dari awal sampai akhir, stage demi stage, beserta file yang dihasilkan di setiap tahap. Tujuannya agar mudah dipahami, direproduksi, dan dilaporkan.

---

## 1. Gambaran umum

Notebook menjalankan satu pipeline utuh:

```
Data mentah Excel
   → pembersihan & rekayasa fitur
   → pemisahan train/test berbasis jalan
   → tuning + validasi silang (RF, XGBoost, ANN)
   → pemilihan model terbaik
   → interpretasi SHAP
   → artefak (model, prediksi, metrik, gambar, tabel)
```

Prinsip penting:

- **Tidak ada kebocoran data**: pemisahan train/test dilakukan per `ROAD_ID`, dan validasi silang juga mengelompok per jalan (`GroupKFold`).
- **Data uji tidak dipakai saat tuning**. Data uji baru dipakai di stage 6 (prediksi) dan stage 7 (SHAP).
- Semua hasil disimpan ke folder `models/` dan `outputs/`, sehingga notebook bisa dijalankan ulang tanpa menghitung ulang dari nol pada tahap tertentu.

---

## 2. Struktur folder

```
pci-ml/
├── datasets/
│   ├── pci-ml-26.xlsx                  # data mentah
│   └── pci-ml-26/<tag>/                # hasil split train/test + meta.json
├── models/<dataset>_<tag>/             # model terlatih + evaluasi + OOF
├── outputs/
│   ├── figures/<tag>/                  # gambar evaluasi & SHAP
│   ├── predictions/<tag>/              # prediksi data uji per model
│   └── shap/<tag>/                     # tabel & metadata SHAP
├── src/pci_ml/                         # kode pipeline (modular)
└── pci-ml-26.ipynb                     # notebook utama
```

`<tag>` adalah penanda kombinasi parameter split, misalnya `n100_t0.2_s0.01_p0`.

---

## 3. Parameter eksperimen (Stage 0)

Semua parameter utama didefinisikan di awal agar konsisten:

| Parameter | Nilai | Arti |
|---|---|---|
| `DATASET` | `pci-ml-26.xlsx` | Nama file data mentah |
| `n_seed` | 100 | Jumlah kandidat seed yang dicari |
| `test_size` | 0.2 | Proporsi data uji (±20%) |
| `share_tol` | 0.01 | Toleransi proporsi uji |
| `pci_min` | 0 | Data uji wajib punya jalan dengan PCI = 0 |
| `CV_SPLITS` | 5 | Jumlah fold validasi silang |
| `RF_N_ITER` | 100 | Iterasi pencarian hyperparameter RF |
| `XGB_N_ITER` | 100 | Iterasi pencarian hyperparameter XGBoost |
| `ANN_N_ITER` | 20 | Iterasi pencarian hyperparameter ANN |
| `ANN_EPOCHS` | 200 | Epoch maksimum pelatihan ANN |
| `ANN_PATIENCE` | 15 | Early stopping ANN |
| `SEED` | 42 | Seed untuk reproducibility |

Stage ini juga mengimpor fungsi bantu dari `src/pci_ml/` dan utilitas tampilan.

---

## 4. Stage 1 — Pembersihan data & rekayasa fitur

**File:** `src/pci_ml/data.py`

Langkah:

1. **Muat** data Excel (`load_data`).
2. **Bersihkan** (`clean_data`):
   - hapus duplikat,
   - buang baris tanpa `PCI` atau `SEGMENT_AREA`,
   - buang area ≤ 0,
   - batasi `PCI` ke rentang 0–100,
   - isi nilai fitur kosong dengan 0.
3. **Normalisasi ke densitas** (`to_density`): setiap fitur kerusakan dibagi `SEGMENT_AREA`, lalu di-clip 0–1. Kolom area kemudian dibuang.
4. **Rekayasa fitur** (`engineer_features`):
   - `TOTAL_DENSITY` = jumlah seluruh densitas kerusakan,
   - `TOTAL_L`, `TOTAL_M`, `TOTAL_H` = jumlah densitas per tingkat keparahan (Low/Medium/High).

**Output:** DataFrame bersih + daftar fitur (59 fitur pada dataset ini).

> Catatan: fitur agregat (`TOTAL_*`) adalah penjumlahan fitur mentah. Karena itu pada interpretasi SHAP, ranking distress memakai fitur mentah, sedangkan fitur agregat dilaporkan terpisah.

---

## 5. Stage 2 — Pemisahan & pemuatan data

### 5a. Pemisahan train/test

**File:** `src/pci_ml/data.py` (`class DataSplit`)

- Memakai `GroupShuffleSplit` dengan grup `ROAD_ID`, sehingga **satu jalan hanya berada di train atau test**, tidak keduanya.
- Mencari seed valid pertama (0..99) yang memenuhi tiga kriteria:
  1. proporsi test ≈ 20% (±1%),
  2. tidak ada fitur yang hanya muncul di test,
  3. data uji memuat minimal satu jalan dengan `PCI = 0`.
- Hasil split disimpan ke `datasets/pci-ml-26/<tag>/`:
  - `train.csv`, `test.csv`, `meta.json` (seed, gap, daftar fitur, jumlah baris, timestamp).

### 5b. Pemuatan input model

**File:** `src/pci_ml/data.py` (`class ModelInput`)

- Memuat kembali `train.csv` dan `test.csv` dari folder `<tag>`.
- Menyiapkan array fitur/target serta metadata jalan (`ROAD_ID`, `ROAD_NAME`, `SEGMENT_ID`).
- Membuat fold validasi silang **`GroupKFold`** berdasarkan `ROAD_ID` (`make_folds`), sehingga tidak ada jalan yang muncul di train dan validasi sekaligus.
- Menggambar distribusi jenis kerusakan train vs test (`plot_data_dist`).

**Output:** array `train_features`, `test_features`, `train_pci`, `test_pci`, fold CV, dan gambar `01_distribusi_jenis_kerusakan.png`.

---

## 6. Stage 3–5 — Tuning, validasi silang, dan pelatihan model

Tiga stage ini memiliki pola yang sama untuk **Random Forest**, **XGBoost**, dan **ANN**:

1. **Tuning hyperparameter** dengan `ParameterSampler` (random search) pada ruang pencarian masing-masing.
2. Setiap kandidat dievaluasi dengan **5-fold GroupKFold**; metrik utama = **MAE**.
3. Kandidat terbaik dilatih ulang pada **seluruh data latih**.
4. Menghitung metrik CV dari prediksi *out-of-fold* (OOF):
   - skor gabungan, rata-rata per fold, dan metrik tiap fold,
   - MAE, RMSE, R².
5. Menyimpan model, input evaluasi, prediksi OOF, dan gambar performa OOF.

| Stage | Model | Ruang hyperparameter utama |
|---|---|---|
| 3 | Random Forest | `n_estimators`, `max_depth`, `min_samples_split/leaf`, `max_features`, `max_samples` |
| 4 | XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_lambda` |
| 5 | ANN | `hidden_layers`, `activation`, `dropout`, `learning_rate`, `weight_decay`, `batch_size` |

**Output per model** (di `models/<dataset>_<tag>/`):

| File | Isi |
|---|---|
| `*_rf.joblib`, `*_xgb.ubj`, `*_ann.pt` | model terlatih |
| `*_eval.json` | hyperparameter terbaik, ruang pencarian, metrik CV |
| `*_cv_predictions.csv` | prediksi OOF per baris + fold + residual |

**Gambar** di `outputs/figures/<tag>/`:

- `03_performa_oof_rf.png`, `04_performa_oof_xgb.png`, `05_performa_oof_ann.png`.

> Catatan ANN: target PCI diskalakan saat pelatihan, lalu dikembalikan ke skala 0–100 saat prediksi.

---

## 7. Stage 6 — Perbandingan & pemilihan model terbaik

**File:** `src/pci_ml/model.py` (`rank_models`), `src/pci_ml/evaluation.py`

Langkah:

1. Muat tiap model dan metrik CV-nya.
2. Prediksi data uji, hitung metrik test (MAE, RMSE, R²).
3. Simpan prediksi test ke `outputs/predictions/<tag>/`:
   - `pci-ml-26_rf_test_predictions.csv`, `..._xgb_...`, `..._ann_...`
   - berisi `road_info`, `actual`, `predicted`, `residual`.
4. **Peringkat model**:
   - dasar: **CV MAE terkecil**,
   - **aturan 1-SE**: model yang MAE-nya masih dalam `MAE_terbaik + SE` dianggap seri,
   - tie-break: **CV RMSE terkecil**.
5. Menampilkan tabel perbandingan dan ringkasan kriteria seleksi.

**Output:** `ranked` (daftar model terurut) dan model terbaik — pada run terakhir **XGBoost**.

> Data uji hanya dipakai untuk **pelaporan performa**, bukan untuk memilih model. Pemilihan model murni berbasis CV.

---

## 8. Stage 7 — Interpretasi SHAP

**File:** `src/pci_ml/shap.py`

Langkah:

1. **Muat model terbaik** dan hitung prediksi baseline pada kondisi **tanpa kerusakan** (semua fitur distress = 0). Baseline ini sekitar **99.94 PCI**.
2. **Pilih explainer**:
   - RF/XGBoost → `TreeExplainer` (eksak),
   - ANN → `DeepExplainer`, fallback `KernelExplainer`.
3. Hitung **nilai SHAP** pada seluruh data uji.
4. **Agregasi per jenis kerusakan**:
   - jumlahkan SHAP seluruh kolom keparahan (L/M/H) per segmen,
   - ambil rata-rata nilai absolut → `mean |SHAP|` (poin PCI),
   - hitung kontribusi relatif, arah (dari SHAP bertanda), dan jumlah kemunculan.
5. **Agregasi fitur rekayasa** (`TOTAL_DENSITY`, `TOTAL_L/M/H`) dilaporkan **terpisah**.
6. Simpan tabel nilai SHAP per segmen, ranking, metadata, dan gambar.

**Tampilan tabel di notebook** (label menggunakan `Mean |SHAP|`):

```
Peringkat | Jenis Kerusakan | Mean |SHAP| | Kontribusi | Arah
Peringkat | Fitur           | Mean |SHAP| | Kontribusi | Arah
```

**Output** di `outputs/shap/<tag>/`:

| File | Isi |
|---|---|
| `*_shap_values.csv` | nilai SHAP tiap segmen + prediksi + rekonstruksi |
| `*_shap_distress_type_ranking.csv` | ranking 19 jenis kerusakan |
| `*_shap_engineered_ranking.csv` | ranking fitur agregat |
| `*_shap_meta.json` | metadata (baseline, explainer, versi, top fitur, path) |

**Gambar** di `outputs/figures/<tag>/`:

- `06_shap_distress_type_importance_xgb.png`

Skema artefak SHAP saat ini: `format_version = 3` (tanpa kolom SHAP bertanda/penurunan PCI di tabel artefak; arah tetap dilaporkan).

Penjelasan konsep `mean |SHAP|` ada di `catatan/shap-feature-importance.md`.

---

## 9. Peta artefak per stage

| Stage | Artefak utama |
|---|---|
| 1 | DataFrame bersih (di memori) |
| 2 | `datasets/<dataset>/<tag>/train.csv`, `test.csv`, `meta.json`; `figures/01_distribusi_jenis_kerusakan.png` |
| 3 | `models/..._rf.joblib`, `..._rf_eval.json`, `..._rf_cv_predictions.csv`; `figures/03_performa_oof_rf.png` |
| 4 | `models/..._xgb.ubj`, `..._xgb_eval.json`, `..._xgb_cv_predictions.csv`; `figures/04_performa_oof_xgb.png` |
| 5 | `models/..._ann.pt`, `..._ann_eval.json`, `..._ann_cv_predictions.csv`; `figures/05_performa_oof_ann.png` |
| 6 | `outputs/predictions/<tag>/*_test_predictions.csv` |
| 7 | `outputs/shap/<tag>/*_shap_*.csv/json`; `figures/06_shap_distress_type_importance_*.png` |

---

## 10. Cara menjalankan & meregenerasi

### Menjalankan normal

Jalankan notebook dari **atas ke bawah** (Run All). Urutan ini penting karena stage selanjutnya membutuhkan variabel dan file dari stage sebelumnya.

### Syarat menjalankan stage tertentu tanpa dari awal

- Stage 2b (muat data) butuh checkpoint split di `datasets/<dataset>/<tag>/`.
- Stage 6 butuh model + `eval.json` dari stage 3–5.
- Stage 7 butuh model terbaik dan prediksi test dari stage 6.

### Regenerasi bersih dari nol

```bash
rm -rf models outputs
mkdir -p models outputs
# lalu jalankan seluruh notebook dari atas
```

`datasets/` tidak perlu dihapus; stage 2 akan memakai/menyegarkan checkpoint split.

---

## 11. Prinsip verifikasi yang dipakai

- **Reproducibility**: seed tetap (`SEED = 42`), split per jalan, CV per jalan.
- **Anti-kebocoran**: data uji tidak digunakan dalam tuning maupun pemilihan model.
- **Konsistensi artefak**: nama folder mengikuti `<tag>` yang sama di `datasets/`, `models/`, dan `outputs/`.
- **Interpretasi jujur**: `mean |SHAP|` = magnitudo kontribusi, bukan kausalitas; arah dilaporkan terpisah.

---

## 12. Ringkasan satu paragraf

Notebook ini memuat data PCI, membersihkannya, mengubah nilai kerusakan menjadi densitas per area, lalu menambahkan fitur agregat total. Data dipisah per jalan menjadi train/test dengan kriteria distribusi dan keberadaan PCI nol. Tiga model (Random Forest, XGBoost, ANN) dituning dengan random search dan dievaluasi memakai 5-fold GroupKFold berbasis jalan; model final dilatih pada seluruh data latih. Model terbaik dipilih dari CV MAE dengan aturan 1-SE, lalu dievaluasi pada data uji. Terakhir, SHAP digunakan untuk menjelaskan kontribusi tiap jenis kerusakan dan fitur agregat terhadap prediksi PCI.
