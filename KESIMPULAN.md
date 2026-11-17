# Kesimpulan Pipeline PCI-ML

Dokumen ini merangkum dua hal:

1. hasil pipeline utama di notebook (baseline), dan
2. hasil uji stabilitas model terhadap 100 variasi split data.

Seluruh metrik MAE/RMSE berada pada skala PCI 0–100, sehingga selisih 1.0 berarti satu poin PCI.

---

## 1. Kesimpulan Pipeline Utama (Baseline)

### 1.1 Konfigurasi

| Parameter | Nilai |
|---|---|
| Dataset | `pci-ml-26.xlsx` |
| Split tag | `n200_t0.2_s0.01_p0` |
| Split seed | 179 |
| Split gap | 0.7544 |
| Data latih | 933 baris, 27 jalan |
| Data uji | 244 baris, 7 jalan |
| Cross-validation | 5-fold GroupKFold berdasarkan `ROAD_ID` |
| Model seed | 42 |
| Kriteria seleksi | CV MAE terendah |

### 1.2 Hasil model

| Model | CV MAE | CV RMSE | CV R² | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---|---|
| **XGBoost** | **4.5091** | **7.2898** | **0.8987** | **4.0375** | 6.9880 | 0.9005 |
| ANN | 5.1079 | 7.8300 | 0.8831 | 4.2819 | **6.4014** | **0.9165** |
| Random Forest | 6.0248 | 9.0553 | 0.8437 | 5.0559 | 7.6827 | 0.8798 |

### 1.3 Pemilihan model

- **XGBoost terpilih** karena CV MAE terendah, yaitu **4.5091**.
- Selisih CV MAE terhadap peringkat kedua (ANN): **0.5988**.
- Selisih CV MAE terhadap Random Forest: **1.5157**.
- Urutan CV MAE dan test MAE sama: **XGBoost < ANN < Random Forest**.
- ANN memiliki test R² tertinggi (0.9165), tetapi test MAE-nya lebih buruk dari XGBoost. Sesuai kriteria yang disepakati, seleksi hanya memakai CV MAE; R² dan test MAE dipakai untuk pelaporan.

### 1.4 Catatan

- Search space XGBoost sekarang hanya memakai `n_estimators=300`. Dengan konfigurasi ini XGBoost tetap unggul jauh di CV maupun test.
- Hyperparameter terbaik yang dipakai:

| Model | Hyperparameter |
|---|---|
| RF | `n_estimators=300`, `max_depth=20`, `min_samples_split=4`, `min_samples_leaf=1`, `max_features=0.3`, `max_samples=0.9` |
| XGB | `n_estimators=300`, `max_depth=4`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.4`, `min_child_weight=1`, `gamma=0.3`, `reg_lambda=1` |
| ANN | `hidden_layers=(128, 64, 32)`, `activation=tanh`, `dropout=0.1`, `learning_rate=0.003`, `weight_decay=0.0001`, `batch_size=32` |

- Setiap model menyimpan input dan output evaluasinya:
  - `models/<tag>/pci-ml-26_<model>_eval.json` — hyperparameter, search space, prediksi out-of-fold, metrik CV/test, dan prediksi test.
  - `models/<tag>/pci-ml-26_<model>_predictions.csv` — nilai aktual, prediksi, dan residual per baris untuk CV dan test, lengkap dengan `ROAD_ID`, `ROAD_NAME`, dan `SEGMENT_ID`.

---

## 2. Kesimpulan Uji Stabilitas (100 Split Seed)

### 2.1 Setup

| Parameter | Nilai |
|---|---|
| Jumlah split diuji | 100 seed |
| Sumber seed | 637 seed valid dari pemindaian 20.000 seed |
| Seed split baseline | 179 (dikecualikan dari 100 seed uji) |
| Rentang gap | 0.217 – 19.329 |
| Pemilihan seed | gap besar → kecil, diambil merata |
| Model seed | 42 (sama untuk semua split) |
| Hyperparameter | sama dengan baseline |
| Cross-validation | 5-fold GroupKFold |

Distribusi gap 100 seed yang dipilih hampir identik dengan seluruh 637 seed valid (median 7.304 vs 7.294; mean 7.728 vs 7.696), sehingga rata-ratanya cukup representatif.

### 2.2 Ringkasan metrik

| Model | CV MAE mean ± std | Rentang CV MAE | Test MAE mean ± std | Rentang Test MAE |
|---|---|---|---|---|
| **XGBoost** | **4.4594 ± 0.2320** | 3.9091 – 5.1457 | **4.5026 ± 0.6218** | 3.3206 – 6.2473 |
| ANN | 5.1805 ± 0.3155 | 4.5514 – 6.3634 | 5.3360 ± 0.7714 | 4.0195 – 7.5712 |
| Random Forest | 5.9783 ± 0.2365 | 5.3752 – 6.5809 | 5.8866 ± 0.8598 | 4.0718 – 7.9652 |

### 2.3 Temuan utama

**XGBoost paling robust.**

- Menang CV MAE di **100/100 split**.
- Menang test MAE di **95/100 split**.
- Margin CV MAE terhadap peringkat kedua selalu positif: min 0.205, median 0.689, maks 1.304.
- Pada 5 split ketika XGBoost bukan yang terbaik di test, selisihnya sangat kecil: **0.0007–0.0907 MAE** (praktis seri).

**Urutan CV stabil, urutan test tidak.**

- CV MAE: `XGBoost > ANN > Random Forest` di **99/100 split**; sekali `XGBoost > RF > ANN`.
- Test MAE:
  - `XGBoost > ANN > RF`: 78 split
  - `XGBoost > RF > ANN`: 17 split
  - `ANN > XGBoost > RF`: 4 split
  - `RF > XGBoost > ANN`: 1 split

**Stabilitas berbeda antara CV dan test.**

- Variasi CV relatif: RF 4.0%, XGBoost 5.2%, ANN 6.1%. RF paling stabil secara relatif, tetapi level MAE-nya paling buruk.
- Variasi test: XGBoost std 0.622, ANN 0.771, RF 0.860. XGBoost paling stabil secara absolut.
- RF paling sensitif di test: MAE bergerak dari 4.07 sampai 7.97 (rentang 3.89).

**CV MAE bukan prediktor level test antar-split.**

Korelasi CV MAE vs test MAE di 100 split bernilai negatif:

| Model | Pearson | Spearman |
|---|---|---|
| Random Forest | −0.741 | −0.749 |
| XGBoost | −0.695 | −0.660 |
| ANN | −0.533 | −0.486 |

Split yang terlihat lebih baik di CV cenderung lebih buruk di test. CV MAE tetap baik untuk **memilih model**, tetapi tidak akurat untuk memperkirakan **level performa test** pada split tertentu.

### 2.4 Pengaruh gap split

Gap yang lebih besar cenderung **menurunkan CV MAE** tetapi **menaikkan test MAE**:

| Model | Korelasi gap vs CV MAE (Spearman) | Korelasi gap vs test MAE (Spearman) |
|---|---|---|
| Random Forest | −0.116 | +0.255 |
| XGBoost | −0.356 | +0.289 |
| ANN | −0.338 | +0.323 |

Rata-rata per kelompok gap:

| Kelompok gap | XGB CV | XGB test | ANN CV | ANN test | RF CV | RF test |
|---|---|---|---|---|---|---|
| Kecil | 4.5403 | 4.2900 | 5.2686 | 5.0491 | 5.9902 | 5.6169 |
| Sedang | 4.5055 | 4.4109 | 5.1892 | 5.2966 | 6.0317 | 5.8821 |
| Besar | 4.3299 | 4.8132 | 5.0809 | 5.6710 | 5.9126 | 6.1691 |

Pola ini konsisten untuk ketiga model: split dengan gap besar membuat CV terlihat sedikit lebih baik, tetapi performa test memburuk.

### 2.5 Baseline vs distribusi 100 split

| Model | Baseline CV MAE | Rata-rata CV 100 split | Baseline test MAE | Rata-rata test 100 split | Test lebih buruk dari baseline |
|---|---|---|---|---|---|
| RF | 6.0248 | 5.9783 | 5.0559 | 5.8866 | 81/100 split |
| XGBoost | 4.5091 | 4.4594 | 4.0375 | 4.5026 | 77/100 split |
| ANN | 5.1079 | 5.1805 | 4.2819 | 5.3360 | 91/100 split |

Posisi baseline dalam distribusi 100 split:

| Model | Split dengan CV MAE lebih baik dari baseline | Split dengan test MAE lebih baik dari baseline |
|---|---|---|
| RF | 59% | 19% |
| XGBoost | 61% | 23% |
| ANN | 41% | 9% |

CV baseline berada di sekitar tengah distribusi (41–61% split lebih baik), tetapi test baseline hanya dikalahkan oleh **9–23% split**. Artinya split seed 179 adalah split test yang relatif mudah, sehingga angka test di notebook **optimistis** dibanding rata-rata split lain.

### 2.6 Implikasi

- **Pemilihan XGBoost tidak sensitif terhadap variasi split.** XGBoost selalu menang CV dan hampir selalu menang test, dengan margin yang besar.
- **Estimasi performa test XGBoost sebaiknya memakai rata-rata antar-split: 4.50 ± 0.62**, bukan 4.04 dari satu split baseline.
- **CV MAE layak dipakai sebagai kriteria seleksi model**, karena urutan model sangat stabil. Namun CV MAE tidak boleh dipakai sebagai estimasi langsung performa test pada split tertentu.
- **ANN lebih baik dari RF pada CV** (peringkat kedua di 99/100 split), tetapi di test keduanya bergantian. ANN menempati peringkat kedua di 78 split, RF di 17 split.
- **RF paling sensitif terhadap perubahan split** dan konsisten berada di peringkat bawah.

### 2.7 Batasan

- Uji ini hanya memvariasikan split data. Model seed tetap 42 dan hyperparameter tidak divariasikan.
- Data uji hanya 244 baris dan 7 jalan, sehingga test MAE lebih berisik daripada CV (933 baris, 27 jalan).
- Pemilihan 100 seed dilakukan merata dari gap besar ke kecil; distribusinya representatif terhadap 637 seed valid, tetapi bukan sampel acak murni.
- Perbandingan baseline vs rata-rata split tidak sepenuhnya setara karena baseline dipilih dengan gap terkecil (0.7544), sedangkan split uji memiliki gap 0.217–19.329.
