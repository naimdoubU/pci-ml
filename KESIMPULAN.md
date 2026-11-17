# Kesimpulan Pipeline PCI-ML

Dokumen ini merangkum dua bagian dengan peran berbeda.

1. Bagian 1, hasil pipeline utama di notebook. Bagian ini dipakai untuk memilih model dan melaporkan performa pada satu split.
2. Bagian 2, uji stabilitas pada 100 variasi split acak. Bagian ini dipakai sebagai bukti pendukung bahwa pilihan model tidak bergantung pada satu split.

Keduanya bukan estimasi performa bebas bias. Alasannya dijelaskan di Bagian 2.8.

Seluruh metrik MAE dan RMSE berada pada skala PCI 0 sampai 100, sehingga selisih 1.0 berarti satu poin PCI.

Catatan dataset: file mentah berisi 1177 baris dan 34 jalan. Ada 55 fitur dasar. Setelah fitur turunan, jumlah fitur menjadi 59. Pembagian data menjadi 939 baris (27 jalan) untuk latih dan 238 baris (7 jalan) untuk uji. Fitur turunan tidak memakai PCI dan tidak memakai data uji.

---

## 1. Hasil Pipeline Utama (Pemilihan Model)

### 1.1 Konfigurasi

| Parameter | Nilai |
|---|---|
| Dataset | `pci-ml-26.xlsx` |
| Split tag | `n100_t0.2_s0.01_p0` |
| Split seed | 39 |
| Split gap | 11.6266 |
| Data latih | 939 baris, 27 jalan |
| Data uji | 238 baris, 7 jalan |
| Cross-validation | 5-fold GroupKFold berdasarkan `ROAD_ID` |
| Model seed | 42 |
| Kriteria seleksi | CV MAE terendah |

Catatan pemilihan split: seed 39 adalah seed valid pertama pada pemindaian seed 0 sampai 99. Gap tidak lagi dipakai untuk memilih seed. Gap hanya dihitung dan disimpan di `meta.json` sebagai deskripsi distribusi. Kriteria valid yang dipakai tetap tiga:

1. Ukuran data uji 20% dengan toleransi 1%.
2. Tidak ada fitur yang hanya muncul di data uji.
3. Data uji memuat setidaknya satu baris dengan PCI 0.

Kriteria 2 dan 3 memakai data uji, sehingga pemilihan seed masih memakai informasi data uji. Ini dibahas di Bagian 2.8.

### 1.2 Aturan seleksi

- Model dipilih dengan CV MAE terendah.
- CV RMSE dipakai hanya jika CV MAE seri.
- Ambang seri belum ditetapkan di kode. Angka ambang harus ditetapkan peneliti sebelum aturan ini dijalankan.
- Pada hasil sekarang tidak ada seri. Margin CV MAE adalah 0.8318 terhadap ANN dan 1.4625 terhadap Random Forest.
- Data uji tidak dipakai untuk memilih model. Data uji hanya dipakai untuk pelaporan.

### 1.3 Hasil model

| Model | CV MAE | CV RMSE | CV R² | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---|---|
| **XGBoost** | **4.8721** | **7.8018** | **0.8769** | **3.3389** | **5.6231** | **0.9377** |
| ANN | 5.7039 | 8.8995 | 0.8398 | 4.6466 | 6.2966 | 0.9219 |
| Random Forest | 6.3346 | 9.3120 | 0.8246 | 4.4072 | 6.8724 | 0.9069 |

### 1.4 Pemilihan model

- **XGBoost terpilih** karena CV MAE terendah, yaitu **4.8721**.
- Selisih CV MAE terhadap peringkat kedua (ANN): **0.8318**.
- Selisih CV MAE terhadap Random Forest: **1.4625**.
- Urutan CV MAE: **XGBoost < ANN < Random Forest**. Urutan ini sama untuk CV RMSE dan CV R².
- Pada test split ini, XGBoost juga unggul MAE (3.3389), RMSE (5.6231), dan R² (0.9377).
- Meskipun begitu, keunggulan test di atas hanya berlaku untuk satu split. Uji 100 split di Bagian 2 menunjukkan angka test baseline ini optimistis dibanding rata-rata split acak.

### 1.5 Hyperparameter dan artefak

| Model | Hyperparameter |
|---|---|
| RF | `n_estimators=300`, `max_depth=20`, `min_samples_split=4`, `min_samples_leaf=1`, `max_features=0.3`, `max_samples=0.9` |
| XGB | `n_estimators=300`, `max_depth=5`, `learning_rate=0.1`, `subsample=0.6`, `colsample_bytree=0.7`, `min_child_weight=5`, `gamma=0`, `reg_lambda=3` |
| ANN | `hidden_layers=(64, 32)`, `activation=relu`, `dropout=0.1`, `learning_rate=0.0005`, `weight_decay=0.001`, `batch_size=32` |

Setiap model menyimpan input dan output evaluasinya:

- `models/<tag>/pci-ml-26_<model>_eval.json`: hyperparameter, search space, prediksi out-of-fold, metrik CV dan test, serta prediksi test.
- `models/<tag>/pci-ml-26_<model>_predictions.csv`: nilai aktual, prediksi, dan residual per baris untuk CV dan test, lengkap dengan `ROAD_ID`, `ROAD_NAME`, dan `SEGMENT_ID`.

Catatan search space XGBoost: `n_estimators` dibatasi pada 300. Hyperparameter yang dipakai di sini adalah hasil tuning pada split baseline, lalu dipakai ulang tanpa tuning ulang di 100 split uji stabilitas.

---

## 2. Uji Stabilitas (100 Split Acak)

### 2.1 Setup

| Parameter | Nilai |
|---|---|
| Jumlah split diuji | 100 seed |
| Sumber seed | 637 seed valid dari pemindaian 20.000 seed |
| Pemilihan seed | acak dari seed valid, tanpa melihat gap |
| Seed sampling split | 42 (reproducible) |
| Seed split baseline | 39 (dikecualikan dari 100 seed uji) |
| Rentang gap 100 seed uji | 0.3193 sampai 17.8282 |
| Median gap 100 seed uji | 7.4265 |
| Model seed | 42 (sama untuk semua split) |
| Hyperparameter | sama dengan baseline, tanpa tuning ulang |
| Cross-validation | 5-fold GroupKFold |

Pemilihan 100 seed dilakukan dengan `numpy.random.default_rng(42).choice` tanpa pengembalian, dari 637 seed valid selain seed 39. Gap tidak dipakai pada tahap pemilihan. Distribusi gap yang muncul murni akibat sampling acak: minimum 0.3193, kuartil bawah 4.6631, median 7.4265, kuartil atas 10.3842, maksimum 17.8282, mean 7.8805. Gap baseline 11.6266 berada di sekitar persentil 78, jadi baseline bukan split bergap kecil.

Ukuran data uji pada 100 split berkisar 224 sampai 247 baris, dan setiap split selalu memuat 7 jalan uji.

Kriteria seed valid tetap tiga, termasuk kriteria 2 dan 3 yang memakai data uji. Gap tidak lagi menjadi kriteria seleksi, tetapi kriteria 2 dan 3 membuat pemilihan seed belum sepenuhnya bebas dari data uji.

Hyperparameter untuk 100 split diambil dari tuning pada split baseline, bukan tuning ulang per split. Akibatnya ada kebocoran tidak langsung: sebagian jalan uji pada tiap split juga muncul di data latih baseline yang dipakai untuk tuning.

### 2.2 Ringkasan metrik

| Model | CV MAE mean ± std | Rentang CV MAE | Test MAE mean ± std | Rentang Test MAE |
|---|---|---|---|---|
| **XGBoost** | **4.5985 ± 0.2043** | 4.0666 sampai 5.0284 | **4.4847 ± 0.6542** | 3.1825 sampai 6.2374 |
| ANN | 5.8202 ± 0.4255 | 4.9483 sampai 8.0116 | 5.6943 ± 1.0671 | 4.0810 sampai 12.1158 |
| Random Forest | 5.9842 ± 0.2406 | 5.5002 sampai 6.7250 | 5.8270 ± 0.8250 | 4.2792 sampai 7.6363 |

Rata-rata test RMSE dan test R²:

| Model | Test RMSE mean ± std | Test R² mean ± std |
|---|---|---|
| XGBoost | 7.1718 ± 0.9143 | 0.8976 ± 0.0333 |
| ANN | 8.8202 ± 1.4965 | 0.8438 ± 0.0662 |
| Random Forest | 8.7280 ± 1.0274 | 0.8499 ± 0.0405 |

### 2.3 Temuan utama

**XGBoost paling robust pada MAE.**

- Menang CV MAE di **100/100 split**.
- Menang test MAE di **99/100 split**.
- Menang test RMSE dan test R² masing-masing di **98/100 split**.
- Margin CV MAE terhadap peringkat kedua selalu positif: min 0.5204, median 1.1186, maks 1.7644.
- Satu-satunya split ketika XGBoost bukan yang terbaik di test adalah seed 16450: ANN 4.6189 berbanding XGBoost 4.6596, selisih 0.0406 MAE, praktis seri.

**Urutan CV stabil di puncak, urutan bawah tidak.**

- CV MAE: XGBoost selalu peringkat pertama. Urutan `XGBoost > ANN > RF` pada 68 split dan `XGBoost > RF > ANN` pada 32 split.
- Test MAE: `XGBoost > ANN > RF` pada 68 split, `XGBoost > RF > ANN` pada 31 split, dan `ANN > XGBoost > RF` pada 1 split.
- Posisi kedua antara ANN dan RF bergantian, tetapi selisih rata-ratanya kecil: CV 5.8202 berbanding 5.9842, test 5.6943 berbanding 5.8270. ANN sedikit lebih baik pada kedua rata-rata.

**Stabilitas berbeda antara CV dan test.**

- Variasi relatif CV MAE: RF 4.0%, XGBoost 4.4%, ANN 7.3%. RF paling stabil secara relatif, tetapi level MAE-nya paling buruk.
- Variasi test: XGBoost std 0.654, RF 0.825, ANN 1.067. XGBoost paling stabil secara absolut.
- ANN paling sensitif di test: MAE bergerak dari 4.0810 sampai 12.1158, rentang 8.0348.

### 2.4 Hubungan CV dan test

Korelasi CV MAE dan test MAE di 100 split bernilai negatif:

| Model | Pearson | Spearman |
|---|---|---|
| Random Forest | -0.776 | -0.792 |
| XGBoost | -0.649 | -0.608 |
| ANN | -0.215 | -0.201 |

Korelasi negatif ini bukan bukti bahwa CV tidak valid. Data CV dan data uji saling lepas. Split yang membuat komposisi fold CV lebih sulit dapat membuat data uji lebih mudah, dan sebaliknya. Karena itu CV MAE tetap baik untuk **memilih model**, tetapi tidak akurat untuk memprediksi **level performa test pada split tertentu**.

### 2.5 Pengaruh gap split

Korelasi gap dengan metrik lemah dan tidak konsisten arahnya:

| Model | Korelasi gap vs CV MAE (Spearman) | Korelasi gap vs test MAE (Spearman) |
|---|---|---|
| Random Forest | -0.040 | +0.186 |
| XGBoost | -0.347 | +0.180 |
| ANN | +0.071 | +0.257 |

Korelasi gap vs test MAE positif tetapi lemah. Nilai p untuk XGBoost 0.074 dan RF 0.063, hanya ANN yang di bawah 0.05. Arah korelasi CV juga tidak seragam. Jadi gap bukan proksi yang andal untuk tingkat kesulitan test, dan keputusan tidak memakai gap untuk seleksi split didukung oleh data ini.

Kelompok gap dibentuk dari kuantil gap 100 seed: kecil 0.3193 sampai 5.3641, sedang 5.3641 sampai 9.4925, besar 9.4925 sampai 17.8282.

| Kelompok gap | XGB CV | XGB test | ANN CV | ANN test | RF CV | RF test |
|---|---|---|---|---|---|---|
| Kecil | 4.6542 | 4.3469 | 5.7592 | 5.3759 | 5.9541 | 5.6920 |
| Sedang | 4.6784 | 4.3404 | 5.8504 | 5.6314 | 6.0674 | 5.6600 |
| Besar | 4.4611 | 4.7711 | 5.8527 | 6.0852 | 5.9319 | 6.1332 |

Pola test sedikit meningkat pada kelompok gap besar, tetapi kelompok kecil dan sedang hampir sama. Pola ini tidak cukup kuat untuk memakai gap sebagai kriteria seleksi.

### 2.6 Baseline dan distribusi 100 split

| Model | Baseline CV MAE | Rata-rata CV 100 split | Baseline test MAE | Rata-rata test 100 split | Test lebih buruk dari baseline |
|---|---|---|---|---|---|
| RF | 6.3346 | 5.9842 | 4.4072 | 5.8270 | 99/100 split |
| XGBoost | 4.8721 | 4.5985 | 3.3389 | 4.4847 | 99/100 split |
| ANN | 5.7039 | 5.8202 | 4.6466 | 5.6943 | 83/100 split |

Posisi baseline dalam distribusi 100 split:

| Model | Split dengan CV MAE lebih baik dari baseline | Split dengan test MAE lebih baik dari baseline |
|---|---|---|
| RF | 94% | 1% |
| XGBoost | 91% | 1% |
| ANN | 44% | 17% |

Pola yang muncul berbeda antara CV dan test:

- **CV baseline cenderung lebih sulit dari rata-rata.** Untuk RF dan XGBoost, 91% sampai 94% split acak punya CV MAE lebih baik dari baseline. Artinya data latih baseline bukan data latih yang mudah.
- **Test baseline justru lebih mudah dari hampir semua split acak.** Hanya 1% split yang punya test MAE lebih baik dari baseline untuk RF dan XGBoost. Test MAE XGBoost baseline 3.3389 sangat dekat dengan nilai minimum 100 split, yaitu 3.1825.
- **Gap tidak menjelaskan kemudahan ini.** Gap baseline 11.6266 termasuk besar (persentil 78), tetapi test-nya paling mudah. Ini bukti tambahan bahwa gap bukan ukuran kesulitan test.

Karena itu angka test di notebook, terutama XGBoost 3.3389, adalah deskripsi satu split yang relatif mudah, bukan estimasi performa yang representatif. Estimasi yang lebih jujur adalah rata-rata 100 split: **4.4847 ± 0.6542** untuk XGBoost.

### 2.7 Implikasi

- **Pemilihan XGBoost tidak sensitif terhadap variasi split.** XGBoost selalu menang CV MAE dan hampir selalu menang test MAE, dengan margin CV yang selalu positif.
- **Estimasi performa test XGBoost sebaiknya memakai rata-rata antar-split: 4.48 ± 0.65**, bukan 3.34 dari satu split baseline. Angka ini masih mengandung bias dari hyperparameter yang dituning hanya pada split baseline.
- **CV MAE layak dipakai sebagai kriteria seleksi model**, karena XGBoost selalu peringkat pertama pada CV. Namun CV MAE tidak boleh dipakai sebagai estimasi langsung performa test pada split tertentu.
- **ANN sedikit lebih baik dari RF pada rata-rata**, baik CV (5.8202 berbanding 5.9842) maupun test (5.6943 berbanding 5.8270), tetapi urutan keduanya tidak stabil antar-split.
- **RF paling sensitif terhadap perubahan split di test** dan konsisten berada di peringkat bawah.

### 2.8 Hal yang perlu dipertimbangkan

- **Kriteria 2 dan 3 memakai data uji.** Kriteria "tidak ada fitur hanya di test" memakai fitur uji, dan kriteria "test harus memuat PCI 0" memakai label uji. Keduanya wajib, sehingga pemilihan seed tetap memakai informasi data uji. Yang sudah dihapus hanya kriteria gap terkecil, jadi bias seleksi berkurang tetapi belum nol.
- **Gap tidak lagi dipakai memilih seed.** Gap hanya dilaporkan. Pemilihan seed sekarang memakai seed valid pertama, dan uji stabilitas memakai sampel acak dari seed valid.
- **Tuning hanya dilakukan pada split baseline.** Hyperparameter yang sama dipakai untuk 100 split. Sebagian jalan uji pada tiap split juga muncul di data latih baseline yang dipakai tuning, sehingga estimasi performa 100 split tidak sepenuhnya bebas kebocoran.
- **Baseline punya keuntungan tuning.** Hyperparameter dipilih pada data latih baseline, sehingga baseline sedikit diuntungkan dibanding split lain yang memakai hyperparameter tersebut tanpa tuning ulang.
- **Split tidak independen.** Semua split memakai 34 jalan yang sama. Satu jalan bisa menjadi data uji di banyak split dan data latih di split lain. Std antar-split karena itu bukan ukuran variasi yang independen.
- **Model seed tunggal.** Semua hasil memakai model seed 42. Variasi akibat inisialisasi dan pengacakan model tidak diukur.
- **Data uji kecil.** Test hanya 224 sampai 247 baris dan selalu 7 jalan, sehingga test MAE lebih berisik daripada CV yang memakai sekitar 930 baris dan 27 jalan.
- **Label PCI bersifat manual.** Nilai PCI berasal dari penilaian manual, sehingga noise label tidak diukur dan tidak masuk ke metrik.
- **Stabilitas yang diuji hanya MAE sebagai fokus.** RMSE dan R² per split tersimpan dan dilaporkan sebagai ringkasan, tetapi tidak dijadikan fokus uji stabilitas.
- **Baseline dan rata-rata split tidak sepenuhnya setara.** Baseline adalah seed valid pertama, sedangkan 100 split adalah sampel acak dari seed valid dan tidak memasukkan seed 39.

### 2.9 Batasan

- Uji ini hanya memvariasikan split data. Model seed tetap 42 dan hyperparameter tidak divariasikan.
- Pemilihan 100 seed dibatasi pada seed yang lolos kriteria valid. Seed yang tidak lolos tidak diuji.
- Perbandingan baseline dan rata-rata split tidak sepenuhnya setara karena baseline dipilih sebagai seed valid pertama dan hyperparameter dituning pada baseline.
- Kesimpulan stabilitas berlaku untuk rentang gap 0.3193 sampai 17.8282 dan untuk dataset ini, bukan untuk dataset lain.
