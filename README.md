# PCI ML - Prediksi Pavement Condition Index

Notebook utama dan satu-satunya: **`pci-ml-26.ipynb`**.

Notebook ini memuat alur lengkap pembersihan data, rekayasa fitur, pemisahan data, tuning hyperparameter Random Forest dengan GroupKFold, pelatihan model final, dan evaluasi. Semua output dan `execution_count` sudah dikosongkan.

## Daftar sel

| No | Tipe | ID | Nama sel | Isi |
|---:|---|---|---|---|
| 0 | Markdown | `b7004c8d` | Style tampilan | CSS agar kode dan output panjang membungkus baris, tidak terpotong scroll horizontal. |
| 1 | Code | `6c3e1eb8` | Autoreload | Memuat `%autoreload` supaya perubahan modul di `src/pci_ml` langsung terpakai tanpa restart kernel. |
| 2 | Code | `config-cell` | **Konfigurasi utama** | Semua parameter (file, split, tuning, model seed, memori) didefinisikan di sini. **Ubah nilai di sini sebelum menjalankan notebook.** |
| 3 | Markdown | `c291079b` | Judul | Judul notebook: prediksi PCI dengan model Machine Learning (RF, XGBoost, ANN). |
| 4 | Markdown | `99129abd` | Bagian 1 | Judul **1. Pembersihan data dan rekayasa fitur**. |
| 5 | Code | `681b9681` | Memuat dataset | Memuat file dataset dan menampilkan preview data mentah. |
| 6 | Code | `364a2070` | Pembersihan dan fitur agregat | Membersihkan data, menghitung fitur density, menambahkan `TOTAL_DENSITY` dan `SEVERITY_WEIGHTED`, lalu preview data bersih. |
| 7 | Markdown | `3eb0df13` | Bagian 2 | Judul **2. Pemisahan data latih dan data uji**. |
| 8 | Code | `abd67a64` | Pemisahan data | Mencari seed split terbaik, memisahkan train/test, dan menyimpan checkpoint. |
| 9 | Markdown | `886fa03b` | Bagian 3 | Judul **3. Pelatihan model *machine learning***. |
| 10 | Code | `5c6e0f40` | Persiapan input model | Memuat checkpoint, menyiapkan `X_train`, `X_test`, `y_train`, `y_test`, dan grup jalan. |
| 11 | Code | `tune4f8a1c` | Tuning dan model final RF | Tuning hyperparameter, pelatihan model final, evaluasi OOF dan test, serta tabel hasil. |
| 12 | Markdown | `d313a54e` | Source code | Daftar modul dan fungsi pendukung di `src/pci_ml`. |

## Konfigurasi (Sel 2)

Sel **config-cell** berisi semua parameter yang bisa diubah:

```python
# Dataset (None = gunakan file pertama di datasets/pci-ml-26/)
FILENAME = None

# Parameter split data
N_SEED = 200
TEST_SIZE = 0.2
SHARE_TOL = 0.01
PCI_MIN = 0

# Parameter tuning Random Forest
TUNING_N_ITER = 100
TUNING_CV_SPLITS = 5

# Model seed (RF, XGBoost, ANN) - terpisah dari split seed
SEED = 42

# Memori
RELEASE_MEMORY = True
```

Ubah nilai di sel ini, lalu jalankan ulang seluruh notebook dari awal.

## Penjelasan tiap sel

### Sel 0 - Style tampilan (Markdown)

Mengatur CSS notebook agar blok kode dan output HTML/Markdown membungkus teks panjang. Ini menjaga tabel dan output tetap terbaca tanpa scroll horizontal.

### Sel 1 - Autoreload (Code)

Menjalankan:

```python
%load_ext autoreload
%autoreload 2
```

Berguna saat mengubah file di `src/pci_ml`; modul akan dimuat ulang otomatis.

### Sel 2 - Konfigurasi utama (Code)

Semua parameter didefinisikan di sini. **Ini adalah satu-satunya tempat yang perlu diubah** untuk mengubah dataset atau parameter. Tidak ada lagi widget dropdown atau textbox.

### Sel 3 - Judul (Markdown)

Menampilkan judul penelitian dan cakupan model: Random Forest, XGBoost, dan ANN.

### Sel 4 - Bagian 1 (Markdown)

Penanda bagian pembersihan data dan rekayasa fitur.

### Sel 5 - Memuat dataset (Code)

- Mengimpor fungsi dari `pci_ml.data`, `pci_ml.memory`, dan `pci_ml.ui`.
- Menentukan nama file: `FILENAME` jika diisi, jika tidak gunakan file pertama.
- Menampilkan preview data mentah dengan `format_dataframe_table`.

### Sel 6 - Pembersihan dan fitur agregat (Code)

- `clean_data` menghapus duplikat, membatasi PCI ke 0-100, membagi fitur density dengan `SEGMENT_AREA`, dan membatasi nilainya ke 0-1.
- `add_aggregate_features` menambahkan:
  - `TOTAL_DENSITY`: jumlah seluruh fitur density.
  - `SEVERITY_WEIGHTED`: fitur density dibobot berdasarkan akhiran `_L` (1), `_M` (2), `_H` (3).
- Kedua fitur agregat disisipkan sebelum `PCI`, sehingga `PCI` tetap menjadi kolom terakhir.
- Menampilkan jumlah fitur (57 pada dataset ini) dan preview data bersih dengan format tabel yang sama seperti tabel hyperparameter.

### Sel 7 - Bagian 2 (Markdown)

Penanda bagian pemisahan data latih dan data uji.

### Sel 8 - Pemisahan data (Code)

- Mengimpor `split_progress` dari `pci_ml.ui`.
- Jika `df` belum ada, data dibersihkan ulang dan fitur agregat dihitung ulang.
- `split_data` mencari seed terbaik dengan `GroupShuffleSplit` berdasarkan parameter dari sel config.
- Progres pencarian seed ditampilkan lewat `split_progress`.
- Hasil split disimpan sebagai checkpoint di `datasets/pci-ml-26/<tag>/`.
- Menampilkan ringkasan split: seed terpilih, jumlah baris train/test, dan jumlah jalan.
- Melepaskan `result` dan `df` dari memori setelah checkpoint tersimpan.

### Sel 9 - Bagian 3 (Markdown)

Penanda bagian pelatihan model.

### Sel 10 - Persiapan input model (Code)

- Menentukan `SEED = 42` untuk model, terpisah dari seed split.
- Memuat checkpoint terbaru untuk file yang dipilih.
- Memvalidasi bahwa checkpoint sudah memuat fitur agregat.
- Menyiapkan:
  - `X_train`, `X_test` sebagai `float32`,
  - `y_train`, `y_test` dari kolom `PCI`,
  - `groups_train` dari `ROAD_ID` untuk GroupKFold,
  - `test_ids` untuk informasi jalan pada data uji.
- Memeriksa tidak ada NaN/inf.
- Menampilkan peringatan jika parameter split saat ini berbeda dari parameter checkpoint.
- Menampilkan tag checkpoint dan ringkasan bentuk train/test.

### Sel 11 - Tuning dan model final Random Forest (Code)

- Menampilkan parameter tuning dari sel config: `TUNING_N_ITER` dan `TUNING_CV_SPLITS`.
- Ruang pencarian hyperparameter tetap sama.
- `tune_random_forest` menjalankan pencarian dengan `GroupKFold` dan memilih kandidat dengan MAE CV terendah.
- Model final dilatih pada seluruh data latih dengan hyperparameter terbaik.
- Evaluasi dilakukan dengan:
  - **CV pooled OOF**: prediksi out-of-fold dari semua fold digabung, dihitung dengan `evaluate_oof`,
  - **Test**: evaluasi pada data uji yang tidak dipakai saat tuning, dihitung dengan `evaluate_predictions`.
- Menampilkan:
  - tabel hyperparameter terbaik,
  - tabel hasil evaluasi (MAE, RMSE, R²),
  - penjelasan metrik dan konteks holdout.

### Sel 12 - Source code (Markdown)

Menampilkan daftar modul pendukung:

- `src/pci_ml/data.py` - pembersihan data, rekayasa fitur agregat, split, dan checkpoint.
- `src/pci_ml/random_forest.py` - factory estimator dan tuning Random Forest.
- `src/pci_ml/experiments.py` - fold, OOF, metrik, bobot, dan kalibrasi yang model-agnostic.
- `src/pci_ml/ui.py` - tabel HTML dan progress split.
- `src/pci_ml/memory.py` - manajemen memori notebook.

## Urutan menjalankan

Jalankan sel secara berurutan dari atas ke bawah:

1. Sel 1 untuk mengaktifkan autoreload.
2. **Sel 2 untuk mengatur konfigurasi** (ubah nilai jika perlu).
3. Sel 5 untuk memuat dataset.
4. Sel 6 untuk membersihkan data dan menghitung fitur agregat.
5. Sel 8 untuk memisahkan data dan membuat checkpoint.
6. Sel 10 untuk memuat checkpoint dan menyiapkan input model.
7. Sel 11 untuk tuning, pelatihan model final, dan evaluasi.

Sel 8 dan sel 10 bergantung pada checkpoint. Jika parameter split diubah di sel 2, jalankan ulang sel 8 sebelum sel 10.