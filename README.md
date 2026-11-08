# PCI ML - Prediksi Pavement Condition Index

Notebook utama dan satu-satunya: **`pci-ml-26.ipynb`**.

Notebook ini memuat alur lengkap pembersihan data, rekayasa fitur, pemisahan data, tuning hyperparameter Random Forest dengan GroupKFold, pelatihan model final, dan evaluasi. Semua output dan `execution_count` sudah dikosongkan.

## Daftar sel

| No | Tipe | ID | Nama sel | Isi |
|---:|---|---|---|---|
| 0 | Markdown | `b7004c8d` | Style tampilan | CSS agar kode dan output panjang membungkus baris, tidak terpotong scroll horizontal. |
| 1 | Code | `6c3e1eb8` | Autoreload | Memuat `%autoreload` supaya perubahan modul di `src/pci_ml` langsung terpakai tanpa restart kernel. |
| 2 | Markdown | `c291079b` | Judul | Judul notebook: prediksi PCI dengan model Machine Learning (RF, XGBoost, ANN). |
| 3 | Markdown | `99129abd` | Bagian 1 | Judul **1. Pembersihan data dan rekayasa fitur**. |
| 4 | Code | `681b9681` | Pemilihan dataset | Menampilkan panel dataset dan parameter split, lalu preview data mentah. |
| 5 | Code | `364a2070` | Pembersihan dan fitur agregat | Membersihkan data, menghitung fitur density, menambahkan `TOTAL_DENSITY` dan `SEVERITY_WEIGHTED`, lalu preview data bersih. |
| 6 | Markdown | `3eb0df13` | Bagian 2 | Judul **2. Pemisahan data latih dan data uji**. |
| 7 | Code | `abd67a64` | Pemisahan data | Mencari seed split terbaik, memisahkan train/test, dan menyimpan checkpoint. |
| 8 | Markdown | `886fa03b` | Bagian 3 | Judul **3. Pelatihan model *machine learning***. |
| 9 | Code | `5c6e0f40` | Persiapan input model | Memuat checkpoint, menyiapkan `X_train`, `X_test`, `y_train`, `y_test`, dan grup jalan. |
| 10 | Code | `tune4f8a1c` | Tuning dan model final RF | Tuning hyperparameter, pelatihan model final, evaluasi OOF dan test, serta tabel hasil. |
| 11 | Markdown | `d313a54e` | Source code | Daftar modul dan fungsi pendukung di `src/pci_ml`. |

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

### Sel 2 - Judul (Markdown)

Menampilkan judul penelitian dan cakupan model: Random Forest, XGBoost, dan ANN.

### Sel 3 - Bagian 1 (Markdown)

Penanda bagian pembersihan data dan rekayasa fitur.

### Sel 4 - Pemilihan dataset (Code)

- Mengimpor fungsi dari `pci_ml.data`, `pci_ml.memory`, dan `pci_ml.ui`.
- Menyiapkan `DEFAULTS` untuk file, jumlah seed, test size, toleransi share, dan PCI minimum.
- `EXPORT_MODE = True` membuat panel ditampilkan sebagai teks, bukan widget.
- `RELEASE_MEMORY = True` mengaktifkan pembebasan memori otomatis.
- `build_panel` menampilkan pilihan dataset dan parameter split.
- `on_select` menampilkan preview data mentah dengan `format_dataframe_table`.
- Sel ini belum melakukan pembersihan; preview masih berisi data asli.

### Sel 5 - Pembersihan dan fitur agregat (Code)

- Mengambil nilai panel dari sel 4 (`cfg`, `filename`).
- `clean_data` menghapus duplikat, membatasi PCI ke 0-100, membagi fitur density dengan `SEGMENT_AREA`, dan membatasi nilainya ke 0-1.
- `add_aggregate_features` menambahkan:
  - `TOTAL_DENSITY`: jumlah seluruh fitur density.
  - `SEVERITY_WEIGHTED`: fitur density dibobot berdasarkan akhiran `_L` (1), `_M` (2), `_H` (3).
- Kedua fitur agregat disisipkan sebelum `PCI`, sehingga `PCI` tetap menjadi kolom terakhir.
- Menampilkan jumlah fitur (57 pada dataset ini) dan preview data bersih dengan format tabel yang sama seperti tabel hyperparameter.

### Sel 6 - Bagian 2 (Markdown)

Penanda bagian pemisahan data latih dan data uji.

### Sel 7 - Pemisahan data (Code)

- Mengimpor `split_progress` dari `pci_ml.ui`.
- Jika `df` belum ada, data dibersihkan ulang dan fitur agregat dihitung ulang.
- `split_data` mencari seed terbaik dengan `GroupShuffleSplit` berdasarkan:
  - jumlah seed: 200,
  - test size: 0.2,
  - toleransi share: 0.01,
  - PCI minimum: 0.
- Progres pencarian seed ditampilkan lewat `split_progress`.
- Hasil split disimpan sebagai checkpoint di `datasets/pci-ml-26/<tag>/`.
- Menampilkan ringkasan split: seed terpilih, jumlah baris train/test, dan jumlah jalan.
- Melepaskan `result` dan `df` dari memori setelah checkpoint tersimpan.

### Sel 8 - Bagian 3 (Markdown)

Penanda bagian pelatihan model.

### Sel 9 - Persiapan input model (Code)

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

### Sel 10 - Tuning dan model final Random Forest (Code)

- Menampilkan panel parameter tuning: jumlah kandidat (100) dan jumlah fold (5).
- Ruang pencarian hyperparameter:
  - `n_estimators`: 300,
  - `max_depth`: `None`, 20, 30,
  - `min_samples_split`: 2, 4, 6,
  - `min_samples_leaf`: 1, 2,
  - `max_features`: 0.1 sampai 0.4,
  - `max_samples`: 0.5 sampai 0.9.
- `tune_random_forest` menjalankan pencarian dengan `GroupKFold` dan memilih kandidat dengan MAE CV terendah.
- Model final dilatih pada seluruh data latih dengan hyperparameter terbaik.
- Evaluasi dilakukan dengan:
  - **CV pooled OOF**: prediksi out-of-fold dari semua fold digabung,
  - **Test**: evaluasi pada data uji yang tidak dipakai saat tuning.
- Menampilkan:
  - tabel hyperparameter terbaik,
  - tabel hasil evaluasi (MAE, RMSE, R²),
  - penjelasan metrik dan konteks holdout.

### Sel 11 - Source code (Markdown)

Menampilkan daftar modul pendukung:

- `src/pci_ml/data.py` - pembersihan data, rekayasa fitur agregat, split, dan checkpoint.
- `src/pci_ml/random_forest.py` - training, tuning, dan evaluasi Random Forest.
- `src/pci_ml/experiments.py` - fold, OOF, metrik, bobot, dan kalibrasi yang model-agnostic.
- `src/pci_ml/ui.py` - panel, progress split, dan tabel HTML.
- `src/pci_ml/memory.py` - manajemen memori notebook.

## Urutan menjalankan

Jalankan sel secara berurutan dari atas ke bawah:

1. Sel 1 untuk mengaktifkan autoreload.
2. Sel 4 untuk memilih dataset dan parameter split.
3. Sel 5 untuk membersihkan data dan menghitung fitur agregat.
4. Sel 7 untuk memisahkan data dan membuat checkpoint.
5. Sel 9 untuk memuat checkpoint dan menyiapkan input model.
6. Sel 10 untuk tuning, pelatihan model final, dan evaluasi.

Sel 7 dan sel 9 bergantung pada checkpoint. Jika parameter split diubah, jalankan ulang sel 7 sebelum sel 9.
