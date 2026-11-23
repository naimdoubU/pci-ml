# Mean Absolute SHAP sebagai Ukuran Kontribusi Jenis Kerusakan

Dokumen ini menjelaskan **apa itu `mean |SHAP|`**, **mengapa dipakai** untuk mengukur besarnya kontribusi tiap jenis kerusakan terhadap prediksi PCI, **cara membacanya**, serta **catatan akademis** agar penggunaannya layak di skripsi.

---

## Ringkasan singkat (TL;DR)

- `mean |SHAP|` adalah **rata-rata besarnya (magnitudo) atribusi model** untuk suatu fitur/jenis kerusakan, dihitung dari seluruh data uji.
- Karena target model adalah **PCI**, nilainya bersatuan **poin PCI**. Contoh: `GATOR_CRACK = 4.83` berarti model rata-rata mengatribusikan sekitar **4.83 poin PCI** pada jenis kerusakan tersebut.
- `mean |SHAP|` mengukur **seberapa besar** pengaruh, **bukan** arahnya (naik/turun) dan **bukan** hubungan kausal.
- Metode ini **lazim dan dapat dipertahankan secara akademis**, dengan syarat definisi, baseline, cara agregasi, dan keterbatasannya dijelaskan.
- Di tabel tampilan, label disederhanakan menjadi **"Mean SHAP"**. Di narasi skripsi, gunakan notasi **`Mean |SHAP|`** agar tidak menyesatkan.

---

## 1. Apa itu SHAP?

SHAP (*SHapley Additive exPlanations*) membagi prediksi model menjadi kontribusi tiap fitur:

```
prediksi = base value + Σ nilai SHAP seluruh fitur
```

- **base value** = prediksi model pada kondisi acuan (*baseline*).
- **nilai SHAP** = berapa besar suatu fitur menggeser prediksi dari baseline, untuk **satu baris data**.
- Nilai SHAP **bertanda**: positif menaikkan prediksi, negatif menurunkan prediksi.
- Satuan nilai SHAP mengikuti satuan target, yaitu **poin PCI**.

Pada proyek ini baseline-nya adalah **jalan tanpa kerusakan** (semua fitur kerusakan = 0), dengan prediksi sekitar **99.94 PCI**. Jadi nilai SHAP dibaca sebagai: *"berapa poin PCI yang digeser oleh fitur ini relatif terhadap jalan tanpa kerusakan."*

---

## 2. Mengapa memakai `mean |SHAP|`?

Nilai SHAP asli bersifat **lokal** (per segmen jalan). Untuk kebutuhan skripsi, kita ingin ringkasan **global**: jenis kerusakan mana yang paling besar pengaruhnya secara umum. Caranya:

1. Ambil nilai absolut `|SHAP|` tiap baris → menghilangkan arah, menyisakan **besaran**.
2. Rata-ratakan seluruh baris data uji → `mean |SHAP|`.

Keunggulan:

| Aspek | Penjelasan |
|---|---|
| **Satuan intuitif** | Berada dalam poin PCI, bukan skor abstrak seperti *gain*. |
| **Berbasis teori permainan** | SHAP punya dasar aksiomatik (efisiensi, simetri, dsb.). |
| **Eksak untuk model tree** | XGBoost/Random Forest memakai `TreeExplainer` yang eksak dan memenuhi `prediksi = base + Σ SHAP`. |
| **Adil antar fitur** | Memperhitungkan interaksi dan skala tiap fitur. |
| **Lazim di literatur** | Disebut juga *SHAP feature importance* (Lundberg & Lee, 2017; Lundberg et al., 2020; Molnar, 2022). |

---

## 3. Definisi operasional di proyek ini

Satu jenis kerusakan diwakili **beberapa kolom** tingkat keparahan (L/M/H). Karena itu agregasinya:

1. Untuk tiap segmen `i` dan jenis kerusakan `j`, jumlahkan nilai SHAP seluruh kolom keparahan:

   ```
   S_ij = Σ_{k ∈ j} φ_ik
   ```

2. Ambil nilai absolut, lalu rata-ratakan seluruh segmen:

   ```
   Mean |SHAP|_j = (1/N) · Σ_i |S_ij|
   ```

3. Kontribusi relatif (share):

   ```
   Kontribusi_j = Mean |SHAP|_j / Σ_j Mean |SHAP|_j
   ```

**Catatan penting:** ini adalah **net effect** per jenis kerusakan. Jika kontribusi tingkat L dan M saling menghapus, nilainya bisa lebih kecil daripada penjumlahan `mean |SHAP|` per kolom. Definisi ini perlu ditulis di metodologi.

---

## 4. Cara membaca hasil (contoh nyata)

Model terbaik: **XGBoost**; data uji: **238 segmen**.

### Kontribusi jenis kerusakan (teratas)

| Peringkat | Jenis Kerusakan | Mean \|SHAP\| (poin PCI) | Kontribusi | Arah |
|---|---|---|---|---|
| 1 | GATOR_CRACK | 4.83 | 52.71% | negatif |
| 2 | POTHOLES | 2.27 | 24.79% | negatif |
| 3 | LT_CRACK | 0.50 | 5.47% | positif |

### Fitur agregat hasil rekayasa

| Peringkat | Fitur | Mean \|SHAP\| (poin PCI) | Kontribusi | Arah |
|---|---|---|---|---|
| 1 | TOTAL_DENSITY | 10.20 | 38.93% | negatif |
| 2 | TOTAL_H | 3.44 | 13.14% | negatif |
| 3 | TOTAL_M | 2.43 | 9.29% | negatif |
| 4 | TOTAL_L | 0.86 | 3.29% | negatif |

Cara membacanya:

- **Besaran** (`Mean |SHAP|`): makin besar, makin besar magnitudo pengaruhnya terhadap prediksi PCI.
- **Kontribusi**: porsi relatif terhadap total; seluruh jenis kerusakan berjumlah 100%.
- **Arah**: dari rata-rata SHAP **bertanda**; "negatif" berarti cenderung **menurunkan** PCI, "positif" berarti menaikkan, "netral" berarti nol, "n/a" berarti jenis kerusakan tidak muncul di data uji.

> Contoh narasi: *"GATOR_CRACK memiliki magnitudo kontribusi terbesar (4.83 poin PCI; 52.71% dari total), dengan arah negatif, artinya kehadirannya cenderung menurunkan prediksi PCI."*

---

## 5. Batasan yang harus disebut di skripsi

1. **Magnitudo, bukan kausalitas.** SHAP menjelaskan **perilaku model**, bukan hubungan sebab-akibat di lapangan. Gunakan kata "model mengatribusikan", bukan "menyebabkan".
2. **Bergantung baseline.** Nilai SHAP berubah bila baseline berubah. Di sini baseline = jalan tanpa kerusakan; nyatakan itu eksplisit.
3. **Bergantung model.** Interpretasi berlaku untuk model terpilih (XGBoost), bukan klaim universal tentang PCI.
4. **Korelasi antar fitur.** Kolom L/M/H saling berkorelasi; atribusi bisa terbagi. Agregasi per jenis kerusakan membantu meredam ini.
5. **Agregasi global bersifat heuristik.** Jaminan formal SHAP ada di level lokal; `mean |SHAP|` adalah cara meringkas yang lazim, bukan aksioma.
6. **Menyembunyikan heterogenitas.** Rata-rata tidak menunjukkan variasi antar segmen; sebaiknya tambahkan std/IQR atau *bootstrap confidence interval*.
7. **Label.** Nilai yang ditampilkan adalah `mean |SHAP|`; jangan menuliskannya sebagai "Mean SHAP" bertanda di narasi akademik.

---

## 6. Saran penguatan

- Tulis definisi operasional (Bagian 3) di metodologi.
- Laporkan **std** atau **bootstrap CI** untuk menunjukkan kestabilan peringkat.
- Lakukan **triangulasi** dengan *permutation importance* atau *partial dependence*.
- Sajikan juga **arah** (signed) secara terpisah agar interpretasi tidak berat sebelah.
- Bahas keterbatasan pada Bagian 5 secara eksplisit di bab pembahasan.

---

## 7. Rujukan

- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- Lundberg, S. M., Erion, G., Chen, H., et al. (2020). *From local explanations to global understanding with explainable AI for trees*. Nature Machine Intelligence, 2, 56–67.
- Molnar, C. (2022). *Interpretable Machine Learning* (2nd ed.).

---

## 8. Contoh paragraf untuk skripsi

> Kontribusi setiap jenis kerusakan diukur menggunakan rata-rata nilai absolut SHAP (`mean |SHAP|`) pada data uji. Untuk jenis kerusakan yang terdiri atas beberapa tingkat keparahan, nilai SHAP tiap kolom dijumlahkan terlebih dahulu per segmen, kemudian diambil nilai absolut dan dirata-ratakan. Karena target model adalah PCI, nilai ini dinyatakan dalam poin PCI dan dimaknai sebagai besarnya simpangan prediksi PCI yang diatribusikan model terhadap jenis kerusakan tersebut, relatif terhadap baseline jalan tanpa kerusakan. Arah pengaruh (positif/negatif) dilaporkan terpisah dari besarannya. Perlu ditegaskan bahwa ukuran ini merefleksikan perilaku model, bukan hubungan kausal, serta dipengaruhi oleh pilihan baseline dan korelasi antar fitur.
