USER GUIDE — Decision Support System (DSS) App

=== Deskripsi ===
Aplikasi ini merupakan aplikasi Decision Support System berbasis web yang dibangun menggunakan Streamlit.
Tujuannya adalah membantu pengguna menentukan alternatif terbaik berdasarkan beberapa kriteria dan bobot menggunakan empat metode yang dapat dipilih:
- SAW (Simple Additive Weighting)
- WP (Weighted Product)
- AHP (Analytical Hierarchy Process)
- TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)

Setiap metode dihitung secara manual (tanpa library metode) dan hasilnya akan ditampilkan.

=== Persiapan Sebelum Menjalankan ===
1. Instal pandas:
    pip install streamlit pandas numpy
2. Buka terminal di folder tempat file app.py berada, lalu jalankan:
    streamlit run app.py

===== Cara Menggunakan Setiap Metode =====
--- SAW (Simple Additive Weighting) ---
Langkah-langkah:
1. Pilih metode SAW di sidebar.
2. Masukkan jumlah alternatif dan kriteria.
3. Isi nama alternatif dan nama kriteria.
4. Tentukan bobot dan tipe kriteria (Benefit / Cost).
5. Masukkan nilai matriks keputusan.
6. Klik Hitung SAW.

Hasil yang ditampilkan:
1. Matriks awal
2. Matriks normalisasi (R)
3. Nilai preferensi (V)
4. Ranking alternatif

--- WP (Weighted Product) ---
Langkah-langkah:
1. Pilih metode WP.
2. Masukkan jumlah alternatif & kriteria.
3. Isi nama alternatif, kriteria, bobot, dan tipe (Benefit/Cost).
4. Masukkan nilai matriks keputusan.
5. Klik Hitung WP.

Hasil yang ditampilkan:
1. Nilai vektor S
2. Nilai vektor V
3. Ranking alternatif

--- AHP (Analytical Hierarchy Process) ---
Langkah-langkah:
1. Pilih metode AHP.
2. Masukkan jumlah kriteria & alternatif.
3. Isi matriks perbandingan berpasangan antar kriteria (skala Saaty: 1–9).
4. Untuk setiap kriteria, isi perbandingan antar alternatif.
5. Klik Hitung AHP.

Hasil yang ditampilkan:
1. Bobot tiap kriteria
2. Nilai konsistensi (CI & CR)
3. Bobot alternatif per kriteria
4. Nilai akhir dan ranking

--- TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) ---
Langkah-langkah:
1. Pilih metode TOPSIS.
2. Masukkan jumlah alternatif & kriteria.
3. Isi nama, bobot, dan tipe (Benefit/Cost).
4. Masukkan nilai matriks keputusan.
5. Klik Hitung TOPSIS.

Hasil yang ditampilkan:
1. Matriks normalisasi
2. Matriks terbobot
3. Solusi ideal positif & negatif
4. Jarak ke solusi ideal (D⁺ dan D⁻)
5. Nilai preferensi dan ranking

Tips
- Total bobot sebaiknya = 1.0
- Gunakan data numerik positif
- Gunakan metode yang sama untuk perbandingan hasil