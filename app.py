import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Decision Support System", page_icon="📊", layout="wide")

# Judul Utama
st.title("🎯 Sistem Pendukung Keputusan (DSS)")
st.markdown("*Metode: SAW, WP, AHP, TOPSIS*")
st.markdown("---")

# Sidebar untuk memilih metode
metode = st.sidebar.selectbox(
    "Pilih Metode DSS:",
    ["SAW (Simple Additive Weighting)", 
     "WP (Weighted Product)", 
     "AHP (Analytical Hierarchy Process)", 
     "TOPSIS"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Tentang Metode")

if "SAW" in metode:
    st.sidebar.info("SAW menggunakan penjumlahan berbobot untuk menentukan alternatif terbaik.")
elif "WP" in metode:
    st.sidebar.info("WP menggunakan perkalian untuk menghubungkan rating kriteria.")
elif "AHP" in metode:
    st.sidebar.info("AHP menggunakan perbandingan berpasangan untuk menentukan bobot.")
elif "TOPSIS" in metode:
    st.sidebar.info("TOPSIS memilih alternatif terdekat dengan solusi ideal positif.")

# ==================== FUNGSI MANUAL (TANPA LIBRARY METODE) ====================

def hitung_saw_manual(data, bobot, tipe):
    """Implementasi manual metode SAW
    
    Step 1: Normalisasi matriks X menjadi matriks R
    Step 2: Hitung nilai preferensi V dengan mengalikan R dengan bobot
    """
    n_alt, n_krit = len(data), len(data[0])
    
    # STEP 1: Normalisasi matriks X ke matriks R
    # Rumus normalisasi:
    # - Benefit: r_ij = x_ij / max(x_ij)
    # - Cost: r_ij = min(x_ij) / x_ij
    matriks_r = [[0 for _ in range(n_krit)] for _ in range(n_alt)]
    
    for j in range(n_krit):
        col = [data[i][j] for i in range(n_alt)]
        if tipe[j] == "Benefit":
            max_val = max(col)
            for i in range(n_alt):
                matriks_r[i][j] = data[i][j] / max_val if max_val > 0 else 0
        else:  # Cost
            min_val = min(col)
            for i in range(n_alt):
                matriks_r[i][j] = min_val / data[i][j] if data[i][j] > 0 else 0
    
    # STEP 2: Hitung nilai preferensi V
    # V_i = Σ (w_j * r_ij)
    nilai_v = []
    for i in range(n_alt):
        v = 0
        for j in range(n_krit):
            v += bobot[j] * matriks_r[i][j]
        nilai_v.append(v)
    
    return matriks_r, nilai_v

def hitung_wp_manual(data, bobot, tipe):
    """Implementasi manual metode WP"""
    n_alt, n_krit = len(data), len(data[0])
    
    # Perbaikan bobot (cost = negatif)
    w_perbaikan = []
    for j in range(n_krit):
        if tipe[j] == "Cost":
            w_perbaikan.append(-bobot[j])
        else:
            w_perbaikan.append(bobot[j])
    
    # Hitung vektor S
    vektor_s = []
    for i in range(n_alt):
        s = 1
        for j in range(n_krit):
            s *= data[i][j] ** w_perbaikan[j]
        vektor_s.append(s)
    
    # Hitung vektor V
    total_s = sum(vektor_s)
    vektor_v = [s / total_s if total_s > 0 else 0 for s in vektor_s]
    
    return w_perbaikan, vektor_s, vektor_v

def hitung_ahp_manual(matrix):
    """Implementasi manual metode AHP"""
    n = len(matrix)
    
    # Normalisasi matriks
    col_sums = [sum(matrix[i][j] for i in range(n)) for j in range(n)]
    matrix_norm = [[matrix[i][j] / col_sums[j] if col_sums[j] > 0 else 0 
                    for j in range(n)] for i in range(n)]
    
    # Hitung bobot (rata-rata baris)
    bobot = [sum(matrix_norm[i]) / n for i in range(n)]
    
    # Hitung lambda max
    weighted_sum = [sum(matrix[i][j] * bobot[j] for j in range(n)) for i in range(n)]
    lambda_vals = [weighted_sum[i] / bobot[i] if bobot[i] > 0 else 0 for i in range(n)]
    lambda_max = sum(lambda_vals) / n
    
    # Hitung CI dan CR
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0
    ri_table = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = ri_table.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0
    
    return matrix_norm, bobot, lambda_max, ci, cr

def hitung_topsis_manual(data, bobot, tipe):
    """Implementasi manual metode TOPSIS"""
    n_alt, n_krit = len(data), len(data[0])
    
    # Normalisasi
    data_norm = [[0 for _ in range(n_krit)] for _ in range(n_alt)]
    for j in range(n_krit):
        col = [data[i][j] for i in range(n_alt)]
        sum_kuadrat = sum(x ** 2 for x in col)
        pembagi = sum_kuadrat ** 0.5
        for i in range(n_alt):
            data_norm[i][j] = data[i][j] / pembagi if pembagi > 0 else 0
    
    # Terbobot
    data_weighted = [[data_norm[i][j] * bobot[j] for j in range(n_krit)] for i in range(n_alt)]
    
    # Solusi ideal
    ideal_pos = []
    ideal_neg = []
    for j in range(n_krit):
        col = [data_weighted[i][j] for i in range(n_alt)]
        if tipe[j] == "Benefit":
            ideal_pos.append(max(col))
            ideal_neg.append(min(col))
        else:
            ideal_pos.append(min(col))
            ideal_neg.append(max(col))
    
    # Jarak
    d_pos = []
    d_neg = []
    for i in range(n_alt):
        dp = sum((data_weighted[i][j] - ideal_pos[j]) ** 2 for j in range(n_krit)) ** 0.5
        dn = sum((data_weighted[i][j] - ideal_neg[j]) ** 2 for j in range(n_krit)) ** 0.5
        d_pos.append(dp)
        d_neg.append(dn)
    
    # Preferensi
    preferensi = [d_neg[i] / (d_pos[i] + d_neg[i]) if (d_pos[i] + d_neg[i]) > 0 else 0 
                  for i in range(n_alt)]
    
    return data_norm, data_weighted, ideal_pos, ideal_neg, d_pos, d_neg, preferensi

# ==================== METODE SAW ====================
if "SAW" in metode:
    st.header("📈 Simple Additive Weighting (SAW)")
    
    col1, col2 = st.columns(2)
    with col1:
        n_alt = st.number_input("Jumlah Alternatif", min_value=2, max_value=10, value=3)
    with col2:
        n_krit = st.number_input("Jumlah Kriteria", min_value=2, max_value=10, value=4)
    
    # Input nama alternatif
    st.subheader("Nama Alternatif")
    alternatif = []
    cols = st.columns(n_alt)
    for i in range(n_alt):
        with cols[i]:
            alternatif.append(st.text_input(f"A{i+1}", value=f"Alternatif {i+1}", key=f"alt_saw_{i}"))
    
    # Input nama kriteria dan bobot
    st.subheader("Kriteria dan Bobot")
    kriteria = []
    bobot = []
    tipe = []
    cols = st.columns(n_krit)
    for i in range(n_krit):
        with cols[i]:
            kriteria.append(st.text_input(f"C{i+1}", value=f"Kriteria {i+1}", key=f"krit_saw_{i}"))
            bobot.append(st.number_input(f"Bobot C{i+1}", min_value=0.0, max_value=1.0, value=0.25, step=0.05, key=f"bobot_saw_{i}"))
            tipe.append(st.selectbox(f"Tipe C{i+1}", ["Benefit", "Cost"], key=f"tipe_saw_{i}"))
    
    # Validasi bobot
    if abs(sum(bobot) - 1.0) > 0.01:
        st.warning(f"⚠ Total bobot = {sum(bobot):.2f}. Sebaiknya total bobot = 1.0")
    
    # Input data matriks keputusan
    st.subheader("Matriks Keputusan")
    data = []
    for i in range(n_alt):
        row = []
        cols = st.columns(n_krit)
        for j in range(n_krit):
            with cols[j]:
                val = st.number_input(f"{alternatif[i]} - {kriteria[j]}", min_value=0.0, value=1.0, key=f"data_saw_{i}_{j}")
                row.append(val)
        data.append(row)
    
    df = pd.DataFrame(data, columns=kriteria, index=alternatif)
    st.write("*Data Awal:*")
    st.dataframe(df)
    
    if st.button("Hitung SAW", type="primary"):
        # Tampilkan data asli dulu
        st.write("*Matriks X (Data Awal):*")
        st.dataframe(df)
        
        # STEP 1: Normalisasi matriks X ke matriks R
        st.subheader("📊 Step 1: Normalisasi Matriks (X → R)")
        st.latex(r"r_{ij} = \begin{cases} \frac{x_{ij}}{\max(x_{ij})} & \text{jika j adalah atribut benefit} \\ \frac{\min(x_{ij})}{x_{ij}} & \text{jika j adalah atribut cost} \end{cases}")
        
        # Tampilkan detail normalisasi per kriteria
        for j, krit in enumerate(kriteria):
            col_values = [data[i][j] for i in range(n_alt)]
            if tipe[j] == "Benefit":
                max_val = max(col_values)
                st.write(f"{krit} (Benefit):** max = {max_val}")
                for i, alt in enumerate(alternatif):
                    st.write(f"  r{i+1}{j+1} = {data[i][j]}/{max_val} = {data[i][j]/max_val:.4f}")
            else:
                min_val = min(col_values)
                st.write(f"{krit} (Cost):** min = {min_val}")
                for i, alt in enumerate(alternatif):
                    st.write(f"  r{i+1}{j+1} = {min_val}/{data[i][j]} = {min_val/data[i][j] if data[i][j] > 0 else 0:.4f}")
        
        matriks_r, nilai_v = hitung_saw_manual(data, bobot, tipe)
        
        df_r = pd.DataFrame(matriks_r, columns=kriteria, index=alternatif)
        st.write("*Matriks R (Hasil Normalisasi):*")
        st.dataframe(df_r.round(4))
        
        # STEP 2: Perhitungan nilai V
        st.subheader("📈 Step 2: Perhitungan Nilai Preferensi (V)")
        st.latex(r"V_i = \sum_{j=1}^{n} w_j \times r_{ij}")
        
        # Tampilkan detail perhitungan
        st.write("*Detail Perhitungan V:*")
        for i, alt in enumerate(alternatif):
            perhitungan = " + ".join([f"({bobot[j]:.2f} × {matriks_r[i][j]:.4f})" for j in range(n_krit)])
            st.write(f"V({alt}) = {perhitungan} = *{nilai_v[i]:.4f}*")
        
        # Hasil akhir
        st.subheader("🏆 Hasil Akhir SAW")
        hasil = pd.DataFrame({
            'Alternatif': alternatif,
            'Nilai V': nilai_v
        }).sort_values('Nilai V', ascending=False).reset_index(drop=True)
        hasil['Ranking'] = range(1, len(hasil) + 1)
        
        st.dataframe(hasil)
        st.success(f"🏆 Alternatif terbaik: *{hasil.iloc[0]['Alternatif']}* dengan nilai V = {hasil.iloc[0]['Nilai V']:.4f}")

# ==================== METODE WP ====================
elif "WP" in metode:
    st.header("🔢 Weighted Product (WP)")
    
    col1, col2 = st.columns(2)
    with col1:
        n_alt = st.number_input("Jumlah Alternatif", min_value=2, max_value=10, value=3)
    with col2:
        n_krit = st.number_input("Jumlah Kriteria", min_value=2, max_value=10, value=4)
    
    # Input nama alternatif
    st.subheader("Nama Alternatif")
    alternatif = []
    cols = st.columns(n_alt)
    for i in range(n_alt):
        with cols[i]:
            alternatif.append(st.text_input(f"A{i+1}", value=f"Alternatif {i+1}", key=f"alt_wp_{i}"))
    
    # Input nama kriteria dan bobot
    st.subheader("Kriteria dan Bobot")
    kriteria = []
    bobot = []
    tipe = []
    cols = st.columns(n_krit)
    for i in range(n_krit):
        with cols[i]:
            kriteria.append(st.text_input(f"C{i+1}", value=f"Kriteria {i+1}", key=f"krit_wp_{i}"))
            bobot.append(st.number_input(f"Bobot C{i+1}", min_value=0.0, max_value=1.0, value=0.25, step=0.05, key=f"bobot_wp_{i}"))
            tipe.append(st.selectbox(f"Tipe C{i+1}", ["Benefit", "Cost"], key=f"tipe_wp_{i}"))
    
    if abs(sum(bobot) - 1.0) > 0.01:
        st.warning(f"⚠ Total bobot = {sum(bobot):.2f}. Sebaiknya total bobot = 1.0")
    
    # Input data matriks keputusan
    st.subheader("Matriks Keputusan")
    data = []
    for i in range(n_alt):
        row = []
        cols = st.columns(n_krit)
        for j in range(n_krit):
            with cols[j]:
                val = st.number_input(f"{alternatif[i]} - {kriteria[j]}", min_value=0.1, value=1.0, key=f"data_wp_{i}_{j}")
                row.append(val)
        data.append(row)
    
    df = pd.DataFrame(data, columns=kriteria, index=alternatif)
    st.write("*Data Awal:*")
    st.dataframe(df)
    
    if st.button("Hitung WP", type="primary"):
        # Hitung manual
        w_perbaikan, vektor_s, vektor_v = hitung_wp_manual(data, bobot, tipe)
        
        st.write("*Bobot Perbaikan (Cost = negatif):*")
        st.write(pd.DataFrame({'Kriteria': kriteria, 'Bobot Asli': bobot, 'Bobot Perbaikan': w_perbaikan}))
        
        hasil = pd.DataFrame({
            'Alternatif': alternatif,
            'Vektor S': vektor_s,
            'Vektor V': vektor_v
        }).sort_values('Vektor V', ascending=False).reset_index(drop=True)
        hasil['Ranking'] = range(1, len(hasil) + 1)
        
        st.write("*Hasil Perhitungan WP:*")
        st.dataframe(hasil)
        st.success(f"🏆 Alternatif terbaik: *{hasil.iloc[0]['Alternatif']}* dengan nilai V = {hasil.iloc[0]['Vektor V']:.4f}")

# ==================== METODE AHP ====================
elif "AHP" in metode:
    st.header("⚖ Analytical Hierarchy Process (AHP)")
    
    col1, col2 = st.columns(2)
    with col1:
        n_krit = st.number_input("Jumlah Kriteria", min_value=2, max_value=7, value=3)
    with col2:
        n_alt = st.number_input("Jumlah Alternatif", min_value=2, max_value=10, value=3)
    
    st.subheader("Nama Kriteria")
    kriteria = []
    cols = st.columns(n_krit)
    for i in range(n_krit):
        with cols[i]:
            kriteria.append(st.text_input(f"C{i+1}", value=f"Kriteria {i+1}", key=f"krit_ahp_{i}"))
    
    st.subheader("Nama Alternatif")
    alternatif = []
    cols = st.columns(n_alt)
    for i in range(n_alt):
        with cols[i]:
            alternatif.append(st.text_input(f"A{i+1}", value=f"Alternatif {i+1}", key=f"alt_ahp_{i}"))
    
    # STEP 1: Perbandingan Berpasangan Kriteria
    st.subheader("📊 Step 1: Matriks Perbandingan Berpasangan Kriteria")
    st.markdown("*Skala Saaty:* 1=Sama penting, 3=Sedikit lebih penting, 5=Lebih penting, 7=Sangat penting, 9=Mutlak lebih penting")
    
    matrix_kriteria = [[1.0 for _ in range(n_krit)] for _ in range(n_krit)]
    
    for i in range(n_krit):
        for j in range(i + 1, n_krit):
            val = st.number_input(
                f"{kriteria[i]} vs {kriteria[j]}", 
                min_value=0.0,
                value=1.0, 
                step=0.1,
                key=f"ahp_krit_{i}_{j}",
                help=f"Seberapa penting {kriteria[i]} dibanding {kriteria[j]}?"
            )
            matrix_kriteria[i][j] = float(val) if val > 0 else 1.0
            matrix_kriteria[j][i] = 1.0 / val if val > 0 else 1.0
    
    df_matrix_kriteria = pd.DataFrame(matrix_kriteria, columns=kriteria, index=kriteria)
    st.write("*Matriks Perbandingan Kriteria:*")
    st.dataframe(df_matrix_kriteria.round(4))
    
    # STEP 2: Perbandingan Berpasangan Alternatif untuk Setiap Kriteria
    st.subheader("📊 Step 2: Matriks Perbandingan Berpasangan Alternatif (per Kriteria)")
    
    matrices_alternatif = []
    for k in range(n_krit):
        with st.expander(f"🔍 Perbandingan Alternatif untuk {kriteria[k]}"):
            matrix_alt = [[1.0 for _ in range(n_alt)] for _ in range(n_alt)]
            
            for i in range(n_alt):
                for j in range(i + 1, n_alt):
                    val = st.number_input(
                        f"{alternatif[i]} vs {alternatif[j]}", 
                        min_value=0.0,
                        value=1.0, 
                        step=0.1,
                        key=f"ahp_alt_{k}{i}{j}",
                        help=f"Untuk {kriteria[k]}, seberapa baik {alternatif[i]} dibanding {alternatif[j]}?"
                    )
                    matrix_alt[i][j] = float(val) if val > 0 else 1.0
                    matrix_alt[j][i] = 1.0 / val if val > 0 else 1.0
            
            df_matrix_alt = pd.DataFrame(matrix_alt, columns=alternatif, index=alternatif)
            st.write(f"*Matriks Perbandingan Alternatif untuk {kriteria[k]}:*")
            st.dataframe(df_matrix_alt.round(4))
            
            matrices_alternatif.append(matrix_alt)
    
    if st.button("Hitung AHP", type="primary"):
        st.subheader("🧮 Hasil Perhitungan AHP")
        
        # Hitung bobot kriteria
        st.write("### 1️⃣ Bobot Kriteria")
        matrix_krit_norm, bobot_kriteria, lambda_max_krit, ci_krit, cr_krit = hitung_ahp_manual(matrix_kriteria)
        
        st.write("*Matriks Ternormalisasi Kriteria:*")
        st.dataframe(pd.DataFrame(matrix_krit_norm, columns=kriteria, index=kriteria).round(4))
        
        df_bobot_krit = pd.DataFrame({
            'Kriteria': kriteria,
            'Bobot': bobot_kriteria,
            'Bobot (%)': [b * 100 for b in bobot_kriteria]
        })
        st.dataframe(df_bobot_krit.round(4))
        
        st.write(f"*λ max:* {lambda_max_krit:.4f}, *CI:* {ci_krit:.4f}, *CR:* {cr_krit:.4f}")
        if cr_krit <= 0.1:
            st.success("✅ Matriks kriteria konsisten (CR ≤ 0.1)")
        else:
            st.error("❌ Matriks kriteria tidak konsisten (CR > 0.1)")
        
        # Hitung prioritas alternatif untuk setiap kriteria
        st.write("### 2️⃣ Prioritas Alternatif per Kriteria")
        prioritas_alternatif = []
        
        for k in range(n_krit):
            matrix_alt_norm, bobot_alt, lambda_max_alt, ci_alt, cr_alt = hitung_ahp_manual(matrices_alternatif[k])
            prioritas_alternatif.append(bobot_alt)
            
            with st.expander(f"📋 Detail {kriteria[k]}"):
                st.write("*Matriks Ternormalisasi:*")
                st.dataframe(pd.DataFrame(matrix_alt_norm, columns=alternatif, index=alternatif).round(4))
                
                df_prioritas = pd.DataFrame({
                    'Alternatif': alternatif,
                    'Prioritas': bobot_alt
                })
                st.dataframe(df_prioritas.round(4))
                
                st.write(f"*λ max:* {lambda_max_alt:.4f}, *CI:* {ci_alt:.4f}, *CR:* {cr_alt:.4f}")
                if cr_alt <= 0.1:
                    st.success("✅ Konsisten")
                else:
                    st.error("❌ Tidak konsisten")
        
        # STEP 3: Perhitungan Skor Akhir (Perkalian Matriks)
        st.write("### 3️⃣ Skor Akhir (Prioritas Alternatif × Bobot Kriteria)")
        
        # Buat matriks prioritas (alternatif x kriteria)
        matriks_prioritas = []
        for i in range(n_alt):
            row = [prioritas_alternatif[j][i] for j in range(n_krit)]
            matriks_prioritas.append(row)
        
        df_matriks_prioritas = pd.DataFrame(matriks_prioritas, columns=kriteria, index=alternatif)
        st.write("*Matriks Prioritas Alternatif:*")
        st.dataframe(df_matriks_prioritas.round(4))
        
        # Hitung skor akhir
        skor_akhir = []
        for i in range(n_alt):
            skor = sum(matriks_prioritas[i][j] * bobot_kriteria[j] for j in range(n_krit))
            skor_akhir.append(skor)
        
        st.write("*Detail Perhitungan Skor:*")
        for i, alt in enumerate(alternatif):
            perhitungan = " + ".join([f"({matriks_prioritas[i][j]:.4f} × {bobot_kriteria[j]:.4f})" for j in range(n_krit)])
            st.write(f"Skor({alt}) = {perhitungan} = *{skor_akhir[i]:.4f}*")
        
        # Hasil akhir dengan ranking
        hasil = pd.DataFrame({
            'Alternatif': alternatif,
            'Skor Akhir': skor_akhir
        }).sort_values('Skor Akhir', ascending=False).reset_index(drop=True)
        hasil['Ranking'] = range(1, len(hasil) + 1)
        
        st.write("### 🏆 Hasil Akhir AHP")
        st.dataframe(hasil)
        st.success(f"🏆 Alternatif terbaik: *{hasil.iloc[0]['Alternatif']}* dengan skor {hasil.iloc[0]['Skor Akhir']:.4f}")

# ==================== METODE TOPSIS ====================
elif "TOPSIS" in metode:
    st.header("🎯 TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)")
    
    col1, col2 = st.columns(2)
    with col1:
        n_alt = st.number_input("Jumlah Alternatif", min_value=2, max_value=10, value=3)
    with col2:
        n_krit = st.number_input("Jumlah Kriteria", min_value=2, max_value=10, value=4)
    
    # Input nama alternatif
    st.subheader("Nama Alternatif")
    alternatif = []
    cols = st.columns(n_alt)
    for i in range(n_alt):
        with cols[i]:
            alternatif.append(st.text_input(f"A{i+1}", value=f"Alternatif {i+1}", key=f"alt_topsis_{i}"))
    
    # Input nama kriteria dan bobot
    st.subheader("Kriteria dan Bobot")
    kriteria = []
    bobot = []
    tipe = []
    cols = st.columns(n_krit)
    for i in range(n_krit):
        with cols[i]:
            kriteria.append(st.text_input(f"C{i+1}", value=f"Kriteria {i+1}", key=f"krit_topsis_{i}"))
            bobot.append(st.number_input(f"Bobot C{i+1}", min_value=0.0, max_value=1.0, value=0.25, step=0.05, key=f"bobot_topsis_{i}"))
            tipe.append(st.selectbox(f"Tipe C{i+1}", ["Benefit", "Cost"], key=f"tipe_topsis_{i}"))
    
    if abs(sum(bobot) - 1.0) > 0.01:
        st.warning(f"⚠ Total bobot = {sum(bobot):.2f}. Sebaiknya total bobot = 1.0")
    
    # Input data matriks keputusan
    st.subheader("Matriks Keputusan")
    data = []
    for i in range(n_alt):
        row = []
        cols = st.columns(n_krit)
        for j in range(n_krit):
            with cols[j]:
                val = st.number_input(f"{alternatif[i]} - {kriteria[j]}", min_value=0.0, value=1.0, key=f"data_topsis_{i}_{j}")
                row.append(val)
        data.append(row)
    
    df = pd.DataFrame(data, columns=kriteria, index=alternatif)
    st.write("*Data Awal:*")
    st.dataframe(df)
    
    if st.button("Hitung TOPSIS", type="primary"):
        # Hitung manual
        data_norm, data_weighted, ideal_pos, ideal_neg, d_pos, d_neg, preferensi = hitung_topsis_manual(data, bobot, tipe)
        
        st.write("*Matriks Ternormalisasi:*")
        st.dataframe(pd.DataFrame(data_norm, columns=kriteria, index=alternatif).round(4))
        
        st.write("*Matriks Ternormalisasi Terbobot:*")
        st.dataframe(pd.DataFrame(data_weighted, columns=kriteria, index=alternatif).round(4))
        
        st.write("*Solusi Ideal:*")
        st.write(pd.DataFrame({
            'Kriteria': kriteria,
            'Ideal Positif (A+)': ideal_pos,
            'Ideal Negatif (A-)': ideal_neg
        }))
        
        hasil = pd.DataFrame({
            'Alternatif': alternatif,
            'D+': d_pos,
            'D-': d_neg,
            'Preferensi': preferensi
        }).sort_values('Preferensi', ascending=False).reset_index(drop=True)
        hasil['Ranking'] = range(1, len(hasil) + 1)
        
        st.write("*Hasil Perhitungan TOPSIS:*")
        st.dataframe(hasil)
        st.success(f"🏆 Alternatif terbaik: *{hasil.iloc[0]['Alternatif']}* dengan nilai preferensi {hasil.iloc[0]['Preferensi']:.4f}")

st.markdown("---")
st.markdown("💡 *Tips:* Gunakan metode berbeda untuk membandingkan hasil dan memvalidasi keputusan Anda.")
st.markdown("✅ *Semua perhitungan menggunakan implementasi manual tanpa library khusus metode DSS*")