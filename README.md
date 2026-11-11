# Contact Angle Analyzer

Web app untuk mengukur sudut kontak (*sessile drop*) langsung dari foto: tanpa menandai lingkaran manual seperti di ImageJ.

## Menjalankan

Windows: klik dua kali `run_app.bat`, lalu buka http://localhost:8501.
PC lain di jaringan lab bisa membuka `http://<IP-PC-ini>:8501`.

Manual:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\streamlit run app.py
```

## Alur pemakaian

1. **Upload**: isi nama sampel (mis. `Entri 0.3`), lalu seret foto ke area upload. Semua foto langsung dianalisis
   paralel. Di tabel: centang foto (atau *Pilih semua* / *Pilih menurut sampel*) untuk mengganti nama sampel atau
   menghapus sekaligus. *Hapus semua* mengosongkan sesi untuk mulai baru.
2. **Atur**: pindah foto lewat thumbnail (warna = keyakinan), tombol Sebelumnya/Berikutnya, atau panah keyboard;
   filter *Perlu dicek* / *Gagal* untuk fokus ke foto bermasalah. Di foto: scroll untuk zoom, seret untuk menggeser.
   Koreksi:
   - *Lokasi tetesan*: klik tetesan yang benar atau seret kotak di sekelilingnya; deteksi diulang di area itu.
   - *Baseline*: klik titik kontak kiri dan kanan pada permukaan substrat.
   - *Titik tepi*: klik minimal 6 titik di tepi tetesan (kaca pembesar membantu), lalu *Hitung*.
3. **Hasil**: rata-rata ± SD per sampel, lalu *Perbandingan* antar kelompok foto pilihan (Δ, uji t Welch, grafik
   dengan titik tiap foto). *Bagikan dan unduh*: laporan HTML satu file (bisa dikirim lewat email/chat), Excel, CSV,
   grafik, ZIP foto beranotasi, ringkasan teks untuk chat, dan *Data latih* (foto + koreksi + hasil).

Mode terang/gelap di kanan atas; pilihannya tersimpan di URL (`?tema=`).

## Peringatan

Setiap foto mendapat skor **keyakinan** dan temuan dalam dua kelompok:

- **Dari pengambilan foto**: penyebabnya cara memotret (kamera dari atas, miring, kurang fokus, silau, tetesan
  terlalu kecil, resolusi rendah). Ini tidak bisa diperbaiki software; foto perlu diambil ulang.
- **Dari analisis**: kontur hanya terdeteksi sebagian, metode fitting berbeda jauh, atau hasil sensitif terhadap
  baseline. Periksa overlay atau gunakan koreksi.

## Metode

| Langkah | Cara |
|---|---|
| Baseline | Garis lurus dengan transisi terang→gelap terkuat (proyeksi gradien multi-sudut), diperhalus sub-piksel di sekitar tetesan |
| Kontur | Rantai tepi Canny → hipotesis busur yang menyentuh baseline → dinilai dari dukungan tepi di sepanjang busur → diperhalus & dikunci ke tepi terluar (siluet) → profil sub-piksel di sepanjang normal |
| Sudut | **Elips** (utama, seperti plugin ImageJ), **Lingkaran**, **Tangen lokal** (polinomial dekat titik kontak), **Tinggi/lebar** (θ = 2·atan(2h/d)) |
| Ketidakpastian | ±°/px: perubahan sudut bila baseline digeser 1 piksel |

Bila elips menghasilkan ujung tetesan yang tidak wajar secara fisik (jauh di atas estimasi lingkaran), metode utama
otomatis beralih ke lingkaran dan diberi peringatan.

## Struktur

```
app.py               antarmuka Streamlit (Upload → Atur → Hasil)
ca_ui/clicker/       komponen gambar interaktif (canvas, kaca pembesar, tanpa build tool)
ca_core/
  baseline.py        deteksi & penghalusan baseline
  detect.py          deteksi kontur tetesan
  geometry.py        fitting elips/lingkaran/tangen & perhitungan sudut
  quality.py         pemeriksaan kualitas foto & analisis
  analyze.py         pipeline utama + overlay
  worker.py          entry point untuk analisis di proses paralel
scripts/run_samples.py   uji batch pada folder samples/ (overlay disimpan ke output/)
samples/             foto contoh
```

## Validasi

Belum divalidasi terhadap nilai ImageJ karena foto mentah dengan nilai acuan belum tersedia. Langkah berikutnya:
ukur 10–20 foto yang sama di ImageJ dan di app ini, lalu bandingkan selisihnya (target ±2°).
