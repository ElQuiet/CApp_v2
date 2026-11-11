---
title: Contact Angle Analyzer
emoji: 💧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Ukur sudut kontak sessile drop dari foto
---

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

## Deploy online (Hugging Face Spaces)

App berjalan di server Hugging Face, jadi bisa dibuka dari mana saja tanpa PC lab yang menyala. Setiap
`git push` ke Space otomatis membangun ulang app (±5–10 menit) memakai `Dockerfile`.

Sekali saja:

1. Buat akun di https://huggingface.co, lalu **New Space** → SDK **Docker** (template *Blank*), hardware
   *CPU basic* (gratis). Pilih **Private** bila kode/model tidak boleh terlihat publik.
2. Settings Space → **Variables and secrets** → *New secret* `CA_PASSWORD` = password untuk tim lab.
3. Buat *access token* (Settings akun → Access Tokens, jenis **Write**) untuk dipakai sebagai password git.
4. Di folder proyek:
   ```bash
   git remote add space https://huggingface.co/spaces/<username>/<nama-space>
   git push space main
   ```
   Saat diminta: username = username Hugging Face, password = access token tadi.

Update berikutnya cukup `git add -A`, `git commit -m "..."`, `git push space main`.

Catatan:
- Space gratis "tidur" bila tidak dibuka beberapa waktu; pembukaan pertama sesudahnya butuh ±1 menit.
- Foto yang diupload diproses di server Hugging Face (tidak disimpan app, hilang saat Space restart).
  Untuk data yang tidak boleh keluar lab, jalankan lokal dengan `run_app.bat`.
- Space **private** hanya bisa dibuka akun Hugging Face yang punya akses (undang anggota lewat
  *Organization*). Space **public** + `CA_PASSWORD`: siapa pun dengan password bisa memakai app, tetapi
  file kode & model terlihat publik.

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
| Lokasi tetesan | Foto diperkecil → hingga 4 kandidat lokasi tetesan → tiap kandidat diukur ulang pada potongan **resolusi penuh** → **model pemilih** (terlatih) memilih yang benar-benar tetesan, bukan kilau/kotoran |
| Baseline | Garis lurus dengan transisi terang→gelap terkuat (proyeksi gradien multi-sudut), diperhalus sub-piksel di sekitar tetesan |
| Kontur | Rantai tepi Canny → hipotesis busur yang menyentuh baseline → dinilai dari dukungan tepi di sepanjang busur → diperhalus & dikunci ke tepi terluar (siluet) → profil sub-piksel di sepanjang normal |
| Kandidat sudut | **Elips**, **Lingkaran**, **Tangen lokal**, **Tinggi/lebar**, fit busur atas saja, baseline di ujung kontur, baseline melalui ujung kiri–kanan siluet (kamera dari atas) |
| Sudut utama | **Model (ImageJ)**: koreksi terlatih terhadap median kandidat, dibatasi ±25°; lihat *Model terlatih* |
| Keyakinan | Peringatan kualitas foto + **model kesalahan** (perkiraan selisih terhadap ImageJ) |
| Ketidakpastian | ±°/px: perubahan sudut bila baseline digeser 1 piksel |

Bila user mengatur baseline atau titik tepi secara manual, model tidak dipakai: hasilnya murni geometri dari input
user.

## Model terlatih

Foto lab sering memuat masalah yang ambigu untuk geometri murni: kamera sedikit dari atas sehingga **pantulan
tetesan** di permukaan pelat ikut terlihat, kaki tetesan gelap/berbayang, tetesan sudut rendah tampak sebagai
elips tertutup, dan banyak objek kecil (kilau, kotoran) yang mirip busur. Model kecil (gradient boosting, disimpan
sebagai pohon keputusan JSON di `ca_core/model/`, cukup numpy untuk dipakai) dilatih dari pengukuran manual ImageJ:

- **ranker**: memilih kandidat lokasi yang merupakan tetesan;
- **angle**: koreksi sudut terhadap median kandidat geometri (dibatasi ±25°, tanpa alasan kuat mengikuti geometri);
- **error**: perkiraan selisih → skor keyakinan.

Data latih: 210 foto berlabel (18 sampel, tahap baseline / sebelum & sesudah pre-degreasing / sesudah ECL) dari
folder `KARAKTERISASI` (nilai ImageJ *Contact Angle* Theta E atau LBADSA), plus 200 tetesan sintetis dengan sudut
pasti agar model tidak "merusak" tetesan yang bersih.

Melatih ulang (mis. setelah ada data baru):

```bash
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python scripts\build_reference.py "C:\...\KARAKTERISASI"     # nilai ImageJ → data/reference_imagej.csv
.venv\Scripts\python scripts\benchmark.py "C:\...\KARAKTERISASI" --tag baru  # kandidat & ciri per foto
.venv\Scripts\python scripts\synthetic_hyps.py 200                          # data sintetis
.venv\Scripts\python scripts\train_model.py baru                            # validasi silang + simpan model
.venv\Scripts\python -m pytest                                               # perbarui EXPECTED bila sengaja berubah
```

Folder/sheet Excel baru perlu ditambahkan ke daftar `BLOCKS` di `scripts/build_reference.py`.

## Struktur

```
app.py               antarmuka Streamlit (Upload → Atur → Hasil)
ca_ui/clicker/       komponen gambar interaktif (canvas, kaca pembesar, tanpa build tool)
ca_core/
  baseline.py        deteksi & penghalusan baseline
  detect.py          deteksi kontur tetesan (beberapa hipotesis)
  geometry.py        fitting elips/lingkaran/tangen & perhitungan sudut
  candidates.py      kandidat pengukuran sudut + ciri geometri
  learned.py         model terlatih (pemilih, sudut, kesalahan); model/*.json
  synthetic.py       foto tetesan sintetis dengan sudut pasti (uji & augmentasi)
  quality.py         pemeriksaan kualitas foto & analisis
  analyze.py         pipeline utama + overlay
  worker.py          entry point untuk analisis di proses paralel
data/reference_imagej.csv   nilai acuan ImageJ per foto
scripts/             build_reference, benchmark, synthetic_hyps, train_model, run_samples
tests/               uji regresi, akurasi sintetis, model, ekspor
samples/             foto contoh
```

## Validasi

Dibandingkan dengan pengukuran manual ImageJ pada 210 foto lab (folder `KARAKTERISASI`, November 2026).
Angka di bawah dari **validasi silang per sampel**: tiap sampel diprediksi oleh model yang dilatih tanpa foto
sampel itu, jadi mencerminkan foto baru.

| | Selisih rata-rata | Median | ≤ 5° | ≤ 10° |
|---|---|---|---|---|
| Geometri saja (versi sebelumnya) | 18,6° | 12,5° | 21% | 40% |
| **Dengan model** | **12,3°** | **8,5°** | **36%** | **53%** |

Rata-rata per sampel & tahap (37 kelompok): selisih 10,4° (sebelumnya 16,4°).
Skor keyakinan berarti: foto dengan perkiraan selisih terendah (sepertiga teratas) rata-rata meleset 8°,
sepertiga terbawah 19°.

Pada tetesan sintetis dengan sudut pasti (bukan data latih): selisih rata-rata 0,7° (geometri saja 0,8°),
92% dalam ±2°.

Batasan:
- Nilai ImageJ sendiri bervariasi ±6° (SD) antar titik dalam satu sampel dan sebagian label meragukan
  (7 ditandai `ref_ok = 0` di `data/reference_imagej.csv` dan tidak dipakai).
- Sisa kesalahan terbesar: deteksi memilih objek lain (kilau, kotoran) pada foto yang ramai, dan sudut 40–70°
  (hanya 12 foto latih). Periksa foto dengan keyakinan rendah lewat overlay di langkah Atur.
- Foto dengan kamera dari atas tetap lebih baik difoto ulang; sudut untuk kasus ini hanya perkiraan.
