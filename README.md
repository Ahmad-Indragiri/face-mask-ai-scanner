# Real-time AI Mask Detection (TensorFlow.js)

Sebuah aplikasi berbasis web untuk mendeteksi penggunaan masker secara *real-time* menggunakan kamera (*webcam*) atau unggahan foto. Proyek ini dibangun dengan **TensorFlow.js** dan memanfaatkan model **MobileNetV2** melalui teknik *Transfer Learning*.

## Fitur Utama
- **Hybrid Detection:** Mendeteksi masker langsung via kamera atau melalui file gambar (JPG/PNG).
- **Graph Model Architecture:** Menggunakan format *Graph Model* untuk performa yang lebih cepat dan stabil di browser.
- **Responsive UI:** Antarmuka gelap yang modern dan responsif untuk berbagai perangkat.
- **Privacy First:** Proses pemrosesan AI dilakukan sepenuhnya di sisi klien (*client-side*), tidak ada data gambar yang dikirim ke server.

## Teknologi yang Digunakan
- **Frontend:** HTML5, CSS3 (Modern UI), Vanilla JavaScript.
- **Deep Learning Framework:** [TensorFlow.js](https://www.tensorflow.org/js).
- **Base Model:** MobileNetV2 (Pre-trained on ImageNet).
- **Training Environment:** Kaggle Notebook.

## Struktur Proyek
```text
.
├── index.html          # Antarmuka pengguna (UI)
├── script.js           # Logika utama AI & penanganan webcam
├── model/              # Folder penyimpanan model AI
│   ├── model.json      # Arsitektur Graph Model
│   └── group1-shard1of1.bin # Bobot (weights) model
└── README.md           # Dokumentasi proyek
