# Real-time AI Mask Detection (TensorFlow.js)

A web-based application for detecting face mask usage in **real time** using a webcam or uploaded images. Built with **TensorFlow.js** and powered by a **MobileNetV2** model trained via transfer learning, this project demonstrates efficient client-side AI inference directly in the browser.

## Live Demo

- **Demo URL:** https://ai-facemask-deteksi.netlify.app/  
- **Repository:** https://github.com/your-username/mask-detector-ai  

## Features

- **Hybrid Detection Mode**  
  Detect mask usage via live webcam or uploaded images (JPG/PNG).
- **Fast In-Browser Inference**  
  Uses TensorFlow.js Graph Model format for optimized performance.
- **Responsive Modern UI**  
  Clean, minimal dark interface optimized for desktop and mobile.
- **Privacy-First Design**  
  All processing happens locally in the browser. No data is sent to servers.

## 🧰 Tech Stack
| Category        | Technology |
|----------------|------------|
| Frontend       | HTML5, CSS3, Vanilla JavaScript |
| AI Framework   | TensorFlow.js |
| Model          | MobileNetV2 (Transfer Learning) |
| Training       | Kaggle Notebook |

## 🏗️ Project Structure
.
├── index.html          # Antarmuka pengguna (UI)
├── script.js           # Logika utama AI & penanganan webcam
├── model/              # Folder penyimpanan model AI
│   ├── model.json      # Arsitektur Graph Model
│   └── group1-shard1of1.bin # Bobot (weights) model
└── README.md           # Dokumentasi proyek

## ⚙️ How It Works
1. Loads a pre-trained MobileNetV2 model using TensorFlow.js  
2. Captures input from webcam or image upload  
3. Preprocesses the image into tensor format  
4. Runs inference directly in the browser  
5. Displays prediction results in real time  

## 🧪 Local Development
### 1. Clone Repository
```bash
git clone https://github.com/your-username/mask-detector-ai.git
cd mask-detector-ai
# Option 1: Python
python -m http.server

# Option 2: Node.js
npx live-server
