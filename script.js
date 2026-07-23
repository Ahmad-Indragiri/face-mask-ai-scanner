// Global variables
let model;
const status = document.getElementById('status');
const video = document.getElementById('webcam');
const preview = document.getElementById('preview');

// 1. Initialize the AI Graph Model
async function initModel() {
    try {
        status.innerText = "Loading AI Graph Model...";
        
        // MENGGUNAKAN loadGraphModel
        model = await tf.loadGraphModel('./model/model.json'); 
        
        status.innerText = "Model Ready. Select camera or upload a photo.";
    } catch (error) {
        console.error("Model loading error:", error);
        status.innerText = "Failed to load model. Check console.";
    }
}

// 2. Webcam Mode
async function useWebcam() {
    preview.style.display = 'none';
    video.style.display = 'block';
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        
        // Start prediction loop after video is active
        video.onloadedmetadata = async () => {
            // Memutar video agar frame tidak membeku
            await video.play();
            predictVideo();
        };
    } catch (error) {
        console.error("Camera error:", error);
        status.innerText = "Camera access denied.";
    }
}

// 3. Upload Mode
function handleUpload(e) {
    video.style.display = 'none';
    preview.style.display = 'block';
    
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
        preview.src = event.target.result;
        preview.onload = () => predictImage(preview);
    };
    reader.readAsDataURL(file);
}

// 4. Prediction Logic (Graph Model Specific)
async function predictImage(imgElement) {
    // Hanya tampilkan teks "Analyzing" jika input berasal dari upload foto
    if (imgElement.id === 'preview') {
        status.innerText = "Analyzing...";
        status.style.color = "#94a3b8"; // Sesuaikan dengan warna teks default web Anda
    }
    
    const tensor = tf.tidy(() => {
        return tf.browser.fromPixels(imgElement)
            .resizeBilinear([224, 224])
            .expandDims(0)
            .toFloat()
            .div(127.5).sub(1); 
    });
    
    try {
        const predictionTensor = model.predict(tensor); 
        
        const tensorData = Array.isArray(predictionTensor) ? predictionTensor[0] : predictionTensor;
        const pred = await tensorData.data();
        
        // Cleanup memori tensor prediksi
        if (Array.isArray(predictionTensor)) {
            predictionTensor.forEach(t => t.dispose());
        } else {
            predictionTensor.dispose();
        }

        // Evaluasi Final yang Benar (pred[1] = Masker, pred[0] = Tanpa Masker)
        if (pred[1] > pred[0]) {
            status.innerText = `✅ MASKER TERDETEKSI (${(pred[1] * 100).toFixed(1)}%)`;
            status.style.color = "green";
        } else {
            status.innerText = `❌ TANPA MASKER (${(pred[0] * 100).toFixed(1)}%)`;
            status.style.color = "red";
        }

    } catch (error) {
        console.error("Prediction error:", error);
        status.innerText = "Error: " + error.message; 
        status.style.color = "red";
        video.style.display = 'none'; 
    } finally {
        // Membersihkan memori tensor input
        tensor.dispose();
    }
}

// 5. Prediction Loop for Video
async function predictVideo() {
    if (video.style.display === 'none') return; // Stop loop if mode changed

    // Pastikan video benar-benar memiliki data frame sebelum diprediksi
    if (video.readyState === 4) {
        await predictImage(video);
    }
    
    // Looping kamera secara real-time
    requestAnimationFrame(predictVideo);
}

// Eksekusi inisialisasi saat web pertama kali dibuka
initModel();