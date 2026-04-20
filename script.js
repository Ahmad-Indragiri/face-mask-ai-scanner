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
        video.onloadedmetadata = () => {
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
    status.innerText = "Analyzing...";
    
    // Preprocessing: Resize ke 224x224 dan normalisasi piksel ke [-1, 1]
    const tensor = tf.tidy(() => {
        return tf.browser.fromPixels(imgElement)
            .resizeBilinear([224, 224])
            .expandDims(0)
            .toFloat()
            .div(127.5).sub(1); 
    });
    
    try {
        // PERUBAHAN KRUSIAL: Graph Model wajib menggunakan executeAsync
        const predictionTensor = await model.executeAsync(tensor);
        const pred = await predictionTensor.data();
        
        // Cleanup memori hasil prediksi agar RAM tidak bocor
        predictionTensor.dispose();

        // Evaluasi hasil (Asumsi index 0 = Dengan Masker, 1 = Tanpa Masker)
        if (pred[0] > pred[1]) {
            status.innerText = `✅ MASKER TERDETEKSI (${(pred[0] * 100).toFixed(1)}%)`;
            status.style.color = "green";
        } else {
            status.innerText = `❌ TANPA MASKER (${(pred[1] * 100).toFixed(1)}%)`;
            status.style.color = "red";
        }
    } catch (error) {
        console.error("Prediction error:", error);
        status.innerText = "Error analyzing image.";
    } finally {
        // Selalu bersihkan tensor input
        tensor.dispose();
    }
}

// 5. Prediction Loop for Video
async function predictVideo() {
    if (video.style.display === 'none') return; // Stop loop if mode changed

    await predictImage(video);
    requestAnimationFrame(predictVideo);
}

// Eksekusi inisialisasi saat web pertama kali dibuka
initModel();