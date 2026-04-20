// Global variables
let model;
const status = document.getElementById('status');
const video = document.getElementById('webcam');
const preview = document.getElementById('preview');

// 1. Initialize the AI Model
async function initModel() {
    try {
        status.innerText = "Loading AI Model...";
        // Load the model from our local folder
        model = await tf.loadLayersModel('/model/model.json');
        status.innerText = "Model Ready. Select camera or upload a photo.";
    } catch (error) {
        console.error("Model loading error:", error);
        status.innerText = "Failed to load model.";
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

// 4. Prediction Logic for Static Images
async function predictImage(imgElement) {
    status.innerText = "Analyzing...";
    
    // Process image: resize, normalize, predict
    const tensor = tf.tidy(() => {
        return tf.browser.fromPixels(imgElement)
            .resizeBilinear([224, 224])
            .expandDims(0)
            .toFloat()
            .div(127.5).sub(1);
    });
    
    const pred = await model.predict(tensor).data();
    tensor.dispose(); // Cleanup memory

    // Probabilities: [0] = WithMask, [1] = WithoutMask
    status.innerText = pred[0] > pred[1] 
        ? "✅ MASKER TERDETEKSI" 
        : "❌ TANPA MASKER";
}

// 5. Prediction Loop for Video
async function predictVideo() {
    if (video.style.display === 'none') return; // Stop loop if mode changed

    await predictImage(video);
    requestAnimationFrame(predictVideo);
}

// Initial Call
initModel();