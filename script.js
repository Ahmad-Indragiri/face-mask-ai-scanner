// Define HTML Elements
const video = document.getElementById('webcam');
const statusText = document.getElementById('status');

// Global variable to store the loaded model
let model;

// Phase 1: Load the TF.js Model
async function initModel() {
    try {
        // Load model.json from the local directory
        model = await tf.loadLayersModel('./model/model.json');
        statusText.innerText = "Model Loaded! Starting camera...";
        startWebcam();
    } catch (error) {
        console.error("Model load error:", error);
        statusText.innerText = "Error: Check Console.";
    }
}

// Phase 2: Start the Webcam
async function startWebcam() {
    try {
        // Request video stream from user's camera
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        
        // Wait until the video is actively playing before starting predictions
        video.addEventListener('playing', () => {
            statusText.innerText = "Analyzing...";
            predictLoop(); // Start the AI loop
        });
    } catch (error) {
        console.error("Camera error:", error);
        statusText.innerText = "Camera Access Denied.";
    }
}

// Phase 3: Real-time Prediction Loop
async function predictLoop() {
    // tf.tidy cleans up GPU memory automatically after each frame
    const prediction = tf.tidy(() => {
        // 1. Capture current frame from video
        let img = tf.browser.fromPixels(video);
        
        // 2. Resize to 224x224 (MobileNetV2 input size requirement)
        img = tf.image.resizeBilinear(img, [224, 224]);
        
        // 3. Expand dimensions from [224,224,3] to [1,224,224,3] (Batch processing)
        const expandedImg = img.expandDims(0);
        
        // 4. Preprocess: Scale pixel values to [-1, 1] as expected by MobileNetV2
        const preprocessedImg = expandedImg.toFloat().div(tf.scalar(127.5)).sub(tf.scalar(1));
        
        // 5. Predict using the model
        return model.predict(preprocessedImg);
    });

    // Extract probabilities from the tensor
    const probabilities = await prediction.data();
    prediction.dispose(); // Manual memory cleanup

    // Classes: [0] = WithMask, [1] = WithoutMask (Based on alphabetical folder names)
    const withMaskProb = probabilities[0];
    const withoutMaskProb = probabilities[1];

    // Update the UI based on the highest probability
    if (withMaskProb > withoutMaskProb) {
        statusText.innerText = `MASK DETECTED (${(withMaskProb * 100).toFixed(1)}%)`;
        statusText.className = "mask";
    } else {
        statusText.innerText = `NO MASK (${(withoutMaskProb * 100).toFixed(1)}%)`;
        statusText.className = "no-mask";
    }

    // Request the browser to call predictLoop again for the next video frame
    requestAnimationFrame(predictLoop);
}

// Execute the application
initModel();