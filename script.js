// ======================================================
// Face Mask Detection V2 + Face Crop Box (BlazeFace)
// ======================================================

let model;
let faceModel;
let isPredicting = false;

const status = document.getElementById("status");
const video = document.getElementById("webcam");
const preview = document.getElementById("preview");

// Canvas overlay untuk gambar box
const canvas = document.createElement("canvas");
canvas.style.position = "absolute";
canvas.style.pointerEvents = "none";
document.body.appendChild(canvas);

const LABELS = [
    "WithoutMask",
    "WithMask"
];

// ----------------------------
// Load Models
// ----------------------------
async function initModel() {
    try {
        status.innerText = "Loading AI Models...";

        const modelUrl = "./model/model.json?v=" + Date.now();
        model = await tf.loadGraphModel(modelUrl);

        faceModel = await blazeface.load();

        status.innerText = "Model Ready.";
    } catch (err) {
        console.error(err);
        status.innerText = "Failed to load model.";
    }
}

// ----------------------------
// Sync canvas overlay ke posisi & ukuran elemen sumber
// ----------------------------
function syncCanvas(sourceEl) {
    const rect = sourceEl.getBoundingClientRect();
    canvas.width = sourceEl.videoWidth || sourceEl.naturalWidth || rect.width;
    canvas.height = sourceEl.videoHeight || sourceEl.naturalHeight || rect.height;
    canvas.style.width = rect.width + "px";
    canvas.style.height = rect.height + "px";
    canvas.style.left = rect.left + window.scrollX + "px";
    canvas.style.top = rect.top + window.scrollY + "px";
}

// ----------------------------
// Webcam
// ----------------------------
async function useWebcam() {
    preview.style.display = "none";
    video.style.display = "block";

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user" }
        });

        video.srcObject = stream;

        video.onloadedmetadata = async () => {
            await video.play();
            predictVideo();
        };
    } catch (err) {
        console.error(err);
        status.innerText = "Cannot access webcam.";
    }
}

// ----------------------------
// Upload Image
// ----------------------------
function handleUpload(e) {
    video.style.display = "none";
    preview.style.display = "block";

    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();

    reader.onload = event => {
        preview.src = event.target.result;

        preview.onload = () => {
            predictImage(preview);
        };
    };

    reader.readAsDataURL(file);
}

// ----------------------------
// Deteksi wajah, gambar box, crop, lalu prediksi masker
// ----------------------------
async function predictImage(imgElement) {
    if (isPredicting) return;
    isPredicting = true;

    const start = performance.now();

    try {
        syncCanvas(imgElement);
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const faces = await faceModel.estimateFaces(imgElement, false);

        if (faces.length === 0) {
            status.innerText = "No face detected.";
            isPredicting = false;
            return;
        }

        // Ambil wajah pertama (paling besar/prioritas)
        const face = faces[0];
        const [x1, y1] = face.topLeft;
        const [x2, y2] = face.bottomRight;

        // Sedikit padding di sekitar wajah supaya masker ikut ter-crop
        const padding = 0.3;
        const boxW = x2 - x1;
        const boxH = y2 - y1;

        let cropX = Math.max(0, x1 - boxW * padding);
        let cropY = Math.max(0, y1 - boxH * padding);
        let cropW = boxW * (1 + 2 * padding);
        let cropH = boxH * (1 + 2 * padding);

        // Pastikan tidak keluar batas gambar
        const imgW = imgElement.videoWidth || imgElement.naturalWidth || imgElement.width;
        const imgH = imgElement.videoHeight || imgElement.naturalHeight || imgElement.height;
        cropW = Math.min(cropW, imgW - cropX);
        cropH = Math.min(cropH, imgH - cropY);

        // Gambar box di canvas overlay
        ctx.strokeStyle = "lime";
        ctx.lineWidth = 3;
        ctx.strokeRect(cropX, cropY, cropW, cropH);

        // Crop & prediksi
        const inputTensor = tf.tidy(() => {
            const img = tf.browser.fromPixels(imgElement);

            const cropped = img.slice(
                [Math.round(cropY), Math.round(cropX), 0],
                [Math.round(cropH), Math.round(cropW), 3]
            );

            return cropped
                .resizeBilinear([224, 224])
                .toFloat()
                .div(255.0)
                .expandDims();
        });

        const prediction = model.predict(inputTensor);
        const outputTensor = Array.isArray(prediction) ? prediction[0] : prediction;
        const scores = Array.from(await outputTensor.data());

        console.log("Raw Scores:", scores);

        const maxIndex = scores.indexOf(Math.max(...scores));
        const label = LABELS[maxIndex];
        const confidence = scores[maxIndex] * 100;

        inputTensor.dispose();
        outputTensor.dispose();

        const end = performance.now();
        const inferenceTime = (end - start).toFixed(1);

        if (label === "WithMask") {
            status.innerHTML =
                `MASK DETECTED<br>
                Confidence : ${confidence.toFixed(2)} %<br>
                Inference : ${inferenceTime} ms`;
            status.style.color = "green";
            ctx.strokeStyle = "lime";
        } else {
            status.innerHTML =
                `NO MASK<br>
                Confidence : ${confidence.toFixed(2)} %<br>
                Inference : ${inferenceTime} ms`;
            status.style.color = "red";
            ctx.strokeStyle = "red";
        }

        // Gambar ulang box dengan warna sesuai hasil
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.lineWidth = 3;
        ctx.strokeRect(cropX, cropY, cropW, cropH);

    } catch (err) {
        console.error(err);
        status.innerText = err.message;
        status.style.color = "red";
    } finally {
        isPredicting = false;
    }
}

// ----------------------------
// Video Loop
// ----------------------------
async function predictVideo() {
    if (video.style.display === "none") {
        return;
    }

    if (video.readyState === 4) {
        await predictImage(video);
    }

    setTimeout(() => {
        requestAnimationFrame(predictVideo);
    }, 100);
}

// ----------------------------
// Initialize
// ----------------------------
initModel();