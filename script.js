// ======================================================
// Face Mask Detection V2
// TensorFlow.js Graph Model
// ======================================================

// ----------------------------
// Global Variables
// ----------------------------
let model;
let isPredicting = false;

const status = document.getElementById("status");
const video = document.getElementById("webcam");
const preview = document.getElementById("preview");

// Pastikan urutan sama dengan train_dataset.class_names
const LABELS = [
    "WithMask",
    "WithoutMask"
];

// ----------------------------
// Load Model
// ----------------------------
async function initModel() {

    try {

        status.innerText = "Loading AI Model...";

        // Tambahkan cache-busting
        const modelUrl = "./model/model.json?v=" + Date.now();

        model = await tf.loadGraphModel(modelUrl);

        console.log(model);

        status.innerText = "Model Ready.";

    } catch (err) {

        console.error(err);

    }

}

// ----------------------------
// Webcam
// ----------------------------
async function useWebcam() {

    preview.style.display = "none";

    video.style.display = "block";

    try {

        const stream = await navigator.mediaDevices.getUserMedia({

            video: {
                facingMode: "user"
            }

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
            console.log(model.inputs);
            console.log(model.outputs);

        };

    };

    reader.readAsDataURL(file);

}

// ----------------------------
// Prediction
// ----------------------------
async function predictImage(imgElement) {

    if (isPredicting) return;

    isPredicting = true;

    const start = performance.now();

    try {

        const inputTensor = tf.tidy(() => {

            return tf.browser.fromPixels(imgElement)

                .resizeBilinear([224, 224])

                .toFloat()

                .div(127.5)

                .sub(1)

                .expandDims();

        });

        const prediction = model.predict(inputTensor);

        const outputTensor = Array.isArray(prediction)
            ? prediction[0]
            : prediction;

        const scores = Array.from(await outputTensor.data());

        console.log("Raw Scores:", scores);

        const maxIndex = scores.indexOf(Math.max(...scores));

        console.log("Predicted Index:", maxIndex);
        console.log("Predicted Label:", LABELS[maxIndex]);

        inputTensor.dispose();

        outputTensor.dispose();

        const label = LABELS[maxIndex];

        const confidence = scores[maxIndex] * 100;

        const end = performance.now();

        const inferenceTime = (end - start).toFixed(1);

        if (label === "WithMask") {

            status.innerHTML =

                `MASK DETECTED<br>
            Confidence : ${confidence.toFixed(2)} %<br>
            Inference : ${inferenceTime} ms`;

            status.style.color = "green";

        }

        else {

            status.innerHTML =

                `NO MASK<br>
            Confidence : ${confidence.toFixed(2)} %<br>
            Inference : ${inferenceTime} ms`;

            status.style.color = "red";

        }

    }

    catch (err) {

        console.error(err);

        status.innerText = err.message;

        status.style.color = "red";

    }

    finally {

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

    // sekitar 10 FPS
    setTimeout(() => {

        requestAnimationFrame(predictVideo);

    }, 100);

}

// ----------------------------
// Initialize
// ----------------------------
initModel();