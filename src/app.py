from flask import Flask, request, render_template_string, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import base64
from config import CONFIG

app = Flask(__name__)

ort_session = ort.InferenceSession(CONFIG['onnx_path'])
clases = ['Papel', 'Piedra', 'Tijera']

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>MLOps - PIEDRA - PAPEL - TIJERA</title>
    <style>
        body { font-family: sans-serif; padding: 20px; text-align: center; }
        #video { border: 1px solid #ccc; width: 320px; height: 320px; }
        .container { margin-top: 20px; }
        #result { font-weight: bold; font-size: 24px; margin-top: 10px; }
    </style>
</head>
<body>
    <h2>MLOps - PIEDRA - PAPEL - TIJERA</h2>
    <div>
        <video id="video" autoplay playsinline></video>
        <canvas id="canvas" width="64" height="64" style="display:none;"></canvas>
    </div>
    <div class="container">
        <button onclick="startCamera()">Activar Cámara</button>
        <div id="result">Esperando predicción...</div>
    </div>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const resultDiv = document.getElementById('result');

        async function startCamera() {
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                    video.srcObject = stream;
                    setInterval(sendFrame, 500); 
                } catch (error) {
                    alert("Error accediendo a la cámara: " + error.message);
                }
            } else {
                alert("Tu navegador no soporta acceso a medios o no estás en localhost.");
            }
        }

        function sendFrame() {
            ctx.drawImage(video, 0, 0, 64, 64);
            const dataURL = canvas.toDataURL('image/jpeg');
            
            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataURL })
            })
            .then(response => response.json())
            .then(data => {
                resultDiv.innerText = data.label + " (" + data.prob + ")";
            })
            .catch(error => console.error('Error:', error));
        }
    </script>
</body>
</html>
"""

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize((CONFIG['img_size'], CONFIG['img_size']))
    img_arr = np.array(img).astype(np.float32) / 255.0
    img_arr = img_arr.reshape(1, 1, CONFIG['img_size'], CONFIG['img_size'])
    return img_arr

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json['image']
        header, encoded = data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        
        input_tensor = preprocess(image_bytes)
        outputs = ort_session.run(None, {"input": input_tensor})
        probs = softmax(outputs[0][0])
        idx = np.argmax(probs)
        
        return jsonify({
            'label': clases[idx],
            'prob': f"{probs[idx]*100:.1f}%"
        })
    except:
        return jsonify({'label': 'Error', 'prob': '0%'})

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)