from flask import Flask, request, render_template_string, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
from config import CONFIG

app = Flask(__name__)

ort_session = ort.InferenceSession(CONFIG['onnx_path'])
clases = CONFIG['classes']

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Clasificador</title>
    <style>
        body { font-family: sans-serif; padding: 20px; text-align: center; }
        #preview { max-width: 300px; display: none; margin: 20px auto; }
        #result { font-weight: bold; margin-top: 20px; white-space: pre-line; }
    </style>
</head>
<body>
    <h2>Sistema de Clasificacion</h2>
    <input type="file" id="fileInput" accept="image/*" onchange="preview()">
    <button onclick="upload()">Analizar</button>
    <img id="preview">
    <div id="result"></div>

    <script>
        function preview() {
            const file = document.getElementById('fileInput').files[0];
            const reader = new FileReader();
            reader.onload = e => {
                document.getElementById('preview').src = e.target.result;
                document.getElementById('preview').style.display = 'block';
            };
            reader.readAsDataURL(file);
        }

        async function upload() {
            const file = document.getElementById('fileInput').files[0];
            const formData = new FormData();
            formData.append('file', file);

            const res = await fetch('/predict', { method: 'POST', body: formData });
            const data = await res.json();
            
            document.getElementById('result').innerText = 
                `Clase: ${data.clase}\nHábitat: ${data.habitat}\nPeligroso: ${data.peligro}\nScore (Accuracy): ${data.prob}`;
        }
    </script>
</body>
</html>
"""

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((CONFIG['img_size'], CONFIG['img_size']))
    img_arr = np.array(img).astype(np.float32) / 255.0
    img_arr = img_arr.transpose(2, 0, 1)
    img_arr = img_arr.reshape(1, 3, CONFIG['img_size'], CONFIG['img_size'])
    return img_arr

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    img_bytes = file.read()
    input_tensor = preprocess(img_bytes)
    outputs = ort_session.run(None, {"input": input_tensor})
    probs = softmax(outputs[0][0])
    idx = np.argmax(probs)
    
    clase_raw = clases[idx]
    parts = clase_raw.split('_')
    
    peligro = "NO" if "no" in parts else "SI"
    habitat = parts[-1].upper()

    return jsonify({
        'clase': clase_raw,
        'peligro': peligro,
        'habitat': habitat,
        'prob': f"{probs[idx]:.4f}"
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)