import torch
import torch.onnx
from train import Net
from config import CONFIG
import os

def convert():
    # 1. Cargar el modelo entrenado
    device = torch.device("cpu")
    model = Net().to(device)
    
    # Cargar los pesos guardados (modelo_animales.pth)
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
    model.eval()

    # 2. Crear una "imagen falsa" para configurar ONNX
    # IMPORTANTE: Ahora es (1, 3, 64, 64) porque usamos 3 canales (RGB)
    dummy_input = torch.randn(1, 3, CONFIG['img_size'], CONFIG['img_size'])

    # 3. Exportar a ONNX
    torch.onnx.export(
        model, 
        dummy_input, 
        CONFIG['onnx_path'], 
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"Modelo convertido a ONNX guardado en: {CONFIG['onnx_path']}")

if __name__ == "__main__":
    convert()