import torch
import torch.onnx
from train import Net
from config import CONFIG
import os

def convert():
    device = torch.device("cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
    model.eval()
    dummy_input = torch.randn(1, 3, CONFIG['img_size'], CONFIG['img_size'])
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

if __name__ == "__main__":
    convert()