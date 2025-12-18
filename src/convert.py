import torch
import torch.onnx
from config import CONFIG
from train import Net
import onnxscript

if __name__ == "__main__":
    model = Net()
    model.load_state_dict(torch.load(CONFIG['model_path']))
    model.eval()

    dummy_input = torch.randn(1, 1, CONFIG['img_size'], CONFIG['img_size'])

    torch.onnx.export(model, dummy_input, CONFIG['onnx_path'],
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})