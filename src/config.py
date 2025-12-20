import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    'data_path': os.path.join(BASE_DIR, 'data', 'animales'),
    'models_dir': os.path.join(BASE_DIR, 'models'),
    'model_path': os.path.join(BASE_DIR, 'models', 'modelo_animales.pth'),
    'onnx_path': os.path.join(BASE_DIR, 'models', 'modelo_animales.onnx'),
    'img_size': 64,
    'batch_size': 16,
    'learning_rate': 0.001,
    'epochs': 50,
    'classes': [
        'no_peligrosos_acuaticos',
        'no_peligrosos_aereos',
        'no_peligrosos_terrestres',
        'peligrosos_acuaticos',
        'peligrosos_aereos',
        'peligrosos_terrestres'
    ]
}