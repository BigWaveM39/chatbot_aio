import torch

print(f"CUDA disponibile: {torch.cuda.is_available()}\n")
print(f"Numero di GPU disponibili: {torch.cuda.device_count()}\n")
print(f"Nome della GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}\n")
print(f"Versione di CUDA supportata da PyTorch: {torch.version.cuda}\n")
print(f"cuDNN abilitato: {torch.backends.cudnn.enabled}")