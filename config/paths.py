import os

# Verifica se il file _private.py esiste
if not os.path.exists('config/_private.py'):
    
    with open('config/_private.py', 'w') as f:
        f.write('''config = {
    "llm_model": "path/to/llm_model.gguf",
    "audio_model": "path/to/audio_model.onnx",
    "audio_model_json": "path/to/audio_model.json",
    "audio_output": "path/to/output.wav",
    "piper_exe": "path/to/piper.exe"
}

print("*"*50)
print("ATTENTION: ensure that the paths are correct and that the files exist in config/_private.py")
print("*"*50)
''')
    
import config._private as private

llm_model = private.config["llm_model"]
audio_model = private.config["audio_model"]
audio_model_json = private.config["audio_model_json"]
audio_output = private.config["audio_output"]
piper_exe = private.config["piper_exe"]