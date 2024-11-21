import config._private as private


'''
NOTE:
ensure that the paths are correct and that the files exist
config = {
    "llm_model": "path/to/llm_model.gguf",
    "audio_model": "path/to/audio_model.onnx",
    "audio_model_json": "path/to/audio_model.json",
    "audio_output": "path/to/output.wav",
    "piper_exe": "path/to/piper.exe"
}
'''

llm_model = private.config["llm_model"]
audio_model = private.config["audio_model"]
audio_model_json = private.config["audio_model_json"]
audio_output = private.config["audio_output"]
piper_exe = private.config["piper_exe"]