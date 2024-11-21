from faster_whisper import WhisperModel
import os

class AudioTranscriber:
    def __init__(self, model_type="base", device="cuda", compute_type="float16"):
        self.model = None
        self.model_type = model_type
        self.device = device
        self.compute_type = compute_type

    def load_model(self):
        if self.model is None:
            self.model = WhisperModel(
                self.model_type,
                device=self.device,
                compute_type=self.compute_type
            )

    def transcribe(self, audio_file):
        self.load_model()
        segments, info = self.model.transcribe(
            audio_file,
            vad_filter=True,
            beam_size=5,
            language="it"
        )

        text = ""
        for segment in segments:
            text += segment.text

        try:
            os.remove(audio_file)
        except:
            print("Error deleting file")

        return text 