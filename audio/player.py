import pyttsx3

class AudioPlayer:
    def __init__(self):
        self.engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        self.engine = pyttsx3.init()
        # Configurazione base del motore TTS
        self.engine.setProperty('rate', 150)    # Velocità di parlato
        self.engine.setProperty('volume', 0.9)  # Volume (0-1)


    def play(self, text: str):
        """
        Riproduce il testo fornito come audio
        Args:
            text (str): Il testo da convertire in audio
        """
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Errore durante la riproduzione audio: {e}")
            
    def stop(self):
        """
        Ferma la riproduzione audio in corso
        """
        try:
            self.engine.stop()
        except Exception as e:
            print(f"Errore durante l'arresto della riproduzione: {e}")

    def __del__(self):
        """
        Cleanup del motore TTS quando l'oggetto viene distrutto
        """
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass 