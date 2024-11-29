from audio.recorder import AudioRecorder
from audio.transcriber import AudioTranscriber
from audio.player import AudioPlayer
from chat.history_manager import ChatHistoryManager
from chat.llm_manager import LLMManager
from config.config_Meta_Llama_3_1_8B_Instruct_Q4_K_M import config
from rag.document_processor import DocumentProcessor

class Chatbot:
    def __init__(self, use_audio=False, stream=False, preload_audio=False):
        self.use_audio = use_audio
        self.stream = stream
        self.history_manager = ChatHistoryManager()
        # Crea una history di default
        try:
            self.current_history = self.history_manager.load_history("default")
        except ValueError:
            # Check if the history already exists before creating it
            if not self.history_manager.history_exists("default"):
                self.current_history = self.history_manager.create_history("default")
            else:
                self.current_history = self.history_manager.load_history("default")
        
        self.llm_manager = LLMManager()
        
        if use_audio or preload_audio:   
            self.audio_recorder = AudioRecorder()
            self.audio_transcriber = AudioTranscriber()
            self.audio_player = AudioPlayer()
        
        self.document_rag = DocumentProcessor()

    def get_user_input(self):
        if self.use_audio:
            audio_file = self.audio_recorder.record()
            return self.audio_transcriber.transcribe(audio_file)
        return input("\nTu: ")

    def generate_response(self, user_input, stream=False, reproduce_audio=False):
        self.current_history.append("user", user_input)
        
        # Esegui la ricerca nel database Chroma
        try:
             # Recupera i documenti rilevanti
            relevant_docs = self.document_rag.similarity_search(user_input)
            
            if not relevant_docs:
                print("No relevant documents found")
            else:
                # Concatenare i contenuti dei documenti
                context = "\nRisultati della ricerca:\n".join([doc.page_content for doc in relevant_docs])
                self.current_history.append("system", context)
        except Exception as e:
            print(f"Errore nella ricerca nel database: {str(e)}\n")
            
        # Converti context in una lista per la concatenazione
        full_context = self.current_history.get_tokenized_context(config["inference_params"]["pre_prompt"])

        for token, llm_response in self.llm_manager.generate_response(full_context, stream=stream):
            yield token, llm_response

        if reproduce_audio and self.use_audio:
            self.audio_player.play(llm_response)

        self.current_history.append("assistant", llm_response)
        return llm_response
        
    # Nuovi metodi per gestire le chat history
    def create_new_chat(self, name: str):
        """Crea una nuova chat history e la imposta come corrente"""
        self.current_history = self.history_manager.create_history(name)
        
    def load_chat(self, name: str):
        """Carica una chat history esistente"""
        self.current_history = self.history_manager.load_history(name)
        
    def delete_chat(self, name: str):
        """Elimina una chat history"""
        return self.history_manager.delete_history(name)
        
    def list_chats(self):
        """Restituisce la lista delle chat disponibili"""
        return self.history_manager.list_histories()
        
    def get_current_chat(self):
        """Restituisce la chat history corrente"""
        return self.current_history

    def toggle_audio(self):
        self.use_audio = not self.use_audio
        
    def toggle_stream(self):
        self.stream = not self.stream

    def is_audio_enabled(self):
        return self.use_audio

    def is_stream_enabled(self):
        return self.stream

    def process_documents(self, file_paths):
        """Processa i documenti forniti e li aggiunge al database"""
        self.document_rag.process_documents(file_paths)

    def run(self):
        print("Benvenuto nel chatbot. Comandi disponibili:")
        print("- 'exit' per uscire")
        print("- '/new nome' per creare una nuova chat")
        print("- '/load nome' per caricare una chat")
        print("- '/list' per vedere le chat disponibili")
        print("- '/delete nome' per eliminare una chat")
        print("- '/process <file_paths>' per processare documenti")
        
        while True:
            user_input = self.get_user_input()
            
            # Gestione dei comandi speciali
            if user_input.startswith('/'):
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                
                if command == '/new' and len(parts) > 1:
                    try:
                        self.create_new_chat(parts[1])
                        print(f"Creata nuova chat: {parts[1]}")
                    except ValueError as e:
                        print(f"Errore: {e}")
                    continue
                    
                elif command == '/load' and len(parts) > 1:
                    try:
                        self.load_chat(parts[1])
                        print(f"Caricata chat: {parts[1]}")
                    except ValueError as e:
                        print(f"Errore: {e}")
                    continue
                    
                elif command == '/list':
                    chats = self.list_chats()
                    print("\nChat disponibili:")
                    for chat in chats:
                        print(f"- {chat['name']} (creata: {chat['created_at']})")
                    continue
                    
                elif command == '/delete' and len(parts) > 1:
                    if self.delete_chat(parts[1]):
                        print(f"Chat eliminata: {parts[1]}")
                    else:
                        print(f"Chat non trovata: {parts[1]}")
                    continue
                    
                elif command == '/process' and len(parts) > 1:
                    file_paths = parts[1].split(',')
                    self.process_documents(file_paths)
                    continue

            if "exit" in user_input.lower():
                break
            
            # Gestione normale del messaggio
            for token, full_response in self.generate_response(user_input, stream=self.stream):
                if self.stream:
                    print(token, end="", flush=True)
                
            if not self.stream:
                print(f"\nAssistant: {full_response}")

if __name__ == "__main__":
    # Esempio di utilizzo:
    # chatbot = Chatbot(use_audio=True)  # Con funzionalità audio
    chatbot = Chatbot(use_audio=False, stream=False)   # Senza funzionalità audio
    chatbot.run() 
    