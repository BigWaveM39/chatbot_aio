import tiktoken
import json
import os

class ChatHistory:
    def __init__(self, history_dir: str):
        self.history_dir = history_dir
        self.history = []
        self.max_tokens = 2048
        self.reserved_tokens = 500
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.history = self._load_all_history()
        
    def append(self, role: str, content: str):
        self._manage_chat_history({'role': role, 'content': content})

    def get_history(self):
        return self.history

    def clear(self):
        self.history = []
        self._save_history()

    def get_tokenized_context (self, preprompt: str) -> list:
        # Calcola i token del preprompt
        preprompt_tokens = self._num_tokens_from_string(preprompt)
        
        # Verifica che il preprompt non superi già il limite
        if preprompt_tokens >= self.max_tokens:
            raise ValueError(f"Il preprompt è troppo lungo ({preprompt_tokens} token). Massimo consentito: {self.max_tokens} token")
        
        # Tokens disponibili per la history
        available_tokens = self.max_tokens - self.reserved_tokens - preprompt_tokens
        
        # Crea la lista risultante iniziando con il preprompt
        result = [{'role': 'system', 'content': preprompt}]
        current_tokens = preprompt_tokens
        
        # Scorri la history dall'ultimo messaggio verso il primo
        for message in reversed(self.history):
            message_tokens = self._count_tokens([message])
            
            # Se aggiungendo questo messaggio superiamo i token disponibili, ci fermiamo
            if current_tokens + message_tokens > available_tokens:
                break
                
            # Altrimenti, aggiungiamo il messaggio all'inizio della lista (dopo il preprompt)
            result.insert(1, message)
            current_tokens += message_tokens
            
        return result

    def _count_tokens(self, messages: list[dict]) -> int:
        num_tokens = 0
        for message in messages:
            # Count tokens in content
            num_tokens += len(self.encoding.encode(message['content']))
            # Count tokens in role
            num_tokens += len(self.encoding.encode(message['role']))
        return num_tokens

    def _num_tokens_from_string(self, string: str) -> int:
        return len(self.encoding.encode(string))
    
    def _load_all_history(self) -> list:
        all_messages = []
        file_index = 0
        
        while True:
            filename = os.path.join(self.history_dir, f'chat_{file_index}.json')
            if not os.path.exists(filename):
                break
                
            with open(filename, 'r', encoding='utf-8') as f:
                messages = json.load(f)
                all_messages.extend(messages)
                
            file_index += 1
            
        return all_messages
    
    def _manage_chat_history(self, new_message: dict) -> bool:
        # Validazione del formato del messaggio
        if not isinstance(new_message, dict):
            raise ValueError("Il messaggio deve essere un dizionario")
            
        # Validazione dei campi richiesti
        required_fields = ['role', 'content']
        if not all(field in new_message for field in required_fields):
            raise ValueError("Il messaggio deve contenere i campi 'role' e 'content'")
            
        # Validazione del ruolo
        valid_roles = ['user', 'assistant', 'system']
        if new_message['role'] not in valid_roles:
            raise ValueError(f"Ruolo non valido. Deve essere uno tra: {', '.join(valid_roles)}")
            
        # Validazione del contenuto
        if not isinstance(new_message['content'], str) or not new_message['content'].strip():
            raise ValueError("Il contenuto del messaggio deve essere una stringa non vuota")
            
        # Validazione della lunghezza (opzionale, puoi modificare il limite)
        message_tokens = self._count_tokens([new_message])
        if message_tokens > 1500:
            raise ValueError(f"Il messaggio è troppo lungo ({message_tokens} token). Massimo consentito: 1500 token")
    
        # Aggiungi il nuovo messaggio alla history
        self.history.append(new_message)
        self._save_history()
      
    def _save_history_chunk(self, messages: list, file_index: int):
        # Crea la directory se non esiste
        os.makedirs(self.history_dir, exist_ok=True)
        
        # Definisci il nome del file usando l'indice
        filename = os.path.join(self.history_dir, f'chat_{file_index}.json')
        
        # Salva la cronologia in formato JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
            
    def _save_history(self):
        try:
            # Inizializza variabili
            current_chunk = []
            current_tokens = 0
            file_index = 0
            
            # Analizza tutti i messaggi nella history
            for message in self.history:
                message_tokens = self._count_tokens([message])
                
                if current_tokens + message_tokens > 1500:
                    self._save_history_chunk(current_chunk, file_index)
                    current_chunk = [message]
                    current_tokens = message_tokens
                    file_index += 1
                else:
                    current_chunk.append(message)
                    current_tokens += message_tokens
            
            # Salva l'ultimo chunk se contiene messaggi
            if current_chunk:
                self._save_history_chunk(current_chunk, file_index)
                
            return True
            
        except Exception as e:
            print(f"Errore durante il salvataggio della chat history: {str(e)}")
            return False