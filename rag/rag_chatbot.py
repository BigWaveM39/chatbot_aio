from transformers import AutoTokenizer
import os
import json
from datetime import datetime
from typing import List, Dict

class RAGChatbot:
    def __init__(self, llm, doc_processor):
        # Assegna il modello LLM e il processore di documenti
        self.llm = llm
        self.doc_processor = doc_processor
        
        # Imposta i limiti per i token
        self.max_context_tokens = 3000
        self.max_response_tokens = 1000
        
        # Inizializza il tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Template per il prompt
        self.prefix = "Utilizza il seguente contesto per rispondere alla domanda. Se l'informazione non è presente nel contesto, dillo chiaramente.\n\nContesto:\n"
        self.suffix = "\n\nDomanda: {query}\nRisposta:"
        
        # Nuove configurazioni per la cronologia
        self.history_dir = "chat_history"
        self.max_tokens_per_file = 8000
        self.chat_history = self.load_chat_history()

    def count_tokens(self, text: str) -> int:
        """Conta i token in un testo utilizzando il tokenizer"""
        return len(self.tokenizer.encode(text))

    def save_chat_history(self, chat_history: List[Dict], file_index: int) -> None:
        """Salva la cronologia della chat in un file JSON"""
        os.makedirs(self.history_dir, exist_ok=True)
        filename = os.path.join(self.history_dir, f"chat_history_{file_index}.json")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)

    def load_chat_history(self) -> List[Dict]:
        """Carica la cronologia completa della chat"""
        chat_history = []
        file_index = 0
        
        if not os.path.exists(self.history_dir):
            return chat_history
        
        while True:
            filename = os.path.join(self.history_dir, f"chat_history_{file_index}.json")
            if not os.path.exists(filename):
                break
                
            with open(filename, 'r', encoding='utf-8') as f:
                chat_history.extend(json.load(f))
            file_index += 1
        
        return chat_history

    def manage_chat_history(self, new_message: Dict) -> None:
        """Gestisce l'aggiunta di nuovi messaggi alla cronologia"""
        current_file_tokens = 0
        message_buffer = []
        file_index = 0
        
        # Aggiunge il nuovo messaggio alla cronologia in memoria
        self.chat_history.append(new_message)
        
        # Gestisce il salvataggio su file
        for message in self.chat_history:
            current_message_tokens = self.count_tokens(message['content'])
            
            if current_file_tokens + current_message_tokens > self.max_tokens_per_file:
                self.save_chat_history(message_buffer, file_index)
                message_buffer = []
                current_file_tokens = 0
                file_index += 1
            
            current_file_tokens += current_message_tokens
            message_buffer.append(message)
        
        # Salva l'ultimo buffer
        if message_buffer:
            self.save_chat_history(message_buffer, file_index)

    def clear_chat_history(self) -> None:
        """Elimina tutti i file della cronologia"""
        if not os.path.exists(self.history_dir):
            return
        
        for filename in os.listdir(self.history_dir):
            if filename.startswith("chat_history_") and filename.endswith(".json"):
                file_path = os.path.join(self.history_dir, filename)
                os.remove(file_path)
        
        self.chat_history = []

    def generate_response(self, query, use_rag=False, k=5):
        try:
            if not use_rag:
                # Genera risposta senza RAG
                messages = [
                    {"role": "system", "content": "Sei un assistente AI amichevole e disponibile."},
                    {"role": "user", "content": query}
                ]
                
                response = ""
                for chunk in self.llm.stream(messages, max_tokens=self.max_response_tokens):
                    if hasattr(chunk, 'content'):
                        response += chunk.content
                        print(chunk.content, end="", flush=True)
                        
                return response

            # Esegue la ricerca di similarità usando il metodo sicuro
            try:
                relevant_docs = self.doc_processor.similarity_search(query, k=k)
            except Exception as e:
                print(f"Errore durante la ricerca dei documenti: {str(e)}")
                return "Mi dispiace, si è verificato un errore durante la ricerca dei documenti pertinenti."
            
            if not relevant_docs:
                return "Mi dispiace, non ho trovato documenti rilevanti per rispondere alla tua domanda."
            
            # Concatena il contenuto dei documenti
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # Calcola i token per ogni componente
            prefix_tokens = len(self.tokenizer.encode(self.prefix))
            suffix_tokens = len(self.tokenizer.encode(self.suffix.format(query=query)))
            query_tokens = len(self.tokenizer.encode(query))
            
            # Calcola i token disponibili per il contesto
            available_tokens = self.max_context_tokens - prefix_tokens - suffix_tokens - query_tokens
            
            # Tronca il contesto se necessario
            context_tokens = self.tokenizer.encode(context)
            if len(context_tokens) > available_tokens:
                context = self.tokenizer.decode(context_tokens[:available_tokens])
            
            # Costruisce il prompt finale
            final_prompt = f"{self.prefix}{context}{self.suffix.format(query=query)}"
            
            # Prepara i messaggi per il modello
            messages = [
                {"role": "system", "content": "Sei un assistente AI che risponde alle domande basandosi sul contesto fornito."},
                {"role": "user", "content": final_prompt}
            ]
            
            # Genera la risposta in streaming
            response = ""
            for chunk in self.llm.stream(messages, max_tokens=self.max_response_tokens):
                if hasattr(chunk, 'content'):
                    response += chunk.content
                    print(chunk.content, end="", flush=True)
            
            # Aggiunge la query e la risposta alla cronologia
            self.manage_chat_history({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            })
            
            if response:
                self.manage_chat_history({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().isoformat()
                })
            
            return response
            
        except Exception as e:
            print(f"Errore durante la generazione della risposta: {str(e)}")
            return "Mi dispiace, si è verificato un errore durante la generazione della risposta."
