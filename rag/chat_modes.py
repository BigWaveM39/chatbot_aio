import os
from typing import List
from datetime import datetime
from .rag_chatbot import RAGChatbot
from .document_processor import DocumentProcessor
from langchain_chroma import Chroma

class ChatModes:

    def create_chroma_db(self) -> bool:
        """Crea il database Chroma dai documenti nella directory 'data'"""
        print("\n=== Creazione Database Chroma ===")
        
        # Verifica directory data
        data_dir = "data"
        if not os.path.exists(data_dir):
            print(f"La directory '{data_dir}' non esiste.")
            return False
        
        # Raccoglie i file
        file_paths = []
        for file in os.listdir(data_dir):
            if file.endswith(('.txt', '.md', '.pdf')):  # aggiungi altre estensioni se necessario
                file_paths.append(os.path.join(data_dir, file))
        
        if not file_paths:
            print("Nessun file da processare trovato nella directory 'data'.")
            return False
        
        # Inizializza e usa DocumentProcessor
        try:
            doc_processor = DocumentProcessor()
            success = doc_processor.process_documents(file_paths)
            
            if success:
                print("Database creato con successo!")
                return True
            else:
                print("Errore durante la creazione del database.")
                return False
                
        except Exception as e:
            print(f"Errore durante la creazione del database: {str(e)}")
            return False

    def rag_chat(self, llm) -> None:
        """Gestisce la modalità di chat RAG"""
        print("\n=== Chat RAG ===")
        print("Digita 'exit' per uscire dalla chat")
        
        # Verifica e inizializza il database
        doc_processor = DocumentProcessor()
        if not os.path.exists(doc_processor.base_dir):
            print("Nessun database trovato. Esegui prima la creazione del database.")
            return
        
        # Seleziona l'ultimo database creato
        db_dirs = [d for d in os.listdir(doc_processor.base_dir) if d.startswith('db_')]
        if not db_dirs:
            print("Nessun database trovato.")
            return
        
        latest_db = sorted(db_dirs)[-1]
        db_path = os.path.join(doc_processor.base_dir, latest_db)
        
        # Carica il database Chroma
        try:
            doc_processor.load_database(doc_processor.get_latest_database_path())
        except Exception as e:
            print(f"Errore nel caricamento del database: {str(e)}")
            return
        
        # Inizializza il RAGChatbot
        chatbot = RAGChatbot(llm, doc_processor)
        
        # Ciclo di interazione
        while True:
            # Input utente
            user_input = input("\nTu: ").strip()
            
            # Gestione uscita
            if user_input.lower() == 'exit':
                break
            
            # Genera e stampa la risposta
            response = chatbot.generate_response(user_input, use_rag=True)
            if response:
                print("\nAssistente:", response) 