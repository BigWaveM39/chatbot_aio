import os
import gc
from datetime import datetime
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

class DocumentProcessor:
    def __init__(self):
        # Configurazioni base
        self.base_dir = "vector_databases"
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = "gpt2"
        self.batch_size = 32
        self.chunk_size = 500
        self.chunk_overlap = 50
        
        # Inizializzazione componenti
        self._initialize_components()
        
    def _initialize_components(self):
        """Inizializza i componenti base del processor"""
        os.makedirs(self.base_dir, exist_ok=True)
        self.db_paths = self._scan_existing_databases()
        self.embedding_function = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.current_db = None
        self.current_db_path = None
        
        # Carica automaticamente l'ultimo database se disponibile
        self._ensure_database_loaded()
    
    def _ensure_database_loaded(self) -> bool:
        """Assicura che ci sia un database caricato e utilizzabile"""
        if self.current_db is not None:
            return True
            
        # Tenta di caricare l'ultimo database disponibile
        if self.db_paths:
            return self.load_database(self.db_paths[-1])
            
        # Se non ci sono database, ne crea uno nuovo
        return self.initialize_new_database() is not None
    
    def similarity_search(self, query: str, k: int = 5):
        """Esegue la ricerca di similarità assicurando che il database sia caricato"""
        if not self._ensure_database_loaded():
            raise RuntimeError("Impossibile inizializzare o caricare un database")
            
        return self.current_db.similarity_search(query, k=k)
    
    def _clear_memory(self):
        """Pulisce la memoria GPU e CPU"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Forza la garbage collection
        gc.collect()

    def _scan_existing_databases(self):
        """Scansiona la directory base per trovare i database esistenti"""
        db_paths = []
        if os.path.exists(self.base_dir):
            # Ottiene tutte le sottodirectory che iniziano con 'db_'
            for item in os.listdir(self.base_dir):
                full_path = os.path.join(self.base_dir, item)
                if os.path.isdir(full_path) and item.startswith('db_'):
                    db_paths.append(full_path)
        
        # Ordina i percorsi per avere i database più recenti alla fine
        db_paths.sort()
        return db_paths

    def initialize_new_database(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_name = f"db_{timestamp}"
        db_path = os.path.join(self.base_dir, db_name)
        os.makedirs(db_path, exist_ok=True)
        
        self.db_paths.append(db_path)
        self.current_db_path = db_path
        
        # Initialize Chroma with client_settings for persistence
        self.current_db = Chroma(
            persist_directory=db_path,
            embedding_function=self.embedding_function,
            collection_name="documents"  # Add a collection name
        )
        
        return self.current_db

    def _create_db_for_batch(self, docs, batch_number):
        try:
            if self.current_db is None:
                raise ValueError("current_db non è stato inizializzato")
            
            print(f"Creazione database per il batch {batch_number}")
            
            # Create Document objects in the format Chroma expects
            ids = [f"doc_{batch_number}_{idx}" for idx in range(len(docs))]
            texts = [doc['text'] for doc in docs]
            metadatas = [doc['metadata'] for doc in docs]
            
            # Add documents to the existing database with explicit IDs
            self.current_db.add_texts(
                metadatas=metadatas,
                texts=texts,
                ids=ids
            )
            
            return True
            
        except Exception as e:
            print(f"Errore nella creazione del batch {batch_number}: {str(e)}")
            return False

    def process_documents(self, file_paths):
        # Verifica se ci sono file da processare
        if not file_paths:
            print("Nessun file da processare")
            return False
        
        try:
            # Inizializza una nuova directory del database e assicurati che current_db sia impostato
            self.current_db = self.initialize_new_database()
            print(f"Inizializzato nuovo database in: {self.current_db_path}")
            total_chunks_processed = 0
            
            # Itera attraverso ciascun file
            for file_path in file_paths:
                try:
                    # Legge il contenuto del file
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    
                    # Divide il testo in chunk
                    chunks = self.text_splitter.split_text(content)
                    
                    # Calcola il numero di batch necessari
                    num_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
                    
                    print(f"Numero di batch: {num_batches}")
                    
                    # Itera attraverso i chunk in batch
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * self.batch_size
                        end_idx = min((batch_idx + 1) * self.batch_size, len(chunks))
                        batch_chunks = chunks[start_idx:end_idx]
                        
                        print(f"Batch {batch_idx}")
                        
                        # Crea una lista di Documenti con contenuto e metadata
                        batch_docs = [
                            {
                                'text': chunk,
                                'metadata': {
                                    'source': file_path,
                                    'chunk_index': idx + start_idx
                                }
                            }
                            for idx, chunk in enumerate(batch_chunks)
                        ]
                        
                        # Crea il database per il batch corrente
                        batch_path = self._create_db_for_batch(batch_docs, batch_idx)
                        
                        if batch_path:
                            # Aggiorna il conteggio totale dei chunk processati
                            total_chunks_processed += len(batch_chunks)
                            
                            # Pulisce la memoria dopo ogni batch
                            self._clear_memory()
                        
                except Exception as e:
                    print(f"Errore nel processamento del file {file_path}: {str(e)}")
                    continue
            
            print(f"Processamento completato. Totale chunks processati: {total_chunks_processed}")
            # Chiudi eventuali database aperti
            if self.current_db is not None:
                try:
                    self.current_db = None
                except:
                    pass   
                
            return True

        except Exception as e:
            print(f"Errore durante il processamento dei documenti: {str(e)}")
            return False
        
                
        

    def configure(self, chunk_size=None, chunk_overlap=None, batch_size=None):
        """Configura i parametri del processore"""
        if chunk_size is not None:
            self.chunk_size = chunk_size
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
        
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
        
        if batch_size is not None:
            self.batch_size = batch_size

    def load_database(self, db_path=None):
        """
        Carica un database esistente. Se db_path non è specificato,
        tenta di caricare l'ultimo database creato.
        """
        try:
            # Se non viene specificato un percorso, usa l'ultimo database creato
            if db_path is None:
                if not self.db_paths:
                    raise ValueError("Nessun database disponibile")
                db_path = self.db_paths[-1]
            
            # Verifica che il percorso esista
            if not os.path.exists(db_path):
                raise ValueError(f"Il percorso del database non esiste: {db_path}")
            
            # Carica il database
            self.current_db = Chroma(
                persist_directory=db_path,
                embedding_function=self.embedding_function
            )
            self.current_db_path = db_path
            print(f"Database caricato con successo da: {db_path}")
            return True
            
        except Exception as e:
            print(f"Errore nel caricamento del database: {str(e)}")
            return False

    def get_latest_database_path(self):
        """Restituisce il percorso dell'ultimo database creato"""
        if not self.db_paths:
            return None
        return self.db_paths[-1]
