import os
import gc
import time
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import json  # Assicurati di importare il modulo json

class DocumentProcessor:
    def __init__(self):
        self.base_directory = "chromadb"
        self.suffix_db_path = "_db_"
        
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # chunk_size=100 e chunk_overlap=50 sono valori adatti per la ricerca precisa ad esempio in elenchi. Per catturare più contesto per documenti descrittivi si consigliano valori più alti, ad esempio chunk_size=500 e chunk_overlap=200
            chunk_overlap=200,
            length_function=lambda text: len(tokenizer.encode(text)),
            separators=["\n\n", "\n", ".", "!", "?", ",", " "]
        )
        self.batch_size = 20
        self.db_paths = []
        self.vector_store = None
        
        self.load_database()


    def _clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        gc.collect()

    def create_metadata_file(self, safe_file_name, db_name, timestamp, original_file_path):
        """Crea un file metadata in formato JSON con informazioni sul database"""
        metadata = {
            "file_name": safe_file_name,
            "db_name": db_name,
            "original_file_path": original_file_path,
            "creation_date": timestamp,
        }
        
        # Crea il percorso per il file metadata
        metadata_dir = os.path.join(self.base_directory, db_name)
        os.makedirs(metadata_dir, exist_ok=True)  # Crea la directory se non esiste
        
        metadata_file_path = os.path.join(metadata_dir, "metadata.json")
        
        with open(metadata_file_path, 'w', encoding='utf-8') as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, indent=4)
        
        print(f"Metadata file created: {metadata_file_path}")

    def initialize_new_database(self, file_name):
        """Inizializza una nuova directory del database con timestamp e nome del file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Estrai solo il nome del file senza estensione
        safe_file_name = os.path.splitext(os.path.basename(file_name))[0]
        # Rimuovi caratteri non validi dal nome del file
        safe_file_name = "".join(c for c in safe_file_name if c.isalnum() or c in (' ', '_')).rstrip()
        db_name = f"{safe_file_name}{self.suffix_db_path}{timestamp}"
        os.makedirs(self.base_directory, exist_ok=True)
        
        # Crea il file metadata
        self.create_metadata_file(safe_file_name, db_name, timestamp, file_name)
        
        return db_name  # Restituisce il nome del database

    def _create_db_for_batch(self, db_name, docs, batch_number):
        try:
            db_path = os.path.join(self.base_directory, db_name, f"db_batch_{batch_number}")
            
            Chroma.from_documents(
                documents=docs,
                embedding=self.embedding_function,
                persist_directory=db_path
            )
            
            self.db_paths.append(db_path)
            return True
        except Exception as e:
            print(f"Error creating database for batch {batch_number}: {e}")
            return False

    def process_documents(self, file_paths):
        """Crea il database vettoriale dai file caricati"""
        if not file_paths:
            print("No files to process")
            return False
        
        # Inizializza una nuova directory del database prima di processare i documenti
        db_name = self.initialize_new_database(file_paths[0])
        print(f"Creating new database in: {db_name}")
            
        total_chunks_processed = 0
        batch_number = len(self.db_paths)
        
        for file_path in file_paths:
            try:
                print(f"\nProcessing file: {file_path}")
                
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                
                print("Splitting document into chunks...")
                chunks = self.text_splitter.split_text(text)
                print(f"Created {len(chunks)} text chunks")
                
                batch_count = (len(chunks) + self.batch_size - 1) // self.batch_size
                print(f"Processing {batch_count} batches...")
                
                for batch_idx in range(0, len(chunks), self.batch_size):
                    current_batch = chunks[batch_idx:batch_idx + self.batch_size]
                    current_batch_num = batch_idx // self.batch_size + 1
                    
                    docs = [
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": file_path,
                                "chunk_id": total_chunks_processed + idx,
                                "batch_number": batch_number
                            }
                        )
                        for idx, chunk in enumerate(current_batch)
                    ]
                    
                    if self._create_db_for_batch(db_name, docs, batch_number):
                        total_chunks_processed += len(current_batch)
                        print(f"Processed batch {current_batch_num}/{batch_count} "
                              f"- Total chunks: {total_chunks_processed}")
                        batch_number += 1
                    else:
                        print(f"Failed to process batch {current_batch_num}")
                    
                    self._clear_memory()
                
                print(f"Completed processing file: {file_path}")
                
            except Exception as file_error:
                print(f"Error processing file {file_path}: {file_error}")
                continue
            finally:
                self._clear_memory()

        print(f"\nSuccessfully processed {total_chunks_processed} total chunks")
        return total_chunks_processed > 0

    def similarity_search(self, query, k=2):
        """Cerca nei database disponibili e combina i risultati usando uno score di similarità"""
        if not self.db_paths or not self.vector_store:
            if not self.load_database():
                return []
            
        all_results = [] 
        
        try:
            # Cerca in ogni database
            for db_path in self.db_paths:
                try:

                    # Esegui la ricerca con score di similarità
                    results_with_scores = self.vector_store.similarity_search_with_score(
                        query, 
                        k=k
                    )
                    
                    # Aggiungi i risultati con i loro score
                    for doc, score in results_with_scores:
                        all_results.append((doc, score))
                    
                except Exception as db_error:
                    print(f"Error searching in database {db_path}: {db_error}")
                    continue
            
            # Ordina tutti i risultati per score di similarità (score più basso = più rilevante)
            all_results.sort(key=lambda x: x[1])
            
            # Prendi i top k risultati
            top_results = [doc for doc, _ in all_results[:k]]
            
            # Debug: stampa informazioni sui risultati selezionati
            print(f"\nSelected {len(top_results)} most relevant chunks from {len(all_results)} total candidates")
            for i, doc in enumerate(top_results, 1):
                print(f"\nChunk {i}:")
                print(f"Batch: {doc.metadata.get('batch_number')}")
                print(f"Chunk ID: {doc.metadata.get('chunk_id')}")
                print(f"Source: {doc.metadata.get('source')}")
            
            return top_results
            
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []

    def get_total_chunks(self):
        return len(self.db_paths) * self.batch_sizeù
    
    def load_database(self, db_name=None) -> bool:
        """Carica il database specificato o il primo database disponibile nella base_directory"""
        
        try:
            # Ottieni la lista delle directory nel base_directory
            db_directories = [d for d in os.listdir(self.base_directory) if os.path.isdir(os.path.join(self.base_directory, d))]
            
            if not db_directories:
                print("Nessun database disponibile da caricare.")
                return False
            
            self.vector_store = None
            self.db_paths = []
            self._clear_memory()
            
            if db_name is None:
                db_path = os.path.join(self.base_directory, db_directories[0])
            else:
                db_path = os.path.join(self.base_directory, db_name)
            
            print(f"Caricando il database: {db_path}")
            
            self.db_paths = [os.path.join(db_path, d) for d in os.listdir(db_path) if d.startswith("db_batch_") and os.path.isdir(os.path.join(db_path, d))]
            
            for db_batch_path in self.db_paths:
                # Carica il database (puoi aggiungere qui la logica per caricare il database)
                self.vector_store = Chroma(persist_directory=db_batch_path, embedding_function=self.embedding_function)
                
            return True
            
        except Exception as e:
            print(f"Errore durante il caricamento del database: {e}")
            return False