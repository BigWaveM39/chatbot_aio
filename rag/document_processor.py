import os
import gc
import time
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

class DocumentProcessor:
    def __init__(self):
        self.base_directory = "chromadb"
        self.prefix_db_path = "chroma_"
        self.suffix_db_path = "_db_"
        
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,  # chunk_size=100 e chunk_overlap=50 sono valori adatti per la ricerca precisa ad esempio in elenchi. Per catturare più contesto per documenti descrittivi si consigliano valori più alti, ad esempio chunk_size=500 e chunk_overlap=200
            chunk_overlap=50,
            length_function=lambda text: len(tokenizer.encode(text)),
            separators=["\n\n", "\n", ".", "!", "?", ",", " "]
        )
        self.batch_size = 20
        self.db_paths = []


    def _clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        gc.collect()

    def initialize_new_database(self):
        """Inizializza una nuova directory del database con timestamp"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.base_directory = f"chroma_dbs_{timestamp}"
        os.makedirs(self.base_directory, exist_ok=True)
        return self.base_directory

    def _create_db_for_batch(self, docs, batch_number):
        try:
            db_path = os.path.join(self.base_directory, f"db_batch_{batch_number}")
            
            vector_store = Chroma.from_documents(
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
        self.initialize_new_database()
        print(f"Creating new database in: {self.base_directory}")
            
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
                    
                    if self._create_db_for_batch(docs, batch_number):
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

    def similarity_search(self, query, k=3):
        """Cerca nei database disponibili e combina i risultati usando uno score di similarità"""
        if not self.db_paths:
            return []
            
        all_results = []
        # Aumentiamo il numero di risultati per database per avere più candidati
        results_per_db = k * 2  
        
        try:
            # Cerca in ogni database
            for db_path in self.db_paths:
                try:
                    vector_store = Chroma(
                        persist_directory=db_path,
                        embedding_function=self.embedding_function
                    )
                    
                    # Esegui la ricerca con score di similarità
                    results_with_scores = vector_store.similarity_search_with_score(
                        query, 
                        k=results_per_db
                    )
                    
                    # Aggiungi i risultati con i loro score
                    for doc, score in results_with_scores:
                        all_results.append((doc, score))
                    
                    del vector_store
                    self._clear_memory()
                    
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
        finally:
            self._clear_memory()

    def get_total_chunks(self):
        return len(self.db_paths) * self.batch_sizeù
    
    def load_database(self):
        self.db_paths = [
            os.path.join(self.base_directory, d) 
            for d in os.listdir(self.base_directory) 
            if os.path.isdir(os.path.join(self.base_directory, d))
        ]
