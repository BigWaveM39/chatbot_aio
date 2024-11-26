from rag.chat_modes import ChatModes
from chat.llm_manager import LLMManager
from dotenv import load_dotenv

def main():
    """Gestisce il menu principale dell'applicazione"""
    
    # Carica le variabili d'ambiente
    load_dotenv()
    
    # Inizializza le modalità di chat
    chat_modes = ChatModes()
    
    # Inizializza il modello LLM
    llm = LLMManager()
    
    while True:
        try:
            # Stampa il menu
            print("\n=== Menu Principale ===")
            print("1. Crea Database Documenti")
            print("2. Chat con RAG")
            print("3. Esci")
            
            choice = input("\nScegli un'opzione (1-3): ").strip()
            
            if choice == "1":
                try:
                    chat_modes.create_chroma_db()
                except Exception as e:
                    print(f"\nErrore durante la creazione del database: {str(e)}")
                    print("Verifica che i documenti siano presenti e accessibili.")
            elif choice == "2":
                try:
                    chat_modes.rag_chat(llm)
                except Exception as e:
                    print(f"\nErrore durante la chat RAG: {str(e)}")
            elif choice == "3":
                print("\nGrazie per aver usato il chatbot! Arrivederci!")
                break
            else:
                 print("\nScelta non valida. Per favore, seleziona un numero tra 1 e 3.")
                
        except Exception as e:
            print(f"\nErrore generico: {str(e)}")
            print("Si è verificato un errore. Riprova.")

if __name__ == "__main__":
    main()
