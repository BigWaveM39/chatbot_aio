import os
import json
from typing import List, Optional
from datetime import datetime
from .history import ChatHistory

class ChatHistoryManager:
    def __init__(self, base_dir: str = 'chat_histories'):
        self.base_dir = base_dir
        self.current_history: Optional[ChatHistory] = None
        os.makedirs(base_dir, exist_ok=True)
    
    def create_history(self, name: str) -> ChatHistory:
        """Creates a new chat history with the given name"""
        history_dir = self._get_history_path(name)
        if os.path.exists(history_dir):
            print(f"Chat history '{name}' already exists")
        else:   
            os.makedirs(history_dir)
            metadata = {
                'name': name,
                'created_at': datetime.now().isoformat(),
                'last_modified': datetime.now().isoformat()
            }
            self._save_metadata(name, metadata)
        
        self.current_history = ChatHistory(name, history_dir)
        return self.current_history
    
    def load_history(self, name: str) -> ChatHistory:
        """Loads an existing chat history"""
        history_dir = self._get_history_path(name)
        if not os.path.exists(history_dir):
            raise ValueError(f"Chat history '{name}' does not exist")
        
        self.current_history = ChatHistory(name, history_dir)
        return self.current_history
    
    def delete_history(self, name: str) -> bool:
        """Deletes a chat history"""
        history_dir = self._get_history_path(name)
        if not os.path.exists(history_dir):
            return False
        
        import shutil
        shutil.rmtree(history_dir)
        return True
    
    def list_histories(self) -> List[dict]:
        """Returns a list of all available chat histories with their metadata"""
        histories = []
        for name in os.listdir(self.base_dir):
            metadata_path = os.path.join(self._get_history_path(name), 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    histories.append(json.load(f))
        return histories
    
    def _get_history_path(self, name: str) -> str:
        return os.path.join(self.base_dir, name)
    
    def _save_metadata(self, name: str, metadata: dict):
        metadata_path = os.path.join(self._get_history_path(name), 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def history_exists(self, name: str) -> bool:
        """Checks if a chat history with the given name exists"""
        return os.path.exists(self._get_history_path(name))
