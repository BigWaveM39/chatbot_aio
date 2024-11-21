from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTextEdit, QPushButton, QCheckBox,
                            QComboBox, QInputDialog, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon
from main import Chatbot
import sys

class StreamWorker(QThread):
    token_received = pyqtSignal(str, str)
    
    def __init__(self, chatbot, message, stream):
        super().__init__()
        self.chatbot = chatbot
        self.message = message
        self.stream = stream
        
    def run(self):
        for token, full_response in self.chatbot.generate_response(self.message, stream=self.stream):
            self.token_received.emit(token, full_response)

class AudioRecordWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, chatbot):
        super().__init__()
        self.chatbot = chatbot
        
    def run(self):
        try:
            audio_file = self.chatbot.audio_recorder.record()
            text = self.chatbot.audio_transcriber.transcribe(audio_file)
            self.finished.emit(text)
        except OSError as e:
            self.error.emit(f"Errore di registrazione audio: {str(e)}")
        except Exception as e:
            self.error.emit(f"Errore: {str(e)}")

class ChatbotGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.chatbot = Chatbot(use_audio=False, stream=False, preload_audio=True)
        self.init_ui()
        self.current_response = ""
        self.update_chat_list()
        
    def init_ui(self):
        self.setWindowTitle('Chatbot Interface')
        self.setGeometry(100, 100, 800, 600)
        
        # Widget principale
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Aggiungi il layout per la gestione delle chat prima dell'area chat
        chat_management_layout = QHBoxLayout()
        
        # Dropdown per selezionare la chat
        self.chat_selector = QComboBox()
        self.chat_selector.currentTextChanged.connect(self.load_selected_chat)
        chat_management_layout.addWidget(self.chat_selector)
        
        # Pulsanti per gestire le chat
        new_chat_button = QPushButton('Nuova Chat')
        new_chat_button.clicked.connect(self.create_new_chat)
        chat_management_layout.addWidget(new_chat_button)
        
        delete_chat_button = QPushButton('Elimina Chat')
        delete_chat_button.clicked.connect(self.delete_current_chat)
        chat_management_layout.addWidget(delete_chat_button)
        
        # Aggiungi il layout di gestione chat al layout principale
        layout.addLayout(chat_management_layout)
        
        # Area chat
        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        layout.addWidget(self.chat_area)
        
        # Area input
        input_layout = QHBoxLayout()
        self.input_field = QTextEdit()
        self.input_field.setMaximumHeight(100)
        input_layout.addWidget(self.input_field)
        
        # Pulsanti e controlli
        button_layout = QVBoxLayout()
        
        # Crea un layout orizzontale per il microfono e la checkbox
        mic_layout = QHBoxLayout()
        
        self.record_button = QPushButton()
        self.record_button.setIcon(QIcon('assets/mic_icon.png'))
        self.record_button.clicked.connect(self.start_recording)
        self.record_button.setEnabled(False)
        mic_layout.addWidget(self.record_button)
        
        self.auto_send_checkbox = QCheckBox('Auto-invio')
        self.auto_send_checkbox.setChecked(True)
        mic_layout.addWidget(self.auto_send_checkbox)
        
        # Aggiungi il layout del microfono al layout principale dei pulsanti
        button_layout.addLayout(mic_layout)
        
        self.send_button = QPushButton('Invia')
        self.send_button.clicked.connect(self.send_message)
        button_layout.addWidget(self.send_button)
        
        self.audio_checkbox = QCheckBox('Usa Audio')
        self.audio_checkbox.stateChanged.connect(self.toggle_audio)
        button_layout.addWidget(self.audio_checkbox)
        
        self.stream_checkbox = QCheckBox('Stream')
        self.stream_checkbox.stateChanged.connect(self.toggle_stream)
        button_layout.addWidget(self.stream_checkbox)
        
        input_layout.addLayout(button_layout)
        layout.addLayout(input_layout)
        
        # Shortcuts
        self.input_field.installEventFilter(self)
        
    def eventFilter(self, obj, event):
        if obj == self.input_field and event.type() == event.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.NoModifier:
                self.send_message()
                return True
        return super().eventFilter(obj, event)
    
    def send_message(self):
        message = self.input_field.toPlainText().strip()
        if not message:
            return
            
        # Aggiungi il messaggio dell'utente alla chat
        self.chat_area.append(f"Tu: {message}")
        self.input_field.clear()
    
        # Prepara l'area per la risposta del bot
        self.chat_area.append("Bot: ")
        self.current_response = ""
        
        # Gestione streaming
        self.stream_worker = StreamWorker(self.chatbot, message, self.stream_checkbox.isChecked())
        self.stream_worker.token_received.connect(self.handle_stream_token)
        self.stream_worker.finished.connect(self.handle_stream_finished)
        self.stream_worker.start()
       
    
    def handle_stream_token(self, token, full_response):
        # Aggiorna l'ultima riga con la risposta completa aggiornata
        cursor = self.chat_area.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.movePosition(
            cursor.MoveOperation.StartOfBlock, 
            cursor.MoveMode.KeepAnchor
        )
        cursor.removeSelectedText()
        cursor.insertText(f"Bot: {full_response}")
        
        # Aggiorna la risposta corrente
        self.current_response = full_response
    
    def handle_stream_finished(self):
        pass
    
    def handle_response(self, response):
        self.chat_area.append(f"Bot: {response}")
        
    def clear_chat(self):
        self.chat_area.clear()
        
    def toggle_audio(self, state):
        self.chatbot.use_audio = state
        self.record_button.setEnabled(state)
    
    def toggle_stream(self, state):
        self.chatbot.stream = state
    
    def start_recording(self):
        # Disabilita i controlli durante la registrazione
        self.record_button.setEnabled(False)
        self.send_button.setEnabled(False)
        self.input_field.setEnabled(False)
        
        # Avvia il worker per la registrazione
        self.audio_worker = AudioRecordWorker(self.chatbot)
        self.audio_worker.finished.connect(self.handle_audio_input)
        self.audio_worker.error.connect(self.handle_audio_error)
        self.audio_worker.start()
    
    def handle_audio_input(self, text):
        # Riabilita i controlli
        self.record_button.setEnabled(True)
        self.send_button.setEnabled(True)
        self.input_field.setEnabled(True)
        
        # Inserisce il testo trascritto nel campo di input
        self.input_field.setPlainText(text)
        
        # Invia automaticamente se la checkbox è selezionata
        if self.auto_send_checkbox.isChecked():
            self.send_message()
    
    def handle_audio_error(self, error_message):
        # Riabilita i controlli
        self.record_button.setEnabled(True)
        self.send_button.setEnabled(True)
        self.input_field.setEnabled(True)
        
        # Mostra l'errore nella chat
        self.chat_area.append(f"Errore: {error_message}")
    
    def update_chat_list(self):
        """Aggiorna la lista delle chat disponibili nel dropdown"""
        current_text = self.chat_selector.currentText()
        self.chat_selector.clear()
        
        chats = self.chatbot.list_chats()
        for chat in chats:
            self.chat_selector.addItem(chat['name'])
            
        # Ripristina la selezione precedente se possibile
        index = self.chat_selector.findText(current_text)
        if index >= 0:
            self.chat_selector.setCurrentIndex(index)
    
    def create_new_chat(self):
        """Crea una nuova chat"""
        name, ok = QInputDialog.getText(self, 'Nuova Chat', 
                                      'Inserisci il nome della nuova chat:')
        if ok and name:
            try:
                self.chatbot.create_new_chat(name)
                self.update_chat_list()
                self.chat_selector.setCurrentText(name)
                self.clear_chat()
            except ValueError as e:
                QMessageBox.warning(self, 'Errore', str(e))
    
    def delete_current_chat(self):
        """Elimina la chat corrente"""
        current_chat = self.chat_selector.currentText()
        if current_chat == "default":
            self.chatbot.get_current_chat().clear()
            self.chat_area.clear()
            return
            
        reply = QMessageBox.question(self, 'Conferma', 
                                   f'Vuoi davvero eliminare la chat "{current_chat}"?',
                                   QMessageBox.StandardButton.Yes | 
                                   QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            self.chatbot.delete_chat(current_chat)
            self.update_chat_list()
            # Carica la chat default dopo l'eliminazione
            self.chatbot.load_chat("default")
            self.clear_chat()
    
    def load_selected_chat(self, chat_name):
        """Carica la chat selezionata"""
        if chat_name:
            try:
                self.chatbot.load_chat(chat_name)
                # Aggiorna l'area chat con la cronologia della chat caricata
                self.chat_area.clear()
                for message in self.chatbot.get_current_chat().get_history():
                    role = "Tu" if message['role'] == "user" else "Bot"
                    self.chat_area.append(f"{role}: {message['content']}")
            except ValueError as e:
                QMessageBox.warning(self, 'Errore', str(e))

def main():
    app = QApplication(sys.argv)
    window = ChatbotGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
