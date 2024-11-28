from flask import Flask, request, jsonify
from main import Chatbot

app = Flask(__name__)

chatbot = Chatbot(use_audio=False, stream=False, preload_audio=False)

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """
    Endpoint per interagire con il chatbot
    Riceve un messaggio e restituisce la risposta del chatbot
    """
    data = request.json
    user_message = data.get('message', '')
    
    for token, full_response in chatbot.generate_response(user_message):
        pass
    
    try:
        # Chiama il metodo di generazione risposta del tuo chatbot
        bot_response = full_response
        
        return jsonify({
            'response': bot_response,
            'status': 'success'\
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/reset', methods=['POST'])
def reset_conversation():
    """
    Endpoint per resettare la conversazione del chatbot
    """
    try:
        # chatbot.reset_conversation()
        return jsonify({
            'message': 'Conversazione resettata',
            'status': 'success'
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)