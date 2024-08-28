from flask import Flask, request, jsonify
from langgraph_bot import execute_agent
import time

app = Flask(__name__)


@app.route('/status', methods=['GET'])
def status():
    return 'The server is up and running!'


@app.route('/message', methods=['POST'])
def message():
    message = request.json.get('message')
    conversation_id = request.headers.get('conversationId')
    initial_time = time.time()
    result = execute_agent(
        message, conversation_id
    )
    print(f"Tiempo de respuesta de Agente entero: {
          time.time() - initial_time}"
          )

    response = {
        'response': result["final_response"]
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=80)
