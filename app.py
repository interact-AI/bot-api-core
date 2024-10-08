from flask import Flask, request, jsonify
from langgraph_bot import execute_agent
import time
from send_to_logger import create_sender, send_message

app = Flask(__name__)
sender = create_sender()


@app.route("/status", methods=["GET"])
def status():
    return "The server is up and running!"


@app.route("/message", methods=["POST"])
async def message():
    message = request.json.get("message")
    conversation_id = request.headers.get("conversationId")
    initial_time = time.time()
    result = execute_agent(message, conversation_id)
    billable_time = time.time() - initial_time
    print(
        f"Tiempo de respuesta de Agente entero: {
            billable_time}"
    )

    response = {"response": result["final_response"]}

    info_for_logger = {
        "conversationId": conversation_id,
        "message": message,
        "response": result["final_response"],
        "timeStamp": time.time(),
        # TODO: Change ownerId and phoneNumber to the real owner
        "ownerId": 1,
        "phoneNumber": "+598 099669085",
    }
    time_to_send = time.time()
    try:
        async with sender:
            await send_message(sender, info_for_logger)
    except Exception as e:
        print(f"Error al enviar a logger {e}")
    print(
        f"Tiempo de respuesta envio a logger: {
            time.time() - time_to_send}"
    )
    return jsonify(response)


if __name__ == "__main__":

    app.run(debug=False, host="0.0.0.0", port=80)
