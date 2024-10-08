from azure.servicebus.aio import ServiceBusClient, ServiceBusSender
from azure.servicebus import ServiceBusMessage
from dotenv import load_dotenv
import os
import json

load_dotenv()

NAMESPACE_CONNECTION_STR = os.getenv("QUEUE_CON_STRING")
QUEUE_NAME = os.getenv("QUEUE_NAME")


def create_sender() -> ServiceBusSender:
    service_bus_client = ServiceBusClient.from_connection_string(
        conn_str=NAMESPACE_CONNECTION_STR, logging_enable=True
    )
    sender = service_bus_client.get_queue_sender(queue_name=QUEUE_NAME)
    return sender


async def send_message(sender: ServiceBusSender, info_payload: dict):
    message = [ServiceBusMessage(json.dumps(info_payload))]
    await sender.send_messages(message)
