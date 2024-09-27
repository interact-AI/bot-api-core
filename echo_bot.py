# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from botbuilder.core import ActivityHandler, MessageFactory, TurnContext
from botbuilder.schema import ChannelAccount
from langgraph_bot import execute_agent
import time


class EchoBot(ActivityHandler):
    async def on_members_added_activity(
        self, members_added: [ChannelAccount], turn_context: TurnContext
    ):
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity(
                    "Buenas! Que tal? Soy un bot de lenguaje natural, ¿en qué puedo ayudarte?"
                )

    async def on_message_activity(self, turn_context: TurnContext):
        initial_time = time.time()
        result = execute_agent(
            turn_context.activity.text, turn_context.activity.conversation.id
        )
        print(f"Tiempo de respuesta de Agente entero: {time.time() - initial_time}")
        return await turn_context.send_activity(
            MessageFactory.text(result["final_response"])
        )
