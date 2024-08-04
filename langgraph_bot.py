# -*- coding: utf-8 -*-
"""LangGraph Memory Test"""

import requests
from typing import List, Dict
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain_groq import ChatGroq
from pprint import pprint
import os
import time
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()


GROQ_LLM = ChatGroq(
    model="llama3-70b-8192",
)

conversation_with_summary = ConversationChain(
    llm=GROQ_LLM,
)

GROQ_LLM = ChatGroq(
    model="llama3-8b-8192",
)

conversation_with_summary_8b = ConversationChain(
    llm=GROQ_LLM,
)

# llama3-70b-8192
# llama3-8b-8192


"""## Utils

## State
"""


# State


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        initial_question: question
        question_category: question category
        final_response: LLM generation
        num_steps: number of steps
        conversation_history: list of conversation history
    """
    initial_question: str
    question_category: str
    final_response: str
    num_steps: int
    conversation_history: List[Dict[str, str]]


"""## Nodes

1. categorize_question
2. product_inquiry_response
3. other_inquiry_response(RAG)
4. state_printer
"""


def categorize_question(state: GraphState):
    initial_time = time.time()
    # Definir la plantilla del prompt
    prompt_template = """\
    Eres un Agente Clasificador de Preguntas que responde en español. Eres un experto en entender de qué trata una pregunta y eres capaz de categorizarla de manera útil.

    Realiza un análisis exhaustivo de la pregunta proporcionada y clasifícala en una de las siguientes categorías:
        producto - utilizado cuando la pregunta está relacionada con un producto
        otro - utilizado cuando la pregunta no está relacionada con un producto

    Proporciona una sola palabra con la categoría (producto, otro)
    ej:
    producto

    CONTENIDO DE LA PREGUNTA:\n\n {initial_question} \n\n
    """
    print("---CLASIFICANDO PREGUNTA INICIAL---")
    initial_question = state['initial_question']
    num_steps = int(state['num_steps'])
    num_steps += 1

    # Inicializar historial de conversación si no está presente
    if state.get('conversation_history') is None:
        state['conversation_history'] = []

    # Agregar la nueva pregunta al historial de conversación
    state['conversation_history'].append(
        {"rol": "usuario", "contenido": initial_question})

    # Combinar el historial de conversación en un solo prompt
    combined_prompt = ""
    for mensaje in state['conversation_history']:
        combined_prompt += f"{mensaje['rol'].capitalize()
                              }: {mensaje['contenido']}\n"

    # Agregar la pregunta actual al prompt
    combined_prompt += prompt_template.format(
        initial_question=initial_question)

    # Invocar la cadena de conversación con el prompt combinado
    respuesta = conversation_with_summary_8b.predict(input=combined_prompt)
    # Analizar la respuesta para extraer la etiqueta de categoría
    if "producto" in respuesta.lower():
        categoria_pregunta = "producto"
    else:
        categoria_pregunta = "otro"

    print(categoria_pregunta)

    state.update(
        {"question_category": categoria_pregunta, "num_steps": num_steps})

    print(f"Tiempo de respuesta de NODO: {time.time() - initial_time}")

    return state


def product_inquiry_response(state):
    initial_time = time.time()
    # Crear una plantilla de prompt que incluya los datos del producto
    prompt = PromptTemplate(
        template="""\
        Eres un Agente de Información de Productos. Eres un experto en entender sobre qué trata una pregunta y puedes proporcionar información de productos concisa y relevante.

        Por favor, proporciona una respuesta directa a la pregunta (CONTENIDO DE LA PREGUNTA) basada en los datos del producto como si ya los conocieras.
        DATOS DEL PRODUCTO:\n{products}\n
        CONTENIDO DE LA PREGUNTA:\n\n {initial_question} \n\n
        """,
        input_variables=["products", "initial_question"],
    )

    print("---REALIZANDO LLAMADA AL ENDPOINT DE PRODUCTOS---")
    initial_time = time.time()
    products = [
        {
            "nombre": "Perifar",
            "precio": "$10 USD",
            "descripcion": "Medicamento para el dolor",
            "fabricante": "Laboratorios Smith",

        },
        {
            "nombre": "Panadol",
            "precio": "$15 USD",
            "descripcion": "Medicamento para el mareo",
            "fabricante": "Roemers",
        },
        {
            "nombre": "Ibuprofeno",
            "precio": "$20 USD",
            "descripcion": "Medicamento para la fiebre",
            "fabricante": "Laboratorios Jack",
        },
        {
            "nombre": "Paracetamol",
            "precio": "$25 USD",
            "descripcion": "Medicamento para el dolor de cabeza",
            "fabricante": "Laboratorios Sanofi",
        },
        {
            "nombre": "Aspirina",
            "precio": "$30 USD",
            "descripcion": "Medicamento para el dolor de muelas",
            "fabricante": "Laboratorios Smith",
        },
        {
            "nombre": "Diclofenaco",
            "precio": "$35 USD",
            "descripcion": "Medicamento para el dolor de espalda",
            "fabricante": "Laboratorios Smith",
        }
    ]
    print(f"Tiempo de respuesta de GET: {time.time() - initial_time}")
    print("---DANDO RESPUESTA A LA CONSULTA DE PRODUCTOS---")
    initial_question = state['initial_question']
    num_steps = int(state['num_steps'])
    num_steps += 1

    # Agregar la pregunta actual al prompt
    input = prompt.template.format(
        initial_question=initial_question, products=products)

    # Invocar la cadena de conversación con el prompt combinado
    print(input)
    response = conversation_with_summary.predict(input=input)
    print(response)
    # Agregar la respuesta del modelo al historial de conversación
    state['conversation_history'].append(
        {"rol": "asistente", "contenido": response})

    state.update({"final_response": response, "num_steps": num_steps})

    print(f"Tiempo de respuesta de NODO: {time.time() - initial_time}")
    
    return state

custom_prompt_template = """Utiliza la siguiente información para responder la pregunta del usuario.
Si no conoces la respuesta, devuelve "RESPUESTA_NO_ENCONTRADA". No trates de inventar una respuesta.


Contexto: {context}
Pregunta: {question}


Solo devuelve la respuesta útil a continuación y nada más.
Respuesta útil:
"""
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

chat_model = ChatGroq(temperature=0, model_name="llama3-8b-8192")
#chat_model = ChatGroq(temperature=0, model_name="Llama2-70b-4096")

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
client = QdrantClient(api_key=qdrant_api_key, url=qdrant_url)

def retrieval_qa_chain(llm, prompt, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def qa_bot():
    embeddings = FastEmbedEmbeddings()
    vectorstore = Qdrant(client=client, embeddings=embeddings, collection_name="rag")
    llm = chat_model
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, vectorstore)
    return qa

def other_inquiry_response(state):

    initial_time = time.time()
    print("---RESPONDIENDO A CONSULTA DESDE RAG---")
    num_steps = int(state['num_steps'])
    num_steps += 1

    chain = qa_bot()
    
    initial_question = state['initial_question']
    
    response = chain.invoke({"query": initial_question})["result"]

    if "RESPUESTA_NO_ENCONTRADA" in response:
        response = "Lo siento, no tengo información sobre eso."
    
    state['conversation_history'].append({"rol": "asistente", "contenido": response})
    state.update({"final_response": response, "num_steps": num_steps})

    print(f"Tiempo de respuesta de NODO: {time.time() - initial_time}")

    return state


def state_printer(state):
    """Imprime el estado."""
    print("---IMPRESORA DE ESTADO---")
    print(f"Pregunta Inicial: {state['initial_question']}\n")
    print(f"Categoría de Pregunta: {state['question_category']}\n")
    print(f"Respuesta: {state['final_response']}\n")
    print(f"Pasos: {state['num_steps']}\n")
    print("Historial de Conversación:")
    for message in state.get('conversation_history', []):
        print(f"{message['rol'].capitalize()}: {message['contenido']}")
    return


"""## Conditional Edges"""


def route_to_respond(state):
    """
    Ruta la pregunta a pregunta de producto o no.
    Args:
        state (dict): El estado actual del gráfico
    Returns:
        str: Siguiente nodo a llamar
    """

    print("---RUTA PARA RESPONDER---")
    question_category = state['question_category']

    if question_category == 'producto':
        print("---RUTA A RESPUESTA DE CONSULTA DE PRODUCTO---")
        return "product_inquiry_response"
    elif question_category == 'otro':
        print("---RUTA A RESPUESTA DE CONSULTA DESDE RAG---")
        return "other_inquiry_response"
    else:
        # Manejar categoría inesperada (error nodo clasificador)
        print("---CATEGORÍA INESPERADA---")
        return "state_printer" 
    
"""## Build the Graph

### Add Nodes
"""

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("categorize_question", categorize_question)
workflow.add_node("product_inquiry_response", product_inquiry_response)
workflow.add_node("state_printer", state_printer)
workflow.add_node("other_inquiry_response", other_inquiry_response)

"""### Add Edges"""

workflow.set_entry_point("categorize_question")

workflow.add_conditional_edges(
    "categorize_question",
    route_to_respond,
    {
        "product_inquiry_response": "product_inquiry_response",
        "other_inquiry_response": "other_inquiry_response",
    },
)

workflow.add_edge("product_inquiry_response", "state_printer")
workflow.add_edge("other_inquiry_response", "state_printer")
workflow.add_edge("state_printer", END)

# Compile
app = workflow.compile()

states = []


def retrieve_state(conversation_id):
    global states
    for state in states:
        if state["conversation_id"] == conversation_id:
            return state
    return {
        "initial_question": "",
        "question_category": "",
        "final_response": "",
        "num_steps": 0,
        "conversation_history": [],
        "conversation_id": conversation_id
    }


def execute_agent(question, conversation_id):
    global states  # Ensure state is recognized as global
    state = retrieve_state(conversation_id)
    state["initial_question"] = question
    states.append(state)
    output = app.invoke(state)

    return output


# NO BORRAR!! PARA PRUEBAS DESDE CONSOLA
# 
def main():
    while True:
        question = input("Ingresar la pregunta: ")
        conversation_id = int(input("Ingresar ID de conversación: "))
        try:
            output = execute_agent(question, conversation_id)
            print(output)
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()