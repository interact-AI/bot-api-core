# -*- coding: utf-8 -*-
"""LangGraph Memory Test"""

from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
import getpass
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
from langsmith import traceable
import re

load_dotenv()
print("Connecting to SQL Database...")
db_uri = os.getenv("SQL_DATABASE_URI")
db = SQLDatabase.from_uri(db_uri)
print("Connected to SQL Database!")
# PRODUCT SEARCH DEPENDENCIES


# PRODUCT SEARCH DEPENDENCIES

GROQ_LLM_70 = ChatGroq(
    model="llama3-70b-8192",
)

conversation_with_summary_70b = ConversationChain(
    llm=GROQ_LLM_70,
)

GROQ_LLM_8 = ChatGroq(
    model="llama3-8b-8192",
)

conversation_with_summary_8b = ConversationChain(
    llm=GROQ_LLM_8,
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
    relevant_products: List[str]
    product_query: str


"""## Nodes

1. categorize_question
2. product_inquiry_response
3. other_inquiry_response(RAG)
4. state_printer
"""


@traceable
def categorize_question(state: GraphState):
    initial_time = time.time()
    # Definir la plantilla del prompt
    # Este nodo no sabe de medicamentos, por lo que mediante el prompt debemos buscar evitar el siguiente comportamiento:
    #   Pregunta: Hola, necesito perifar, que opciones tienen?
    #   Categoría de Pregunta: otro
    #   Respuesta: Lo siento, no tengo información sobre eso.
    prompt_template = """\
    Eres un Agente categorizador de mensajes recibidos por una farmacia. Respondes en español.
    Realiza un análisis exhaustivo del mensaje proporcionado y clasifícalo en una de las siguientes categorías:
        producto - cuando el mensaje está relacionado con productos o medicamentos de la farmacia. Un producto puede ser un mediacmento del que no conoces el nombre, si en la pregunta hay un nombre desconocido, se debe clasificar como producto.
        otro - cuando el mensaje no está relacionado con productos de la farmacia.

    Proporciona una sola palabra con la categoría (producto, otro)
    ej:
    producto

    CONTENIDO DE LA PREGUNTA:\n {initial_question} \n
    """
    print("---CLASIFICANDO PREGUNTA INICIAL---")
    initial_question = state["initial_question"]
    num_steps = int(state["num_steps"])
    num_steps += 1

    # Inicializar historial de conversación si no está presente
    if state.get("conversation_history") is None:
        state["conversation_history"] = []

    # Agregar la nueva pregunta al historial de conversación
    state["conversation_history"].append(
        {"rol": "usuario", "contenido": initial_question}
    )

    # Combinar el historial de conversación en un solo prompt
    combined_prompt = ""
    # TODO: Agregar el historial de conversación al prompt. Que el historial de conversación no contenga el mensaje actual, solo los anteriores.
    # for mensaje in state['conversation_history']:
    #     combined_prompt += f"{mensaje['rol'].capitalize()}: {mensaje['contenido']}\n"

    # Agregar la pregunta actual al prompt
    combined_prompt += prompt_template.format(initial_question=initial_question)

    # Invocar la cadena de conversación con el prompt combinado
    respuesta = conversation_with_summary_8b.predict(input=combined_prompt)
    # Analizar la respuesta para extraer la etiqueta de categoría
    if "producto" in respuesta.lower():
        categoria_pregunta = "producto"
    else:
        categoria_pregunta = "otro"

    print(categoria_pregunta)

    state.update({"question_category": categoria_pregunta, "num_steps": num_steps})

    print(f"Tiempo de respuesta de NODO: {time.time() - initial_time}")

    return state


def extract_sql_query(input_string):
    match = re.search(r'SQLQuery:\s*"([^"]*)"', input_string)
    if match:
        return match.group(1)
    else:
        return None


@traceable
def product_search(state):
    print("-- SQL DATABASE --")
    print(f"SQL DATABASE DIALECT: {db.dialect}")
    print(f"SQL DATABASE TABLES: {db.get_usable_table_names()}")
    table_name = os.getenv("MEDICAMENTOS_TABLE")
    table_info = db.get_table_info([table_name])

    prompt_template = PromptTemplate(
        template="""\
        Dada una pregunta de entrada, primero cree una consulta {dialect} sintácticamente correcta para ejecutar, luego observe los resultados de la consulta y devuelva la respuesta.
Utilice el siguiente formato:

Pregunta: "Pregunta aquí"
SQLQuery: "Consulta SQL para ejecutar"
SQLResult: "Resultado de la consulta SQL"
Respuesta: "Respuesta final aquí"

Use solo la siguientes tabla "{table_name}":

{table_info}

Esta pregunta puede contener un nombre completo o parcial de producto o medicamento desconocido, en ese caso buscalo en la columna Nombre. De lo contrario, busca en la columna que te parezca pertinente.
Pregunta de entrada: {initial_question}
        """,
        input_variables=["dialect", "table_name", "table_info", "initial_question"],
    )

    prompt = prompt_template.template.format(
        initial_question=state["initial_question"],
        dialect=db.dialect,
        table_info=table_info,
        table_name=table_name,
    )
    response = conversation_with_summary_70b.predict(input=prompt)

    print(f"SQL chain response: {response}")
    sql_query = extract_sql_query(response)
    print(f"SQL Query: {sql_query}")
    result = db.run(sql_query)
    state["product_query"] = sql_query
    state["relevant_products"] = result

    return state


@traceable
def product_inquiry_response(state):
    initial_time = time.time()
    # Crear una plantilla de prompt que incluya los datos del producto
    prompt = PromptTemplate(
        template="""\
        Eres un Agente de Información de Productos que trabaja en una Farmacia. Eres un experto en entender sobre qué trata una pregunta y puedes proporcionar información de productos concisa y relevante.

        Por favor, proporciona una respuesta directa a la pregunta (CONTENIDO DE LA PREGUNTA) basada en los productos.
        Los productos fueron encontrados en la base de datos de la farmacia con la siguiente consulta SQL: {product_query}
        Productos encontrados:\n{products}\n
        CONTENIDO DE LA PREGUNTA:\n {initial_question} \n
        """,
        input_variables=["products", "initial_question"],
    )

    print("---REALIZANDO LLAMADA AL ENDPOINT DE PRODUCTOS---")
    products = state["relevant_products"]
    print("---DANDO RESPUESTA A LA CONSULTA DE PRODUCTOS---")
    initial_question = state["initial_question"]
    num_steps = int(state["num_steps"])
    num_steps += 1

    # Agregar la pregunta actual al prompt
    input = prompt.template.format(
        initial_question=initial_question,
        products=products,
        product_query=state["product_query"],
    )

    # Invocar la cadena de conversación con el prompt combinado
    print(input)
    response = conversation_with_summary_70b.predict(input=input)
    print(response)
    # Agregar la respuesta del modelo al historial de conversación
    state["conversation_history"].append({"rol": "asistente", "contenido": response})

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
    prompt = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )
    return prompt


chat_model = ChatGroq(temperature=0, model_name="llama3-8b-8192")
# chat_model = ChatGroq(temperature=0, model_name="Llama2-70b-4096")

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
client = QdrantClient(api_key=qdrant_api_key, url=qdrant_url)


def retrieval_qa_chain(llm, prompt, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
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
    num_steps = int(state["num_steps"])
    num_steps += 1

    chain = qa_bot()

    initial_question = state["initial_question"]

    response = chain.invoke({"query": initial_question})["result"]

    if "RESPUESTA_NO_ENCONTRADA" in response:
        response = "Lo siento, no tengo información sobre eso."

    state["conversation_history"].append({"rol": "asistente", "contenido": response})
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
    for message in state.get("conversation_history", []):
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
    question_category = state["question_category"]

    if question_category == "producto":
        print("---RUTA A RESPUESTA DE CONSULTA DE PRODUCTO---")
        return "product_response"
    elif question_category == "otro":
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
workflow.add_node("product_search", product_search)
workflow.add_node("product_inquiry_response", product_inquiry_response)
workflow.add_node("state_printer", state_printer)
workflow.add_node("other_inquiry_response", other_inquiry_response)

"""### Add Edges"""
workflow.set_entry_point("categorize_question")

workflow.add_conditional_edges(
    "categorize_question",
    route_to_respond,
    {
        "product_response": "product_search",
        "other_inquiry_response": "other_inquiry_response",
    },
)

workflow.add_edge("product_search", "product_inquiry_response")
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
        "conversation_id": conversation_id,
    }


def execute_agent(question, conversation_id):
    global states  # Ensure state is recognized as global
    state = retrieve_state(conversation_id)
    state["initial_question"] = question
    states.append(state)
    output = app.invoke(state)

    return output


# NO BORRAR!! PARA PRUEBAS DESDE CONSOLA


def main():
    while True:
        question = input("Ingresar la pregunta: ")
        conversation_id = int(input("Ingresar ID de conversación: "))
        output = execute_agent(question, conversation_id)
        print(output)


if __name__ == "__main__":
    main()
