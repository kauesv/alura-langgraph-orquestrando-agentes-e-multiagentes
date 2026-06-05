from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import operator
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AnyMessage, AIMessage
from typing_extensions import TypedDict
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import sqlite3
import os
from uuid import uuid4

# =========================
# CONFIGURAÇÕES GERAIS
# =========================
load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# =========================
# Gemini
# =========================
Gemini_model = "gemini-2.5-flash"
#Gemini_model = "gemini-2.5-flash-lite"

llm = ChatGoogleGenerativeAI(
    model=Gemini_model
)
# =========================
# TavilySearch
# =========================
Tavily_search_tool = TavilySearch(
    max_results=5,
    tavily_api_key=tavily_api_key
)

@tool
def tavily_search(query: str) -> str:
    """Pesquisa informações atualizadas na internet."""
    result = Tavily_search_tool.invoke({"query": query})
    return str(result)

# =========================
# Checkpoint - A gente defini como vai ficar em cada Nó
# =========================
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

# =========================
# Agente
# =========================
def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    """Combina listas de mensagens sem duplicar mensagens pelo id, atualizando mensagens existentes."""

    for message in right:
        if not message.id:
            message.id = str(uuid4())
    
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
    
            if existing.id == message.id:
                merged[i] = message
                break
        else:
    
            merged.append(message)
    return merged

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]

class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        self.tools = {tool.name: tool for tool in tools}
        self.model = model.bind_tools(tools)

        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_gemini)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")

        self.graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["action"]   # Adiciona interrupção antes de chamar a ação
        )

    def call_gemini(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages

        #print("Mensagens enviadas ao modelo:", messages)
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            #print(f"Calling Tool: {t['name']} with args: {t['args']}")
            tool = self.tools.get(t["name"])

            if tool is None:
                raise ValueError(f"Tool não encontrada: {t['name']}")

            result = tool.invoke(t["args"])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {'messages': results}

# =========================
# 
# =========================
from datetime import date
current_date = date.today().strftime("%d/%m/%Y") 

prompt = f"""Você é um assistente de pesquisa inteligente e altamente atualizado. \
Sua principal prioridade é encontrar as informações mais RECENTES e em TEMPO REAL sempre que possível. \
A data atual é {current_date}. \
Ao buscar sobre o tempo ou eventos que se referem a "hoje" ou "agora", \
você DEVE **incluir a data atual '{current_date}' na sua consulta para a ferramenta de busca**. \
Por exemplo, se a pergunta é "tempo em cidade x hoje", a consulta para a ferramenta deve ser "tempo em cidade x {current_date}". \
Ignore ou descarte informações que claramente se refiram a datas passadas ou futuras ao responder perguntas sobre "hoje". \
Use o mecanismo de busca para procurar informações, sempre buscando o 'hoje' ou o 'agora' quando o contexto indicar. \
Você tem permissão para fazer múltiplas chamadas (seja em conjunto ou em sequência). \
Procure informações apenas quando tiver certeza do que você quer. \
Se precisar pesquisar alguma informação antes de fazer uma pergunta de acompanhamento, você tem permissão para fazer isso!
"""

abot = Agent(model=llm, tools=[tavily_search], system=prompt, checkpointer=memory)

# =========================
# thread_id
# =========================
import uuid
dynamic_thread_id = str(uuid.uuid4())
print(f"Meu novo Thread ID dinâmico é: {dynamic_thread_id}")

# =========================
# Teste
# =========================
session_id = str(uuid.uuid4())
print(f"DEBUG: Iniciando nova conversa com ID: {session_id}\n")

user_message = "Como está o tempo em São Paulo hoje?"

messages = [HumanMessage(content=user_message)]
thread_config = {"configurable": {"thread_id": session_id}}

print("--- Etapa 1: Agente processa a entrada e decide a ação ---")
print(f"Você: {user_message}")

# Vai executar até o ponto de interrupção
for event in abot.graph.stream({"messages": messages}, thread_config):
    for k, v in event.items():
        if k == "llm":
            last_message = v.get('messages', [])[-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                print(f"\nAgente (decisão): {last_message.tool_calls}")
                print("\n--- AGENTE PAUSADO: Intervenção Humana Necessária ---")
            else:
                print(f"\nAgente (resposta direta/sem tool_calls): {last_message.content}")
                print("\n--- AGENTE PAUSADO (resposta direta, sem ação pendente) ---")

# apos o loop ele pergunta para o usuario
current_state = abot.graph.get_state(thread_config)

# ultima mensagem de estado
last_state_message = current_state.values['messages'][-1]

# Gerencia a intervenção humana
if current_state and current_state.next == ('action',) and isinstance(last_state_message, AIMessage) and last_state_message.tool_calls:
    tool_calls_pending = last_state_message.tool_calls
    if tool_calls_pending:
        print("\nO agente decidiu executar a(s) seguinte(s) ação(ões) de ferramenta:")
        for tc in tool_calls_pending:
            print(f"- Ferramenta: {tc['name']}, Argumentos: {tc['args']}")

        user_input = input("\nVocê deseja que o agente execute esta(s) ação(ões)? (sim/não): ").lower()

        # resposta do usuário
        if user_input == 'sim':
            
            # executa
            print("\n--- Etapa 2: Retomando a execução (Agente executará a ação) ---")
            for event in abot.graph.stream(None, thread_config):
                for k, v in event.items():
                    if k == "action":
                        print(f"DEBUG: Ferramenta executada e resultado retornado: {v}")
                    elif k == "llm":
                        final_response_message = v.get('messages', [])[-1].content
                        print(f"\nAgente (resposta final): {final_response_message}")
                    elif k == END:
                        print(f"DEBUG: Grafo terminou a execução.")
            print("\n--- FIM DA INTERAÇÃO ---")
        else:
            print("\nExecução da ação cancelada pelo usuário.")
            print("--- FIM DA INTERAÇÃO ---")
    else:
        print("\nO agente não decidiu nenhuma ação de ferramenta apesar da pausa. Interação encerrada.")
else:
    print("\nO agente respondeu diretamente ou não pausou em uma ação. Não há ações pendentes para aprovar.")
    if current_state:
        final_response_message = current_state.values['messages'][-1].content
        print(f"Agente (resposta direta): {final_response_message}")
    print("--- FIM DA INTERAÇÃO ---")
