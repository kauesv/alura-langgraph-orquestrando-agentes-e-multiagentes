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
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

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

        self.graph = graph.compile(checkpointer=checkpointer)

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
prompt_system = """Você é um assistente de pesquisa inteligente. Use o mecanismo de busca (tavily_search) para procurar informações.
Você tem permissão para fazer múltiplas chamadas à ferramenta (em conjunto ou em sequência).
Busque informações apenas quando tiver certeza do que procurar.
Se precisar de mais detalhes para formular uma pergunta de acompanhamento, você tem permissão para fazer isso.
Quando solicitado a comparar informações (ex: qual é mais quente, maior, etc.), use as informações do histórico da conversa e dos resultados das ferramentas.
""" 

abot = Agent(model=llm, tools=[tavily_search], system=prompt_system, checkpointer=memory)

# =========================
# Teste
# =========================
thread = {"configurable": {"thread_id": "1"}}  # consegue "criar um novo chat", a memoria fica armazenada e conseguimos continuar com a memoria
# thread = {"configurable": {"thread_id": str(uuid4())}} 

cidade = "São Paulo"
# cidade = "RJ"
# cidade = "Coritiba"

messages = [HumanMessage(content=f"Como está o tempo em {cidade} hoje (21/08/2025)?")]
# messages = [HumanMessage(content=f"Em um rank, quais foi a cidade com temperatura mais alta?")]

print(f"\n--- Pergunta 1: Tempo em {cidade} ---")
for event in abot.graph.stream({"messages": messages}, thread):
    for k, v in event.items():
        if k in ("llm", "action"): 
             print(f"{k}: {v['messages']}")
