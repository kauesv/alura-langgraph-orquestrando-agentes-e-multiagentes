from google import genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from openrouter import OpenRouter
from openrouter import errors

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
import sys
import os


# =========================
# CONFIGURAÇÕES GERAIS
# =========================
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# =========================
# Clients
# =========================
client_google = genai.Client(api_key=gemini_api_key)
client_openrouter = OpenRouter(api_key=openrouter_api_key)

# =========================
# Modelos
# =========================
modelo_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
modelo_openrouter = "baidu/cobuddy:free"

# =========================
# Tavily
# =========================
tool = TavilySearch(max_results=4)
# print(type(tool))
# print(tool.name)

# =========================
# Agents
# =========================
class AgentState(TypedDict): 
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_gemini)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_gemini(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:
                print("\n ....bad tool name....")
                result = "bad tool name, retry"
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}

# =========================
# Chamando Agents manualmente
# =========================
prompt = """Você é um assistente de pesquisa inteligente. Use o mecanismo de busca para procurar informações. \
Você tem permissão para fazer múltiplas chamadas (seja em conjunto ou em sequência). \
Procure informações apenas quando tiver certeza do que você quer. \
Se precisar pesquisar alguma informação antes de fazer uma pergunta de acompanhamento, você tem permissão para fazer isso!
"""

abot = Agent(modelo_google, [tool], system=prompt)

messages = [HumanMessage(content="Como está o tempo em São Paulo hoje?")]

print("Iniciando interação do agente:")
final_result_state = None

for s in abot.graph.stream({"messages": messages}):
    print(s)
    print("---")
    final_result_state = s

print("\nResultado Final:")
if final_result_state and 'llm' in final_result_state and final_result_state['llm']['messages']:
    print(f"{final_result_state['llm']['messages'][-1].content[0]["text"]}")
else:
    print("Nenhum resultado final ou resultado inesperado.")

# Usando o invoke
# result = abot.graph.invoke({"messages": messages})
# print(result)
# print(result['messages'][-1].content)

# =========================
# Chamando Agents interativo
# =========================
# print("\n--- Agente de Pesquisa Interativo ---")
# print("Digite sua pergunta ou 'sair' para encerrar.")

# while True:
#     user_input = input("\nVocê: ") 
#     if user_input.lower() == "sair":
#         print("Agente: Encerrando a conversa. Até logo!")
#         break

#     messages = [HumanMessage(content=user_input)]

#     print("Agente: Pensando e buscando...")
#     final_result_state = None
#     try:

#         current_state = {}
#         for s in abot.graph.stream({"messages": messages}):
#             current_state.update(s)

#         print("\nAgente:")

#         if 'llm' in current_state and 'messages' in current_state['llm'] and current_state['llm']['messages']:
#             final_message = current_state['llm']['messages'][-1]
#             if hasattr(final_message, 'content'):
#                 print(final_message.content)
#             else:
#                 print("Não foi possível extrair o conteúdo da resposta final do LLM.")
#         else:
#             print("Não foi possível obter uma resposta do agente para esta pergunta.")

#     except Exception as e:
#         print(f"Agente: Ocorreu um erro durante a execução: {e}")
#         print("Tente novamente ou digite 'sair'.")

# print("\n--- Conversa Encerrada ---")

# =========================
# Visualização do Graph
# =========================
from pathlib import Path

mermaid_code = abot.graph.get_graph().draw_mermaid()
# print(mermaid_code)

try:
    image_data = abot.graph.get_graph().draw_mermaid_png()

    # Caminho para salvar no diretório atual do projeto
    output_path = Path("./graph_mermaid.png")

    # Salva a imagem
    # with open(output_path, "wb") as f:
    #     f.write(image_data)

    # print(f"Imagem salva em: {output_path.resolve()}")
except Exception as e:
    print(f"Erro ao tentar gerar PNG do Mermaid: {e}")
    print("\nCertifique-se de que a sua versão do LangGraph possui o método `.draw_mermaid_png()`.")
    print("Como alternativa, use `.draw_mermaid()` para obter a string e visualizar externamente.")