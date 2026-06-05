from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain.agents import create_agent
import os

# =========================
# CONFIGURAÇÕES GERAIS
# =========================
load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# =========================
# Modelo
# =========================
Gemini_model = "gemini-2.5-flash"
# Gemini_model = "gemini-2.5-flash-lite"

llm = ChatGoogleGenerativeAI(
    model=Gemini_model
)

# =========================
# tavily
# =========================
tavily = TavilySearch(max_results=3)

@tool
def pesquisar_web(query: str) -> str:
    """Pesquisa informações atualizadas na internet."""
    print(f"🔍 Pesquisando: {query}")
    resultado = tavily.invoke({"query": query})
    return str(resultado)

# =========================
# Create Agent
# =========================
agent = create_agent(
    model=llm,
    tools=[
        pesquisar_web
    ]
)

# =========================
# Invoca o agent
# =========================
response = agent.invoke({
    "messages": [
        {
            "role": "system",
            "content": "Se precisar de informações do tempo atualizadas, utilize a ferramenta tavily_search."
        },
        {
            "role": "user",
            "content": "Como está o tempo em São Paulo hoje (21/08/2025)?"
        }
    ]
})

# =========================
# Result
# =========================
# print(response["messages"][-1].content)

for msg in response["messages"]:
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print("🔧 Tool chamada:")
        print(msg.tool_calls)

    if msg.__class__.__name__ == "ToolMessage":
        print("\n📄 Resposta da tool:")
        print(msg.content)