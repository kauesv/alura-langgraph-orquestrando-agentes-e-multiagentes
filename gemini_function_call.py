from google import genai
from dotenv import load_dotenv
import os

# =========================
# CONFIGURAÇÕES GERAIS
# =========================
load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY2")

# =========================
# CLIENTE GEMINI
# =========================
client = genai.Client(api_key=gemini_api_key)

# =========================
# CLIENTE TAVILY
# =========================
from tavily import TavilyClient
tavily = TavilyClient(api_key=tavily_api_key)

def tavily_search(query: str) -> str:
    result = tavily.search(query=query, max_results=5, search_depth="basic")
    return str(result)

# =========================
# TAVILY SEARCH LANGCHAIN
# =========================
from langchain_tavily import TavilySearch
tavily_tool = TavilySearch(
    max_results=5,
    tavily_api_key=tavily_api_key
)

def tavily_search_2(query: str) -> str:
    result = tavily_tool.invoke({"query": query})
    return str(result)

# =========================
# RESPONSE GEMINI
# =========================
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Como está o tempo em São Paulo hoje (21/08/2025)?",
    config={
        "system_instruction": (
            "Se precisar de informações do tempo atualizadas, "
            "utilize a ferramenta tavily_search."
        ),
        "tools": [tavily_search_2],
    },
)
print(response)
#print(response.text)

# =========================
# USO DA TOOL FUNCTION CALLING
# =========================
used_tool = False

for item in response.automatic_function_calling_history:
    for part in item.parts:

        # Chamada da tool
        fc = getattr(part, "function_call", None)
        if fc:
            used_tool = True

            print("\n=== FUNCTION CALL ===")
            print("Tool:", fc.name)
            print("Args:", dict(fc.args))

        # Resposta da tool
        fr = getattr(part, "function_response", None)
        if fr:
            print("\n=== FUNCTION RESPONSE ===")
            print("Tool:", fr.name)
            print("Response:")
            print(fr.response)

print("\nUsou tool?", used_tool)