from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import TypedDict
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from typing import TypedDict, List
import sqlite3
import json
import os

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
    model=Gemini_model,
    temperature=0
)

# =========================
# TavilySearch
# =========================
Tavily_search_tool = TavilySearch(
    max_results=3,
    tavily_api_key=tavily_api_key
)

def tavily_search(query: str) -> str:
    """Pesquisa informações atualizadas na internet."""
    result = Tavily_search_tool.invoke({"query": query})
    return result

# =========================
# Checkpoint - A gente defini como vai ficar em cada Nó
# =========================
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

# =========================
# AgenteState
# =========================
class AgentState(TypedDict):
    task: str  # tarefa do usuário
    plan: str  # plano para escrever (PLAN_PROMPT)
    draft: str  # rascunho do texto (WRITER_PROMPT)
    critique: str  # análise/crítica do texto (REFLECTION_PROMPT)
    content: List[str]  # lista de documentos pesquisados (Tavily) (RESEARCH_PLAN_PROMPT) (RESEARCH_CRITIQUE_PROMPT)
    revision_number: int  # número da revisão atual
    max_revisions: int  # número máximo de revisões

# =========================
# Prompts
# =========================
PLAN_PROMPT = """Você é um escritor especialista com a tarefa de criar um esboço de alto nível para uma redação. \
Escreva esse esboço para o tópico fornecido pelo usuário. Apresente um plano da redação junto com quaisquer notas \
ou instruções relevantes para as seções."""

WRITER_PROMPT = """Você é um assistente de redação com a tarefa de escrever excelentes redações de 5 parágrafos. \
Gere a melhor redação possível para a solicitação do usuário e o esboço inicial. \
Se o usuário fornecer críticas, responda com uma versão revisada das suas tentativas anteriores. \
Utilize todas as informações abaixo conforme necessário:

------

{content}"""

REFLECTION_PROMPT = """Você é um professor corrigindo uma redação submetida. \
Gere uma crítica e recomendações para a submissão do usuário. \
Forneça recomendações detalhadas, incluindo pedidos sobre extensão, profundidade, estilo, etc."""

RESEARCH_PLAN_PROMPT = """Você é um pesquisador encarregado de fornecer informações que podem \
ser usadas ao escrever a seguinte redação. Gere uma lista de consultas de pesquisa que \
recolham quaisquer informações relevantes. Gere no máximo 3 consultas."""

RESEARCH_CRITIQUE_PROMPT = """Você é um pesquisador encarregado de fornecer informações que podem \
ser usadas ao fazer quaisquer revisões solicitadas (conforme descrito abaixo). \
Gere uma lista de consultas de pesquisa que recolham quaisquer informações relevantes. Gere no máximo 3 consultas."""

# =========================
# BaseModel
# =========================
from pydantic import BaseModel

class Queries(BaseModel):
    queries: List[str]

# =========================
# Regras dos agentes - regras do fluxo
#
# planejador para o escritor, para o pensador e assim por diante. Isso não significa que ele não possa se retroalimentar, 
# ou seja, ele pode passar por mais de uma etapa mais de uma vez. Após isso, geraremos uma crítica, uma análise em relação ao que foi 
# escrito, e buscaremos na internet. Continuaremos utilizando o Tavilli, pois ele se mostra bastante eficiente. Por fim, os novos documentos 
# serão anexados a essa etapa de geração, onde começaremos a escrever novamente o ensaio, considerando que é um fluxo que se retroalimenta.
# =========================
def plan_node(state: AgentState):
    """
    Função responsável por gerar um plano de redação a partir de uma tarefa especificada no estado do agente.
    Utiliza o modelo LLM para criar um esboço de alto nível, fornecendo instruções ou notas relevantes para as seções do texto.
    
    Args:
        state (AgentState): Estado atual do agente, contendo ao menos a tarefa ('task').

    Returns:
        dict: Um dicionário contendo o plano gerado em 'plan'.
    """
    messages = [
        SystemMessage(content=PLAN_PROMPT), 
        HumanMessage(content=state['task'])
    ]
    response = llm.invoke(messages)
    return {"plan": response.content}

def research_plan_node(state: AgentState):
    """
    Função responsável por realizar pesquisas na internet com base em consultas geradas para embasar a redação.

    Esta função utiliza o modelo LLM para gerar uma lista de consultas de pesquisa relevantes ao tema da tarefa fornecida pelo usuário.
    Em seguida, realiza pesquisas utilizando a ferramenta TavilySearch para cada consulta gerada, agregando os resultados encontrados ao conteúdo do estado do agente.

    Args:
        state (AgentState): Estado atual do agente, contendo informações como a tarefa ('task') e o conteúdo previamente coletado ('content').

    Returns:
        dict: Um dicionário atualizado com o conteúdo ('content') que inclui todos os resultados das pesquisas realizadas.
    """
    queries = llm.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily_search(str(q))
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}

def generation_node(state: AgentState):
    """
    Função responsável por gerar um rascunho inicial do texto com base no plano criado e nos conteúdos pesquisados.

    Esta função utiliza o modelo LLM para redigir um rascunho a partir das informações contidas no estado do agente, 
    agregando o plano de redação ('plan'), os conteúdos coletados ('content') e a tarefa principal ('task').
    As mensagens necessárias são formatadas e passadas ao LLM, que retorna um texto rascunho. 
    O número de revisões é automaticamente incrementado.

    Args:
        state (AgentState): Estado atual do agente contendo a tarefa ('task'), o plano ('plan'), conteúdos ('content'), 
                            e o número de revisão ('revision_number').

    Returns:
        dict: Um dicionário com o 'draft' (rascunho gerado) e o número de revisão atualizado ('revision_number').
    """
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)
        ),
        user_message
        ]
    response = llm.invoke(messages)
    return {
        "draft": response.content, 
        "revision_number": state.get("revision_number", 1) + 1
    }

def reflection_node(state: AgentState):
    """
    Função responsável por realizar uma reflexão crítica sobre o rascunho gerado.

    Esta função utiliza um modelo LLM para analisar e criticar o conteúdo do rascunho fornecido no estado do agente.
    As mensagens apropriadas — um prompt de reflexão e o rascunho atual — são passadas ao modelo, que retorna uma
    crítica ou sugestão de melhoria.

    Args:
        state (AgentState): Estado atual do agente, que deve conter o rascunho ('draft') a ser avaliado.

    Returns:
        dict: Um dicionário contendo a crítica gerada pelo modelo sob a chave 'critique'.
    """
    messages = [
        SystemMessage(content=REFLECTION_PROMPT), 
        HumanMessage(content=state['draft'])
    ]
    response = llm.invoke(messages)
    return {"critique": response.content}

def research_critique_node(state: AgentState):
    """
    Função responsável por realizar uma nova pesquisa a partir das críticas geradas na etapa de reflexão.

    Esta função utiliza um modelo LLM para sugerir queries de pesquisa baseadas na crítica existente no estado do agente.
    Cada query sugerida é utilizada para buscar informações adicionais usando a função `tavily_search`. Os resultados dessas 
    pesquisas são agregados à lista de conteúdos do agente.

    Args:
        state (AgentState): Estado atual do agente, que deve conter uma crítica ('critique') e o conteúdo acumulado ('content').

    Returns:
        dict: Um dicionário contendo a lista atualizada de conteúdos sob a chave 'content'.
    """
    queries = llm.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily_search(str(q))
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}

def should_continue(state):
    """
    Determina se o fluxo de revisões deve continuar ou ser finalizado.

    Esta função compara o número atual de revisões ('revision_number') com o limite máximo permitido ('max_revisions').
    Se o número de revisões exceder o máximo, retorna o estado final (END); caso contrário, direciona o fluxo para a etapa de reflexão ("reflect").

    Args:
        state (dict): O estado atual, contendo ao menos as chaves 'revision_number' e 'max_revisions'.

    Returns:
        str: 'END' se o limite de revisões foi atingido, ou 'reflect' para continuar o ciclo.
    """
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"

# =========================
# Graph
#
# Por fim, garantiremos que todas as etapas do grafo tenham sido cumpridas e esperamos como 
# resposta um conteúdo criado que seja relevante em relação ao assunto que desejamos pesquisar.
# =========================
builder = StateGraph(AgentState)  # Define o estado base do agente

builder.add_node("planner", plan_node)  # Gera o plano inicial
builder.add_node("research_plan", research_plan_node)  # Pesquisa inicial baseada no plano
builder.add_node("generate", generation_node)  # Escreve o texto. Geração do texto/redação
builder.add_node("reflect", reflection_node)  # Analisa criticamente. Reflexão/análise sobre o texto gerado
builder.add_node("research_critique", research_critique_node)  # Pesquisa para revisar. Pesquisa iterativa baseada em críticas da reflexão

# Definição do ponto de entrada
builder.set_entry_point("planner")

# Lógica para encerrar ou continuar revisando
builder.add_conditional_edges(
    "generate", 
    should_continue, 
    {END: END, "reflect": "reflect"}
)

# Fluxo entre camadas de lógica
builder.add_edge("planner", "research_plan")                # Após planejar, pesquisa inicial
builder.add_edge("research_plan", "generate")               # Com pesquisa, escreve
builder.add_edge("reflect", "research_critique")            # Após reflexão, pesquisa para revisão
builder.add_edge("research_critique", "generate")           # Gera nova versão a partir da pesquisa crítica

graph = builder.compile(checkpointer=memory)  # Compila o grafo

# =========================
# Teste
# =========================
thread = {"configurable": {"thread_id": "1"}}
for s in graph.stream({
    'task': "Qual é a diferença entre o langchain e langsmith",
    "max_revisions": 2,
    "revision_number": 1,
    "content": [], 
}, thread):
    print("-" * 50)
    print(s)