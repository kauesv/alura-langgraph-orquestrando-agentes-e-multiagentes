from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.graph import StateGraph, START, END, add_messages
from typing_extensions import TypedDict, Literal, Annotated
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json

# =========================
# CONFIGURAÇÕES GERAIS
# =========================
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# =========================
# Gemini
# =========================
gemini_model = "gemini-2.5-flash"
gemini_embeddings = "models/gemini-embedding-001"

google_embeddings = GoogleGenerativeAIEmbeddings(model=gemini_embeddings)

llm = ChatGoogleGenerativeAI(
    model=gemini_model,
    temperature=0
)

# =========================
# 
# =========================
profile = {
    "name": "Sarah",
    "full_name": "Sarah Chen",
    "user_profile_background": "Engenheira de software sênior liderando uma equipe de 5 desenvolvedores",
}

prompt_instructions = {
    "triage_rules": {
        "ignore": "Newsletters, spam, comunicados gerais da empresa",
        "notify": "Membro da equipe doente, notificações do sistema de build, atualizações de status de projeto",
        "respond": "Perguntas diretas de membros da equipe, solicitações de reunião, relatórios de bugs críticos",
    },
    "agent_instructions": """Você é um assistente executivo altamente eficiente. Execute as tarefas solicitadas de forma direta, sem diálogos desnecessários. Responda apenas com a chamada das ferramentas, a menos que seja estritamente necessário fornecer uma resposta textual.

Tarefas:
- Responder ao e-mail de entrada.
- Salvar a tarefa de acompanhamento na memória.
- Sugerir uma próxima ação, como 'Gostaria que eu agendasse uma reunião para discutir isso?'"""
}

# =========================
# Roteador para classificar a intenção do e-mail
# Defini uma estrutura que ajuda a llm com a saida da informação. Definir um output parser
# =========================
class Router(BaseModel):
    # Raciocínio por trás da classificação
    reasoning: str
    # Classificação do e-mail como 'ignorar', 'notificar' ou 'responder'
    classification: Literal["ignore", "notify", "respond"]

llm_router = llm.with_structured_output(Router)

# =========================
# Prompt do sistema para a triagem de e-mails
# e
# Prompt do usuário, que contém os dados do e-mail
# =========================
triage_system_prompt = """Você é um assistente de triagem de e-mails.
Classifique os e-mails como 'ignore', 'notify' ou 'respond'.

Regras:
- ignore: {triage_no}
- notify: {triage_notify}
- respond: {triage_email}
"""

triage_user_prompt = """
De: {author}
Para: {to}
Assunto: {subject}
Corpo da Mensagem:
{email_thread}
"""

# =========================
# InMemoryStore - RAM
# =========================
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool # É uma lib para gerenciar memorias dos LLMs

store = InMemoryStore(index={"embed": google_embeddings})

manage_memory_tool = create_manage_memory_tool(
    store=store,
    namespace=("email_assistant", "{langgraph_user_id}", "collection")
)
search_memory_tool = create_search_memory_tool(
    store=store,
    namespace=("email_assistant", "{langgraph_user_id}", "collection")
)

# =========================
# Tools
# =========================
@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Envia um e-mail para o destinatário especificado com o assunto e conteúdo fornecidos."""
    return json.dumps({"to": to, "subject": subject, "content": content})

@tool
def schedule_meeting(attendees: list[str], subject: str, duration_minutes: int, preferred_day: str) -> str:
    """Agenda uma reunião com os participantes especificados."""
    return f"Reunião '{subject}' agendada para {preferred_day} com {len(attendees)} participantes"

@tool
def check_calendar_availability(day: str) -> str:
    """Verifica os horários disponíveis para o dia fornecido."""
    return f"Horários disponíveis em {day}: 9:00, 14:00, 16:00"

tools = [write_email, schedule_meeting, check_calendar_availability, manage_memory_tool, search_memory_tool]

# =========================
# Nó com o status atual da conversa
# =========================
agent_system_prompt_memory = """
< Função >
Você é o(a) assistente executivo(a) de {full_name}. Sua prioridade é maximizar o desempenho de {name}.
{instructions}
</ Função >

< Ferramentas >
1. write_email(to, subject, content) - Envia e-mails
2. schedule_meeting(attendees, subject, duration_minutes, preferred_day) - Agenda reuniões
3. check_calendar_availability(day) - Verifica horários disponíveis
4. manage_memory - Armazena informações na memória
5. search_memory - Busca informações na memória
</ Ferramentas >
"""

# Prompt dinâmico
def create_prompt(state):
    return [
        {"role": "system", "content": agent_system_prompt_memory.format(
            instructions=prompt_instructions["agent_instructions"], **profile
        )},
    ] + state['messages']

# =========================
# Cria um agente react
#
# create_react_agent não é mais usado nas versões atuais, recomenda o create_agent do langchain, mas perdendo o uso da função create_prompt
# =========================
response_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=create_prompt,
    store=store
)

# =========================
# State do GRAFO
# =========================
class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]

# =========================
# Triagem do router
# =========================
def triage_router(state: State) -> dict:
    author = state['email_input']['from']
    to = state['email_input']['to']
    subject = state['email_input']['subject']
    body = state['email_input']['body']

    system_prompt = triage_system_prompt.format(
        triage_no=prompt_instructions["triage_rules"]["ignore"],
        triage_notify=prompt_instructions["triage_rules"]["notify"],
        triage_email=prompt_instructions["triage_rules"]["respond"]
    )
    user_prompt = triage_user_prompt.format(
        author=author,
        to=to,
        subject=subject,
        email_thread=body
    )
    result = llm_router.invoke(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}]
    )

    if result.classification == "respond":
        message_content = json.dumps({
            "sender": author,
            "recipient": to,
            "subject": subject,
            "body": body
        })
        return {
            "messages": [
                HumanMessage(content=f"Respond to the email {message_content}")
            ],
            "__next__": "response_agent"
        }
    elif result.classification in ["ignore", "notify"]:
        return {"__next__": END}
    else:
        raise ValueError(f"Invalid classification: {result.classification}")

# =========================
# Router tools
# =========================
def route_to_tools(state):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tool_node"
    return END

# =========================
# Fluxo com do GRAFO
# =========================
builder = StateGraph(State)
builder.add_node("triage_router", triage_router)
builder.add_node("response_agent", response_agent)
builder.add_node("tool_node", ToolNode(tools))
builder.set_entry_point("triage_router")

# Condições - se no "__next__" for response_agent ele termina
builder.add_conditional_edges(
    "triage_router",
    lambda state: state.get("__next__", None),
    {"response_agent": "response_agent", "END": END}
)

# Condições - faz analise e valida se tem uma chamada de ferramenta
# se tiver chamada de ferramenta tool_node se não termina
builder.add_conditional_edges(
    "response_agent",
    route_to_tools,
    {"tool_node": "tool_node", END: END}
)

# Apos a ação do tool_node ele volta para o response_agent
builder.add_edge("tool_node", "response_agent")

email_agent = builder.compile()

# =========================
# Human in the loop
# =========================
def human_in_the_loop_schedule(email_sender, email_recipient, email_subject):
    # Checa memória de longo prazo se já há reunião
    search_results = search_memory_tool.invoke(
        {"query": f"Reunião agendada para {email_sender}"},
        config=config
    )

    if isinstance(search_results, str):
        # retorna string, transforma em dict vazio
        search_results = {}

    if search_results.get("results"):
        email_content = write_email.invoke(
            {
                "to": email_sender,
                "subject": f"Re: {email_subject}",
                "content": "Olá, acabei de agendar uma conversa contigo para discutirmos esse assunto."
            },
            config=config
        )
        print("================== Tool Message ==================")
        print(f"Name: write_email\nContent: {email_content}")
        return

    # pergunta, se quer ou nao agendar a reunião
    decision = input(f"Deseja agendar uma reunião para discutir o pedido de {email_sender}? (sim/não): ").strip().lower()
    
    if decision == "sim":
        meeting_output = schedule_meeting.invoke(
            {
                "attendees": [email_recipient.split('<')[0].strip(), email_sender.split('<')[0].strip()],
                "subject": f"Acompanhamento do pedido de {email_sender}",
                "duration_minutes": 30,
                "preferred_day": "amanhã"
            },
            config=config
        )
        print("================== Tool Message ==================")
        print(f"Name: schedule_meeting\nContent: {meeting_output}")

        memory_output = manage_memory_tool.invoke(
            {"action": "create", "content": f"Reunião agendada para discutir o pedido de {email_sender}"},
            config=config
        )
        print("================== Tool Message ==================")
        print(f"Name: manage_memory\nContent: {memory_output}")

        email_content = write_email.invoke(
            {
                "to": email_sender,
                "subject": f"Re: {email_subject}",
                "content": "Já agendei uma reunião contigo para discutirmos esse assunto."
            },
            config=config
        )
        print("================== Tool Message ==================")
        print(f"Name: write_email\nContent: {email_content}")

    else:
        email_content = write_email.invoke(
            {
                "to": email_sender,
                "subject": f"Re: {email_subject}",
                "content": "Estou acompanhando seu pedido e entrarei em contato assim que houver novidades."
            },
            config=config
        )
        print("================== Tool Message ==================")
        print(f"Name: write_email\nContent: {email_content}")

        memory_output = manage_memory_tool.invoke(
            {"action": "create", "content": f"E-mail de acompanhamento enviado para {email_sender}"},
            config=config
        )
        print("================== Tool Message ==================")
        print(f"Name: manage_memory\nContent: {memory_output}")

# =========================
# TESTE
# =========================
config = {"configurable": {"langgraph_user_id": "lance"}}

# Pré-popula a memória para teste
initial_memory_content = "Acompanhamento necessário: Alice Smith perguntou sobre os endpoints de API ausentes na documentação do serviço de autenticação (/auth/refresh e /auth/validate). Sarah precisa revisar e esclarecer se eles foram intencionalmente omitidos ou se a documentação precisa de atualização."
manage_memory_tool.invoke({"action": "create", "content": initial_memory_content}, config=config)

# e a Alice vai mandar um novo email sobre o que a gente populou
email_input = {
    "from": "Alice Smith <alice.smith@company.com>",
    "to": "Sarah Chen <sarah.chen@company.com>",
    "subject": "Acompanhamento",
    "body": "Olá Sarah, como está minha solicitação?"
}

# 1️⃣ Agente responde
response = email_agent.invoke({"email_input": email_input}, config=config)
for msg in response["messages"]:
    if isinstance(msg, ToolMessage):
        print("================== Mensagem da Ferramenta ==================")
        print(f"Nome: {msg.name}\nConteúdo: {msg.content}")

# 2️⃣ Human-in-the-loop
human_in_the_loop_schedule(email_input["from"], email_input["to"], email_input["subject"])