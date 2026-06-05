from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
import os


# =========================
# CONFIGURAÇÕES GERAIS
# =========================
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# =========================
# Gemini
# =========================
Gemini_model = "gemini-2.5-flash"
# Gemini_model = "gemini-2.5-flash-lite"

llm = ChatGoogleGenerativeAI(
    model=Gemini_model,
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
        "ignore": "Newsletters de marketing, e-mails de spam, comunicados gerais da empresa",
        "notify": "Membro da equipe doente, notificações do sistema de build, atualizações de status de projeto",
        "respond": "Perguntas diretas de membros da equipe, solicitações de reunião, relatórios de bugs críticos",
    },
    "agent_instructions": "Use estas ferramentas quando apropriado para ajudar a gerenciar as tarefas de Sarah de forma eficiente."
}

email = {
    "from": "Alice Smith <alice.smith@company.com>",
    "to": "Sarah Chen <sarah.chen@company.com>",
    "subject": "Dúvida rápida sobre a documentação da API",
    "body": """
Olá Sarah,

Eu estava revisando a documentação da API para o novo serviço de autenticação e notei que alguns endpoints parecem estar faltando nas especificações. Você poderia me ajudar a esclarecer se isso foi intencional ou se devemos atualizar a documentação?

Especificamente, estou procurando por:
- /auth/refresh
- - /auth/validate

Obrigada!
Alice""",
}

# =========================
# Defini uma estrutura que ajuda a llm com a saida da informação. Definir um output parser
# =========================
class Router(BaseModel): 
    """Analisa o e-mail não lido e o roteia de acordo com seu conteúdo."""

    reasoning: str = Field(
        description="Raciocínio passo a passo por trás da classificação."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="A classificação de um e-mail: 'ignore' para e-mails irrelevantes, "
        "'notify' para informações importantes que não precisam de resposta, "
        "'respond' para e-mails que precisam de uma resposta",
    )

# =========================
# prompt final
# =========================
from projetos_professor.prompts import triage_system_prompt, triage_user_prompt

system_prompt = triage_system_prompt.format(
    full_name=profile["full_name"],
    name=profile["name"],
    examples=None,
    user_profile_background=profile["user_profile_background"],
    triage_no=prompt_instructions["triage_rules"]["ignore"],
    triage_notify=prompt_instructions["triage_rules"]["notify"],
    triage_email=prompt_instructions["triage_rules"]["respond"],
)

user_prompt = triage_user_prompt.format(
    author=email["from"],
    to=email["to"],
    subject=email["subject"],
    email_thread=email["body"],
)

# =========================
# 
# =========================
llm_router = llm.with_structured_output(Router)

# result = llm_router.invoke(
#     [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt},
#     ]
# )

# print(result)

# =========================
# Tools
# =========================
from langchain_core.tools import tool

@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Escreve e envia um e-mail."""
    # Resposta de placeholder - em um aplicativo real, enviaria o e-mail
    return f"E-mail enviado para {to} com o assunto '{subject}'"

@tool
def schedule_meeting(
    attendees: list[str], 
    subject: str, 
    duration_minutes: int, 
    preferred_day: str
) -> str:
    """Agenda uma reunião no calendário."""

    return f"Reunião '{subject}' agendada para {preferred_day} com {len(attendees)} participantes"

@tool
def check_calendar_availability(day: str) -> str:
    """Verifica a disponibilidade do calendário para um determinado dia."""

    return f"Horários disponíveis em {day}: 9:00 AM, 2:00 PM, 4:00 PM"

tools=[write_email, schedule_meeting, check_calendar_availability]

# =========================
# Nó com o status atual da conversa
# =========================
from projetos_professor.prompts import agent_system_prompt

def create_prompt(state):
    return [
        {
            "role": "system",
            "content": agent_system_prompt.format(
                instructions=prompt_instructions["agent_instructions"],
                **profile
            )
        }
    ] + state['messages']

# =========================
# Cria um agente react
#
# create_react_agent não é mais usado nas versões atuais, recomenda o create_agent do langchain, mas perdendo o uso da função create_prompt
# =========================
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,  
    tools=tools,
    prompt=create_prompt,
)

# =========================
# Testa o agente criado
# =========================
# response = agent.invoke(
#     {"messages": [{
#         "role": "user",
#         "content": "qual é minha disponibilidade para terça-feira?"
#     }]}
# )

# print(response["messages"][-1].pretty_print())

# =========================
# State do GRAFO
# =========================
from langgraph.graph import add_messages #Adiciona mensagem no final da lista

class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]

# =========================
# Triagem do router
# =========================
def triage_router(state: State) -> Command[
    Literal["response_agent", "__end__"]
]:
    author = state['email_input']['author']
    to = state['email_input']['to']
    subject = state['email_input']['subject']
    email_thread = state['email_input']['email_thread']

    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=prompt_instructions["triage_rules"]["ignore"],
        triage_notify=prompt_instructions["triage_rules"]["notify"],
        triage_email=prompt_instructions["triage_rules"]["respond"],
        examples=None
    )
    user_prompt = triage_user_prompt.format(
        author=author, 
        to=to, 
        subject=subject, 
        email_thread=email_thread
    )
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    if result.classification == "respond":
        print("📧 Classificação: RESPONDER - Este e-mail requer uma resposta")
        goto = "response_agent"
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Responda ao e-mail {state['email_input']}",
                }
            ]
        }
    elif result.classification == "ignore":
        print("🚫 Classificação: IGNORAR - Este e-mail pode ser ignorado com segurança")
        update = None
        goto = END
    elif result.classification == "notify":
        # Em um cenário real, isso faria outra coisa
        print("🔔 Classificação: NOTIFICAR - Este e-mail contém informações importantes")
        update = None
        goto = END
    else:
        raise ValueError(f"Classificação inválida: {result.classification}")
    return Command(goto=goto, update=update)

# =========================
# Fluxo com o stategraph
# =========================
email_agent = StateGraph(State)
email_agent = email_agent.add_node("triage_router", triage_router)
email_agent = email_agent.add_node("response_agent", agent)
email_agent = email_agent.add_edge(START, "triage_router")
email_agent = email_agent.compile()

# =========================
# TESTANDO com o uso do langgraph
# =========================
email_input_ignorado = {
    "author": "Equipe de Marketing <marketing@amazingdeals.com>",
    "to": "Sarah Chen <sarah.chen@company.com>",
    "subject": "🔥 OFERTA EXCLUSIVA: Desconto por Tempo Limitado em Ferramentas para Desenvolvedores! 🔥",
    "email_thread": """Prezado(a) Desenvolvedor(a),

Não perca esta oportunidade INCRÍVEL! 

🚀 POR TEMPO LIMITADO, obtenha 80% DE DESCONTO em nosso Pacote Premium para Desenvolvedores! 

✨ RECURSOS:
- Preenchimento de código revolucionário com IA
- Ambiente de desenvolvimento baseado em nuvem
- Suporte ao cliente 24/7
- E muito mais!

💰 Preço Normal: R$ 999/mês
🎉 SEU PREÇO ESPECIAL: Apenas R$ 199/mês!

🕒 Corra! Esta oferta expira em:
APENAS 24 HORAS!

Clique aqui para resgatar seu desconto: https://amazingdeals.com/special-offer

Atenciosamente,
Equipe de Marketing
---
Para cancelar a inscrição, clique aqui
""",
}

email_input_respondido = {
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "Sarah Chen <sarah.chen@company.com>",
    "subject": "Dúvida rápida sobre a documentação da API",
    "email_thread": """Olá Sarah,

Eu estava revisando a documentação da API para o novo serviço de autenticação e notei que alguns endpoints parecem estar faltando nas especificações. Você poderia me ajudar a esclarecer se isso foi intencional ou se devemos atualizar a documentação?

Especificamente, estou procurando por:
- /auth/refresh
- /auth/validate

Obrigada!
Alice""",
}

response = email_agent.invoke({"email_input": email_input_ignorado})

for m in response["messages"]:
    print(m.pretty_print())