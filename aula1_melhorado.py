import os
from google import genai
from dotenv import load_dotenv
from google.genai import types
from typing import TypedDict
import re 
import time
from openrouter import OpenRouter
from openrouter import errors
import sys
import json

# ATENÇÃO! A LIB QUE O PROFESSOR ESTA USANDO ESTA DEFASADA, AGORA O GOOGLE USA O SDK google-genai PARA AS APIS
# ESTOU USANDO O OPENROUTER, PQ BATI O LIMITE NO GOOGLE

# =========================
# CONFIGURAÇÕES GERAIS
# =========================
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# =========================
# Clients
# =========================
client = genai.Client(api_key=gemini_api_key)
client_openrouter = OpenRouter(api_key=openrouter_api_key)

# =========================
# Chamada LLM Direta
# =========================
# response = client.models.generate_content(
#     model="gemini-2.5-flash",
#     contents="Hello world"
# )

# print(response.text)

# =========================
# Criando a Classe Agent
# =========================
class Agent:
    def __init__(self, system="", sdk="Google", modelo="gemini-2.5-flash"):
        self.system = system
        self.sdk = sdk
        self.modelo = modelo
        self.messages = []
        print(f"SDK: {self.sdk}")
        print(f"Modelo: {self.modelo}")

    def __call__(self, message):
        if self.sdk == "openrouter":
            self.messages.append({
                "role": "user",
                "content": message
            })
        else:
            self.messages.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=message)]
                )
            )

        if self.sdk == "openrouter":
            result = self.execute_open_router()

            self.messages.append({
                "role": "assistant",
                "content": result
            })
        else:
            result = self.execute()

            self.messages.append(
                types.Content(
                    role="model",
                    parts=[types.Part(text=result)]
                )
            )

        return result

    def execute(self):
        response = client.models.generate_content(
            model=self.modelo,
            contents=self.messages,
            config=types.GenerateContentConfig(
                temperature=0,
                system_instruction=self.system if self.system else None
            )
        )

        return  f"Resposta: {response.text}"

    def execute_open_router(self):
        try:
            response = client_openrouter.chat.send(
                model=self.modelo,
                messages=self.messages
            )
        except errors.TooManyRequestsResponseError as e:
            return f"Modelo com limite/sobrecarga: {self.modelo}"

        return f"Resposta: {response.choices[0].message.content}"

# if __name__ == "__main__":

#     models_openrouter = [
#         "google/gemma-4-31b-it:free",
#         "baidu/cobuddy:free",
#         "deepseek/deepseek-v4-flash:free",
#     ]

#     agent_google = Agent(system="Você é um assistente útil e objetivo.")
#     print(agent_google("Qual é a capital da França?"))
#     print("\n" + "="*50 + "\n") 
#     agent_openrouter = Agent(system="Você é um assistente útil e objetivo.", sdk="openrouter", modelo="baidu/cobuddy:free")
#     print(agent_openrouter("Qual é a capital da França?"))
# sys.exit(0)

# =========================
# TOOLS PARA O OPENROUTER
# =========================
tools = [
    {
        "type": "function",
        "function": {
            "name": "consultar_estoque",
            "description": "retorna a quantidade disponível de um item no inventário (ex: teclado)",
            "parameters": {
                "type": "object",
                "properties": {
                    "nome_produto": {"type": "string"}
                },
                "required": ["nome_produto"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_preco_produto",
            "description": "retorna o preço unitário de um produto (ex: mouse gamer)",
            "parameters": {
                "type": "object",
                "properties": {
                    "nome_produto": {"type": "string"}
                },
                "required": ["nome_produto"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ferramenta_encontrar_produto_mais_caro",
            "description": "Retorna o nome e o preço do produto mais caro no inventário."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ferramenta_calcular_valor_total_lista",
            "description": "Calcula o valor total de uma lista de itens de compra. Recebe uma string com itens separados por vírgula (ex: teclado, mouse gamer, monitor)",
            "parameters": {
                "type": "object",
                "properties": {
                    "nomes_produtos": {"type": "string"}
                },
                "required": ["nomes_produtos"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "listar_produtos_disponiveis",
            "description": "Listar todos os nomes de produtos da loja",
        }
    }
]

# =========================
# 
# =========================
PROMPT = """
Você deve identificar se a pergunta do usuário precisa consultar ferramentas.

Use as ferramentas disponíveis quando precisar saber:
- quantidade em estoque
- preço de produto
- o produto mais caro no inventário
- calcular o valor total de uma lista de itens
- listas todos os produtos da loja

Não responda com texto explicativo.
""".strip()

def listar_produtos_disponiveis():
    """Retorna a lista completa dos nomes de produtos da loja"""
    produtos = [
        "monitor",
        "teclado",
        "mouse gamer",
        "webcam",
        "headset",
        "impressora"
    ]

    return f"Produto da loja: {produtos}." 

def consultar_estoque(item: str) -> str:
    """Retorna a quantidade disponível no estoque para um produto/item."""
    item = item.lower()
    estoque = {
        "monitor": 75,
        "teclado": 120,
        "mouse gamer": 80,
        "webcam": 40,
        "headset": 60,
        "impressora": 15
    }

    if item in estoque:
        return f"Temos {estoque[item]} {item}s em estoque."
    else:
        return f"Item '{item}' não encontrado no inventário."

def consultar_preco_produto(produto: str) -> str:
    """Retorna o preço atual de um produto."""
    produto = produto.lower()
    precos = {
        "monitor": 999.90,
        "teclado": 150.00,
        "mouse gamer": 99.50,
        "webcam": 120.00,
        "headset": 180.00,
        "impressora": 750.00
    }

    if produto in precos:
        return f"O preço de um(a) {produto} é R$ {precos[produto]:.2f}."
    else:
        return f"Produto '{produto}' não encontrado na lista de preços."

def ferramenta_calcular_valor_total_lista(lista_itens: str) -> str:
    """
    Calcula o valor total de uma lista de itens de compra.
    Recebe uma string com itens separados por vírgula (ex: "teclado, mouse gamer, monitor").
    """
    precos_do_inventario = { 
        "monitor": 999.90,
        "teclado": 150.00,
        "mouse gamer": 99.50,
        "webcam": 120.00,
        "headset": 180.00,
        "impressora": 750.00
    }

    itens_processados = [item.strip().lower() for item in lista_itens.split(',')]

    valor_total = 0.0
    itens_nao_encontrados = []


    for item in itens_processados: 
        if item in precos_do_inventario:
            valor_total += precos_do_inventario[item]
        else:
            itens_nao_encontrados.append(item)

    resposta = f"O valor total dos itens encontrados é R$ {valor_total:.2f}."
    if itens_nao_encontrados:
        resposta += f" Os seguintes itens não foram encontrados e não foram incluídos no cálculo: {', '.join(itens_nao_encontrados)}."

    return resposta

def ferramenta_encontrar_produto_mais_caro() -> str: 
    """
    Retorna o nome e o preço do produto mais caro no inventário.
    Esta função não precisa de argumentos.
    """
    
    precos_do_inventario = {
        "monitor": 999.90,
        "teclado": 150.00,
        "mouse gamer": 99.50,
        "webcam": 120.00,
        "headset": 180.00,
        "impressora": 750.00
    }

    if not precos_do_inventario: 
        return "Nenhum produto encontrado na lista de preços para comparação."

    
    nome_produto_mais_caro = max(precos_do_inventario, key=precos_do_inventario.get)
    valor_produto_mais_caro = precos_do_inventario[nome_produto_mais_caro]

    return f"O produto mais caro é o(a) {nome_produto_mais_caro} com preço de R$ {valor_produto_mais_caro:.2f}." 

def run_react_agent(pergunta: str, modelo: str, sdk: str) -> str:
    if sdk == "OpenRouter":
        history = [{
            "role": "system",
            "content": PROMPT
        }]
    else:
        history = []

    if sdk == "OpenRouter":
        history.append({
            "role": "user",
            "content": pergunta
        })

        try:
            response = client_openrouter.chat.send(
                model=modelo,
                messages=history,
                tools=tools,
                tool_choice="auto"
            )

            message = response.choices[0].message
            response_text = response.choices[0].message.content
        except errors.TooManyRequestsResponseError as e:
            response_text = f"Modelo com limite/sobrecarga: {modelo}"

        history.append({
            "role": "assistant",
            "content": response_text
        })
        if message.tool_calls:
            tool = message.tool_calls[0]
            action_name = tool.function.name
            argumentos = json.loads(tool.function.arguments)
            if argumentos:
                action_arg = list(argumentos.values())[0]
            else:
                action_arg = ""
       
    else:
        history.append(
            types.Content(
                role="user",
                parts=[types.Part(text=pergunta)]
            )
        )

        response = client.models.generate_content(
            model=modelo,
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=PROMPT,
                temperature=0,
                tools=[consultar_estoque,
                    consultar_preco_produto,
                    ferramenta_encontrar_produto_mais_caro,
                    ferramenta_calcular_valor_total_lista,
                    listar_produtos_disponiveis
                ]
            )
        )

        response_text = response.text.strip()

        history.append(
            types.Content(
                role="model",
                parts=[types.Part(text=response_text)]
            )
        )

        response_historico = response.automatic_function_calling_history
        if not response_historico:
            return "Nenhuma Tool foi chamada"
        else:
            for content in response_historico:
                for part in content.parts:
                    if part.function_call:
                        action_name = part.function_call.name
                        argumentos = dict(part.function_call.args)
                        if argumentos:
                            action_arg = list(argumentos.values())[0]
                        else:
                            action_arg = ""
                   
    print(f"Função: {action_name}")
    print(f"Produto(s): {action_arg}")

    if action_name == "consultar_estoque":
        observacao_da_acao = consultar_estoque(action_arg)
    elif action_name == "consultar_preco_produto":
        observacao_da_acao = consultar_preco_produto(action_arg)
    elif action_name == "encontrar_produto_mais_caro":
        observacao_da_acao = ferramenta_encontrar_produto_mais_caro()
    elif action_name == "calcular_valor_total_lista":
        observacao_da_acao = ferramenta_calcular_valor_total_lista(action_arg)
    elif action_name == "listar_produtos_disponiveis":
        observacao_da_acao = listar_produtos_disponiveis()
    else:
        observacao_da_acao = f"Erro: Ação '{action_name}' desconhecida. Verifique o prompt ou a implementação da ferramenta."

    return observacao_da_acao

# =========================
# TESTANDO O PROMPT REACT
# =========================
models_openrouter = [
    "google/gemma-4-31b-it:free",
    "baidu/cobuddy:free",
    "deepseek/deepseek-v4-flash:free",
]

models_google = [
    "gemini-2.5-flash"
]
pergunta_0 = "Quantos mouses gamers estão no inventário?"
pergunta_1 = "Quantos teclados temos em estoque?"
pergunta_2 = "Qual o preço de um headset?"
pergunta_3 = "Temos cadeiras em estoque?"
pergunta_4 = "Qual é o produto mais caro?"
pergunta_5 = "Qual o valor de um teclado, uma impressora e uma webcam?"
pergunta_6 = "Quais produtos tem na loja?"

def iniciar_conversacao_com_agente():
    print("--- Agente de Inventário Interativo ---")
    print("Digite sua pergunta sobre o inventário, ou digite 'sair' para encerrar.")
    print("-" * 50)

    while True:
        print("-" * 50)
        print("\nExemplos de SDKs e Modelos:")
        print(f"\nSDK OpenRouter:\n {models_openrouter}")
        print(f"\nSDK Google:\n {models_google}")

        sdk = input("\nSDK: ")
        modelo = input("\nModelo: ")
        pergunta_usuario = input("\nPergunta: ")

        if pergunta_usuario.lower().strip() == 'sair':
            print("Encerrando a conversa. Até logo!")
            break

        print("\nAgente: Processando...")
        try:

            resposta_agente = run_react_agent(pergunta=pergunta_usuario, modelo=modelo, sdk=sdk)
            print(f"\nAgente: {resposta_agente}")
        except Exception as e:

            print(f"\nAgente: Ocorreu um erro ao processar sua pergunta: {e}")
            print("Por favor, tente novamente ou digite 'sair'.")

if __name__ == "__main__":
    iniciar_conversacao_com_agente()