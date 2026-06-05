import os
from google import genai
from dotenv import load_dotenv
from google.genai import types
from typing import TypedDict
import re 
import sys

# ATENÇÃO! A LIB QUE O PROFESSOR ESTA USANDO ESTA DEFASADA, AGORA O GOOGLE USA O SDK google-genai PARA AS APIS

# =========================
# CONFIGURAÇÕES GERAIS
# =========================
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# =========================
# Clients
# =========================
client = genai.Client(api_key=gemini_api_key)
modelo = "gemini-2.5-flash"

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

# if __name__ == "__main__":

#     agent_google = Agent(system="Você é um assistente útil e objetivo.")
#     print(agent_google("Qual é a capital da França?"))
# sys.exit(0)

# =========================
# PROMPT REACT
# =========================
PROMPT_REACT = """
Você funciona em um ciclo de Pensamento, Ação, Pausa e Observação.
Ao final do ciclo, você fornece uma Resposta.
Use "Pensamento" para descrever seu raciocínio.
Use "Ação" para executar ferramentas - e então retorne "PAUSA".
A "Observação" será o resultado da ação executada.
Ações disponíveis:
  - consultar_estoque: retorna a quantidade disponível de um item no inventário (ex: "consultar_estoque: teclado")
  - consultar_preco_produto: retorna o preço unitário de um produto (ex: "consultar_preco_produto: mouse gamer")
  - encontrar_produto_mais_caro: retorna o nome e o preço do produto mais caro no inventário (não requer argumentos)
  - calcular_valor_total_lista: calcula o valor total de uma lista de itens de compra. Recebe uma string com itens separados por vírgula (ex: "teclado, mouse gamer, monitor")

Exemplo:
Pergunta: Quantos monitores temos em estoque?
Pensamento: Devo consultar a ação consultar_estoque para saber a quantidade de monitores.
Ação: consultar_estoque: monitor
PAUSA

Observação: Temos 75 monitores em estoque.
Resposta: Há 75 monitores em estoque.

Exemplo:
Pergunta: Qual é o produto mais caro?
Pensamento: Preciso usar a ação encontrar_produto_mais_caro para descobrir qual produto tem o maior preço.
Ação: encontrar_produto_mais_caro
PAUSA

Observação: O produto mais caro é o(a) monitor com preço de R$ 999.90.
Resposta: O produto mais caro é o(a) monitor com preço de R$ 999.90.

Exemplo:
Pergunta: Quanto custa um teclado e um mouse gamer?
Pensamento: O usuário quer saber o valor total de vários itens. Devo usar a ação calcular_valor_total_lista com os itens "teclado, mouse gamer".
Ação: calcular_valor_total_lista: teclado, mouse gamer
PAUSA

Observação: O valor total dos itens encontrados é R$ 249.50.
Resposta: O valor total do teclado e do mouse gamer é R$ 249.50.
""".strip()

def consultar_estoque(item: str) -> str:
    """
    Simula a consulta de estoque de um item no inventário.
    """
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
    """
    Simula a consulta do preço unitário de um produto.
    """
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

# print(consultar_estoque("mouse gamer"))
# print(consultar_preco_produto("impressora"))
# print(consultar_estoque("monitor"))

def run_react_agent(pergunta: str, max_iterations: int = 5) -> str:
    current_prompt = pergunta
    for i in range(max_iterations):
        response = client.models.generate_content(
            model=modelo,
            contents=[(
                types.Content(
                    role="user",
                    parts=[types.Part(text=current_prompt)]
                )
            )],
            config=types.GenerateContentConfig(
                system_instruction=PROMPT_REACT,
                temperature=0
            )
        )

        response_text = response.text.strip()

        print(f"\n--- Iteração {i+1} ---")
        print(f"Modelo pensou/respondeu:\n{response_text}\n")

        response_match_final = re.search(r"Resposta:\s*(.*)", response_text, re.DOTALL)
        if response_match_final:
            return response_match_final.group(1).strip()

        match = re.search(r"Ação:\s*(\w+)(?::\s*([^\n]*))?", response_text)

        if match:
            action_name = match.group(1).strip()
            action_arg = match.group(2).strip() if match.group(2) is not None else ""

            observacao_da_acao = ""

            if action_name == "consultar_estoque":
                observacao_da_acao = consultar_estoque(action_arg)
            elif action_name == "consultar_preco_produto":
                observacao_da_acao = consultar_preco_produto(action_arg)
            elif action_name == "encontrar_produto_mais_caro":
                observacao_da_acao = ferramenta_encontrar_produto_mais_caro()
            elif action_name == "calcular_valor_total_lista":
                observacao_da_acao = ferramenta_calcular_valor_total_lista(action_arg)
            else:
                observacao_da_acao = f"Erro: Ação '{action_name}' desconhecida. Verifique o prompt ou a implementação da ferramenta."

            current_prompt = f"Observação: {observacao_da_acao}"

            print(f"Executou ação: {action_name} com argumento '{action_arg}'")
            print(f"Observação: {observacao_da_acao}\n")

        else:
            return f"Erro: O agente não conseguiu extrair uma Ação ou Resposta final após {i+1} iterações. Última resposta do modelo: {response_text}"

    return "Erro: Limite máximo de iterações atingido sem uma resposta final do agente."

# =========================
# TESTANDO O PROMPT REACT
# =========================
def iniciar_conversacao_com_agente():
    print("--- Agente de Inventário Interativo ---")
    print("Digite sua pergunta sobre o inventário, ou digite 'sair' para encerrar.")
    print("-" * 50)

    while True:
        pergunta_usuario = input("\nVocê: ")

        if pergunta_usuario.lower().strip() == 'sair':
            print("Encerrando a conversa. Até logo!")
            break

        print("\nAgente: Processando...")
        try:

            resposta_agente = run_react_agent(pergunta_usuario)
            print(f"\nAgente: {resposta_agente}")
        except Exception as e:

            print(f"\nAgente: Ocorreu um erro ao processar sua pergunta: {e}")
            print("Por favor, tente novamente ou digite 'sair'.")

if __name__ == "__main__":
    iniciar_conversacao_com_agente()