from dotenv import load_dotenv
import re
import os

# =========================
# CONFIGURAÇÕES GERAIS
# =========================
load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")

# =========================
# Busca Agentica Tavily
# =========================
# client = TavilyClient(api_key=tavily_api_key)
# result = client.search("O que são multiagentes de Inteligência Artificial?",
#                        include_answer=True)
# print(result["answer"])

# =========================
# Busca Regular DuckduckGo
# =========================
from ddgs import DDGS

cidade = "Belém"

query = f"""
Liste os 5 principais restaurantes em {cidade}, segundo avaliações recentes no TripAdvisor ou sites similares de turismo.
Para cada restaurante, informe:
- Tipo de culinária (ex: regional, italiana, japonesa)
- Uma breve descrição (máx. 2 linhas)
- Avaliação média (se disponível)
- Faixa de preço

Responda apenas com dados atualizados e relevantes para turistas.
"""

def search(query, max_results=6):
    try:
        results = DDGS().text(query, max_results=max_results)
        return [i["href"] for i in results]
    except Exception as e:
        raise e

# for link in search(query):
#     print(link)

# =========================
# Extraindo informações
# =========================
import requests
from bs4 import BeautifulSoup

def scrape_restaurantes_info(url):
    if not url:
        print("Erro: URL vazia ou não localizada.")
        return None

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
        'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Erro ao carregar a página {url}: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    return soup

search_results = search(query)
if search_results:
    url = search_results[0]
else:
    url = None

soup = scrape_restaurantes_info(url)

print(f"Website: {url}\n\n")

restaurante_text_data = []

for tag in soup.find_all(['h1', 'h2', 'h3', 'p']):
    if tag.name == 'h1' and 'restaurantes' in tag.get_text(" ", strip=True).lower():
        restaurante_text_data.append(f"Título da Página: {tag.get_text(' ', strip=True)}")

    elif tag.name in ['h2', 'h3'] or (tag.name == 'p' and 'destaque' in tag.get('class', [])):
        restaurante_text_data.append(f"Nome/Destaque: {tag.get_text(' ', strip=True)}")
        
    elif tag.name == 'p':
        
        if len(tag.get_text(" ", strip=True)) > 50:
            restaurante_text_data.append(f"Conteúdo: {tag.get_text(' ', strip=True)}")

final_restaurante_data = "\n".join(restaurante_text_data)

final_restaurante_data = re.sub(r'\s+', ' ', final_restaurante_data).strip()

print(f"Website: {url}\n\n")
print(final_restaurante_data)