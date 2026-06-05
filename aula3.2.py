from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from selenium.common.exceptions import WebDriverException, TimeoutException
from tavily import TavilyClient
from dotenv import load_dotenv
import os
import re

# =========================
# CONFIGURAÇÕES GERAIS
# =========================
load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")

# =========================
# Clients
# =========================
cliente_tavily = TavilyClient(api_key=tavily_api_key)

# =========================
# Busca Agentica Tavily (Busca a url mais relevante)
# =========================
cidade = "Belém do Pará"
tavily_query = f"restaurantes em {cidade} tripadvisor com maior quantidade de reviews e faixa de preço"

print("Iniciando busca agêntica por URLs do Tripadvisor com Tavily...")
tripadvisor_url = None
try:
    tavily_results = cliente_tavily.search(query=tavily_query, max_results=5)

    if tavily_results and tavily_results["results"]:
        print(f"Tavily encontrou {len(tavily_results['results'])} resultados. Analisando...")

        for result in tavily_results["results"]:
            url = result["url"]
            
            if "tripadvisor.com" in url or "tripadvisor.com.br" in url:
                tripadvisor_url = url
                break

        if not tripadvisor_url:
            print("Nenhum URL relevante do Tripadvisor foi encontrado nos primeiros resultados.")
            
    else:
        print("Tavily não encontrou resultados para a busca agêntica.")

except Exception as e:
    print(f"Erro na busca agêntica com Tavily: {e}. Verifique sua chave API ou conexão.")

if tripadvisor_url:
    clean_url = re.sub(r'-oa\d+-', '-', tripadvisor_url)
    tripadvisor_url = clean_url
    print(f"  ✅ URL encontrada limpa de paginação.")

print("-" * 50)
print(f"URL Final do Tripadvisor para raspagem: {tripadvisor_url if tripadvisor_url else 'NÃO ENCONTRADO'}")
print("-" * 50)

# =========================
# Obtem o HTML e o titulo da pagina
# =========================
def scrape_restaurantes_info(url):
    if not url:
        print("Erro: URL vazia ou não localizada para raspagem.")
        return None

    try:
        service = Service(ChromeDriverManager().install())
        options = webdriver.ChromeOptions()
        # options.add_argument("--headless") # Segundo Plano
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
        
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(30)
        
    except Exception as e:
        print(f"Erro ao inicializar o driver do Selenium: {e}")
        return None

    try:
        print(f"Tentando carregar a página com Selenium: {url}")
        driver.get(url)
        import time
        time.sleep(20)

        driver.implicitly_wait(10)

        response_text = driver.page_source

    except TimeoutException:
        print(f"Erro de tempo limite ao carregar a página {url}.")
        return None
    except WebDriverException as e:
        print(f"Erro ao carregar a página {url} com Selenium: {e}. Pode ser um bloqueio ou problema de conexão.")
        return None
    finally:
        driver.quit()

    print(response_text)

    soup = BeautifulSoup(response_text, 'html.parser')
    return soup

soup_tripadvisor = None

if 'tripadvisor_url' in locals() and tripadvisor_url:
    print(f"\nTentando raspar a página identificada: {tripadvisor_url}")
    soup_tripadvisor = scrape_restaurantes_info(tripadvisor_url)

    if soup_tripadvisor:
        print("HTML da página do Tripadvisor obtido com sucesso!")
        page_title_tag = soup_tripadvisor.find('title')
        if page_title_tag:
            print(f"Título da página: {page_title_tag.get_text(strip=True)}")
        else:
            print("Não foi possível encontrar o título da página.")
    else:
        print("Falha ao raspar a página do Tripadvisor. Verifique o URL ou se o site está bloqueando.")
else:
    print("Não há URL do Tripadvisor válido para raspar (obtido no Bloco 1).")

print("-" * 50)

# =========================
# Validar HTML
# =========================
with open("pagina.html", "w", encoding="utf-8") as f:
    f.write(soup_tripadvisor)

# =========================
# Obtem os dados do HTML
# =========================
print("\nIniciando a extração detalhada dos restaurantes da página do Tripadvisor...")
restaurantes_detalhados = [] 

restaurant_blocks = soup_tripadvisor.find_all('div', class_=lambda c: c and 'tbrcR' in c)

if not restaurant_blocks:
    print("AVISO: Nenhum bloco de restaurante principal encontrado com os seletores configurados ('tbrcR').")
    print("Por favor, **REVISE A INSPEÇÃO MANUAL** se o layout da página mudou e atualize este seletor.")

top_n_restaurants = restaurant_blocks[:5] 

for block in top_n_restaurants:

    nome_link_tag = block.find('a', class_=lambda c: c and 'BMQDV' in c and 'ukgoS' in c)
    nome = "Nome não encontrado"
    if nome_link_tag:
        nome_div_tag = nome_link_tag.find('div', class_=lambda c: c and 'biGQs' in c and 'fiohW' in c)
        if nome_div_tag:
            nome = nome_div_tag.get_text(strip=True)

    reviews_tag = block.find('div', {'data-automation': 'bubbleReviewCount'})
    reviews = reviews_tag.find('span').get_text(strip=True) if reviews_tag and reviews_tag.find('span') else "Reviews não encontrados"


    rating_tag = block.find('div', {'data-automation': 'bubbleRatingValue'})
    rating = rating_tag.find('span').get_text(strip=True) if rating_tag and rating_tag.find('span') else "Avaliação não encontrada"


    culinaria_preco_div = block.find('div', class_=lambda c: c and 'ZvrsW' in c and 'biqBm' in c)

    tipo_culinaria = "Tipo de Culinária não encontrado"
    preco = "Preço não encontrado"

    if culinaria_preco_div:

        spans_info = culinaria_preco_div.find_all('span', class_=lambda c: c and 'biGQs' in c and 'pZUbB' in c and 'ZNjnF' in c)


        if len(spans_info) >= 1:

            tipo_culinaria = spans_info[0].get_text(strip=True)


            for i in range(1, len(spans_info)):
                text = spans_info[i].get_text(strip=True)
                if '$' in text: 
                    preco = text
                    break 


    localizacao = "Localização não especificada (do nome)"
    if ' - ' in nome:

        partes_nome = nome.split(' - ')
        if len(partes_nome) > 1:
            localizacao = partes_nome[-1].strip()


    link = "Link não encontrado"
    if nome_link_tag and 'href' in nome_link_tag.attrs:
        link = "https://www.tripadvisor.com.br" + nome_link_tag['href']


    restaurantes_detalhados.append({
        "Nome": nome,
        "Avaliação": rating,
        "Reviews": reviews,
        "Preço": preco,
        "Tipo Culinária": tipo_culinaria,
        "Localização": localizacao,
        "Link": link
    })


if restaurantes_detalhados:
    print(f"\n--- {len(restaurantes_detalhados)} Restaurantes Extraídos do Tripadvisor ---")
    for i, r in enumerate(restaurantes_detalhados):
        print(f"Restaurante #{i+1}:")
        print(f"  Nome: {r['Nome']}")
        print(f"  Avaliação: {r['Avaliação']}")
        print(f"  Reviews: {r['Reviews']}")
        print(f"  Preço: {r['Preço']}")
        print(f"  Tipo Culinária: {r['Tipo Culinária']}")
        print(f"  Localização: {r['Localização']}")
        print(f"  Link: {r['Link']}")
        print("-" * 40)
else:
    print("Nenhum detalhe de restaurante foi extraído. **Verifique os seletores HTML no Bloco 4**.")

print("-" * 50)