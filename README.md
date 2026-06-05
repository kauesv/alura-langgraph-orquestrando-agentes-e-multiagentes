# LangGraph: Orquestrando Agentes e Multiagentes

Repositorio com meus estudos praticos do curso da Alura sobre agentes com LangGraph, uso de ferramentas, memoria, checkpoints, fluxo humano-no-loop e composicao de agentes com Gemini.

O projeto esta organizado como uma trilha incremental de scripts, em que cada arquivo explora uma etapa diferente da construcao de agentes e grafos de execucao.

---

## Visao Geral

Neste repositorio voce vai encontrar exemplos de:

- integracao com `Gemini` via `google-genai` e `langchain-google-genai`
- agentes com ferramentas e roteamento via `LangGraph`
- busca externa com `Tavily`
- persistencia de estado com `SqliteSaver`
- interrupcao para aprovacao humana antes de executar ferramentas
- memoria semantica com `langmem`
- experimentos de raspagem web com `requests`, `BeautifulSoup` e `Selenium`
- um projeto do professor com interface `Gradio`

---

## Estrutura do Projeto

### Trilha principal

- `aula1.py`: introducao a chamadas com Gemini e a uma classe `Agent` customizada, incluindo um prompt no estilo ReAct.
- `aula1_melhorado.py`: refinamento dos conceitos iniciais da aula 1.
- `aula2.py`: agente com ferramenta de busca `Tavily`, loop `llm -> action` e geracao do grafo Mermaid.
- `aula3.1.py`: experimento de busca com `DuckDuckGo` e extracao de conteudo com `BeautifulSoup`.
- `aula3.2.py`: busca agenciada com `Tavily` e raspagem mais pesada com `Selenium` para paginas dinamicas.
- `aula4.py`: uso de `SqliteSaver` para checkpoints e continuidade de contexto entre execucoes.
- `aula5.1.py`: introducao a `interrupt_before` para pausar o grafo antes da chamada de ferramenta.
- `aula5.2.py`: inspecao do estado pausado e controle mais explicito da intervencao humana.
- `aula6.1.py`: fluxo de escrita com planejamento, pesquisa, geracao e reflexao usando `StateGraph`.
- `aula6.2.py`: versao mais limpa e documentada do pipeline iterativo de redacao.
- `aula7.py`: roteamento de e-mails entre triagem e agente de resposta.
- `aula8.py`: assistente executivo com triagem, ferramentas e memoria semantica.

### Experimentos auxiliares

- `gemini_function_call.py`: testes de function calling com o SDK do Google.
- `langchain_gemini_function_call.py`: variacao do mesmo tema usando abstractions do LangChain.

### Arquivos de apoio

- `.env.example`: variaveis de ambiente esperadas pelos scripts.
- `requirements.txt`: dependencias Python do repositorio.

---

## Tecnologias Utilizadas

- Python 3.11+
- LangGraph
- LangChain
- Google Gemini
- Tavily
- OpenRouter
- Gradio
- Selenium
- BeautifulSoup
- SQLite

---

## Configuracao do Ambiente

### 1. Criar e ativar o ambiente virtual

Windows PowerShell:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar as dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar as variaveis de ambiente

Copie o arquivo de exemplo:

```bash
copy .env.example .env
```

Depois preencha as chaves conforme os scripts que voce pretende executar:

```env
LANGSMITH_TRACING=true
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=""
LANGSMITH_PROJECT="Workspace 5"

OPENAI_API_KEY=""
GEMINI_API_KEY=""
TAVILY_API_KEY=""
OPENROUTER_API_KEY=""
```

Observacoes:

- `GEMINI_API_KEY` e a chave mais importante para a maior parte das aulas.
- `TAVILY_API_KEY` e necessaria nas aulas com busca web.
- `OPENROUTER_API_KEY` aparece em experimentos especificos.
- `LANGSMITH_*` e opcional, mas util para tracing.

---

## Como Executar

### Rodar scripts individuais

Exemplos:

```bash
python aula1.py
python aula2.py
python aula4.py
python aula7.py
python aula8.py
```

### Rodar o projeto com Gradio

```bash
python aula6.2.py
```
Isso sobe uma interface local para gerar redacoes com planejamento, pesquisa e revisao iterativa.

---

## Aprendizados Cobertos

- modelagem de estado com `TypedDict`
- composicao de nos e arestas com `StateGraph`
- tool calling com LangChain
- persistencia e retomada de execucao com checkpoint
- aprovacao humana antes de executar acoes
- roteamento entre agentes especializados
- uso de memoria semantica para assistentes
- combinacao entre pesquisa externa e geracao de respostas

---

## Observacoes

- Alguns scripts sao experimentais e podem exigir ajustes dependendo da versao das bibliotecas.
- O repositorio mistura exemplos do curso com adaptacoes pessoais e testes paralelos.
- Parte dos fluxos grava dados em `checkpoints.db`, entao o comportamento pode variar conforme o historico salvo.

---

## Contato

Para mais informações ou para discutir qualquer um dos repositórios, sinta-se à vontade para entrar em contato:

- **Email:** [kauesousavieira534@gmail.com](mailto:kauesousavieira534@gmail.com)
- **LinkedIn:** [LinkedIn](https://www.linkedin.com/in/kaue-sousa-vieira/)

---
Obrigado por visitar meu repositório!