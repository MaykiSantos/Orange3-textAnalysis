# Text Analysis — add-on do Orange

Add-on do [Orange Data Mining](https://orangedatamining.com) para limpar,
traduzir e classificar textos (ex.: análise de sentimento) com modelos do
[Hugging Face](https://huggingface.co). Os widgets aparecem na categoria
**Text Analysis** da barra lateral do Orange.

## Requisitos

- **Orange 3** (testado com a versão 3.40) e **Python 3.9 ou superior**
- **Orange3-Text** (opcional, recomendado): fornece o widget Corpus e os
  demais widgets de texto

Bibliotecas Python (instaladas automaticamente com o add-on):

| Biblioteca | Para que é usada |
|---|---|
| `transformers` | Carregar e executar os modelos do Hugging Face |
| `torch` | Motor de execução dos modelos (CPU ou GPU) |
| `sentencepiece` | Tokenizadores de modelos multilíngues (XLM-RoBERTa, T5, M2M100) |
| `protobuf` | Leitura dos arquivos de tokenizador SentencePiece |
| `ftfy` | Correção de codificação quebrada (ex.: `pÃ©ssimo` → `péssimo`) |
| `emoji` | Remoção de emojis e conversão em texto em português |

## Instalação

O add-on deve ser instalado **no mesmo Python que o Orange usa**.

**Pelo terminal**, dentro da pasta do projeto (troque `python` pelo
executável do Python do Orange, se for outro):

```bash
python -m pip install .
```

**Pelo próprio Orange**, colando no widget *Python Script* e clicando em
*Run* (ajuste o caminho da pasta):

```python
import sys, subprocess
pasta = "/caminho/para/orange3-text-analysis"
r = subprocess.run([sys.executable, "-m", "pip", "install", pasta],
                   capture_output=True, text=True)
print(r.stdout[-1500:]); print(r.stderr[-1500:])
```

Depois, **reinicie o Orange**.

Para desenvolvimento, instale em modo editável (as alterações no código
valem após reiniciar o Orange, sem reinstalar):

```bash
python -m pip install -e .
```

> **Atualizando a partir do `orange3-sentimento`:** desinstale o pacote
> antigo com `python -m pip uninstall orange3-sentimento`, senão os widgets
> aparecem duplicados. Modelos `.pkcls` salvos com o pacote antigo precisam
> ser gerados de novo.

## Widgets

Cada widget tem uma documentação própria na pasta [`doc/`](doc/), com
capturas de tela, fluxos de exemplo e (para os que usam modelos do
Hugging Face) os critérios para configurar um modelo no modo "Avançado".

### [Limpeza de Texto](doc/limpeza_texto.md)

Limpa o texto e cria uma nova coluna com o resultado, mostrando uma prévia
antes/depois. Opções: corrigir codificação quebrada, remover HTML, links,
e-mails e menções (remover ou trocar pelos marcadores `http`, `email` e
`@user`), hashtags, emojis (remover ou converter em texto), números,
stopwords, caracteres repetidos (`muuuuito` → `muuuito`), espaços
excessivos, minúsculas, acentos e pontuação, além de descartar textos
curtos e duplicados. É um pré-processamento de **texto corrido** (não
"palavra a palavra" como o Preprocess Text do Orange3-Text) — a forma
esperada pelos demais widgets deste add-on.

### [Tradução](doc/traducao.md)

Traduz textos com `unicamp-dl/translation-pt-en-t5` (português → inglês),
`unicamp-dl/translation-en-pt-t5` (inglês → português) ou
`facebook/m2m100_418M` (100 idiomas, com listas de idioma de origem e
destino). Pode criar uma nova coluna ou substituir o texto original. Divide
textos longos em frases, roda em segundo plano com barra de progresso e usa
a GPU, se houver.

### [Análise de Sentimento](doc/analise_sentimento.md)

Transforma um modelo de classificação de texto do Hugging Face (ex.:
`cardiffnlp/twitter-xlm-roberta-base-sentiment`) em um modelo do Orange,
compatível com *Predictions*, *Save Model* e *Load Model*. Lê os rótulos do
próprio modelo, aceita parâmetros avançados em JSON e, ao receber o corpus
rotulado na entrada, usa a variável-alvo dele para que o *Predictions*
calcule as métricas de avaliação.

### [Zero-shot Classification](doc/zeroshot.md)

Classifica textos por tópicos definidos pelo próprio usuário (ex.:
`esporte, política, tecnologia, saúde`), usando um modelo zero-shot
multilíngue (NLI) do Hugging Face — não precisa de dados rotulados nem de
um modelo já treinado para as classes desejadas. Também produz um modelo
do Orange, usado junto com *Predictions*.

### [Embeddings de Texto](doc/embeddings_texto.md)

Gera embeddings de sentença com um modelo do Hugging Face, prontos para uso
com o widget nativo *Distances* (ex.: para clusterização, mapas de
similaridade, projeções como t-SNE/MDS).

### [Extração de Tópicos (BERT)](doc/extracao_topicos_bert.md)

Descobre tópicos automaticamente a partir de embeddings de sentença do
Hugging Face, agrupados com o BERTopic (UMAP + HDBSCAN) — não é preciso
definir os tópicos antecipadamente. Mostra o resultado num gráfico de
dispersão interativo, com seleção de documentos por tópico.

## Fluxo de exemplo

```
Corpus → Limpeza de Texto ─┬─► Análise de Sentimento ──► Predictions
                           └──────────────────────────► Predictions
```

Para usar um modelo disponível só em inglês, inclua o widget **Tradução**
(português → inglês) depois da Limpeza de Texto. Veja mais exemplos de
fluxo na documentação de cada widget, em [`doc/`](doc/).

## Estrutura do projeto

```
orange3-text-analysis/
├── README.md
├── pyproject.toml
├── doc/                    # documentação de cada widget + imagens
├── tests/
└── text_analysis/
    ├── modelo.py           # adaptador Hugging Face → modelo do Orange
    ├── limpeza.py          # funções de limpeza de texto
    ├── traducao.py         # funções de tradução
    ├── embeddings.py       # geração de embeddings de sentença
    ├── topicos.py          # adaptador zero-shot (NLI) → modelo do Orange
    ├── topicos_bertopic.py # extração de tópicos com BERTopic
    ├── stopwords/          # listas de stopwords (português, inglês)
    └── widgets/
        ├── owlimpezatexto.py
        ├── owtraducao.py
        ├── owsentimentohf.py
        ├── owtopicos.py          # Zero-shot Classification
        ├── owembeddingstexto.py
        ├── owtopicosbertopic.py  # Extração de Tópicos (BERT)
        └── icons/
```

## Observações

- Cada modelo é baixado do Hugging Face no primeiro uso (de centenas de MB
  a cerca de 2 GB) e fica em cache em `~/.cache/huggingface`.
- Em sistemas com vírgula decimal (ex.: português do Brasil), o add-on
  ajusta temporariamente a formatação numérica ao carregar os modelos,
  evitando o erro do `protobuf` com tokenizadores SentencePiece.
