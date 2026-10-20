<p align="center">
  <img src="https://github.com/MaykiSantos/Orange3-textAnalysis/blob/main/doc/images/text-analysis-logo.png?raw=true" width="500" alt="Text Analysis">
</p>

Add-on do [Orange Data Mining](https://orangedatamining.com) para limpar,
traduzir e classificar textos (ex.: análise de sentimento, zero-shot,
extração de tópicos) com modelos do [Hugging Face](https://huggingface.co).
Os widgets aparecem na categoria **Text Analysis** da barra lateral do
Orange.

## Widgets disponíveis

![Limpeza de Texto, Tradução, Análise de Sentimento, Zero-shot Classification, Embeddings de Texto e Extração de Tópicos (BERT)](https://github.com/MaykiSantos/Orange3-textAnalysis/blob/main/doc/images/widgets_disponiveis.png?raw=true)

Cada widget tem uma documentação própria na pasta [`doc/`](doc/), com
capturas de tela, fluxos de exemplo e (para os que usam modelos do
Hugging Face) os critérios para configurar um modelo no modo "Avançado".

### <img src="https://raw.githubusercontent.com/MaykiSantos/Orange3-textAnalysis/cb6612d38868a4244e4626dc78b17072cefc119a/text_analysis/widgets/icons/limpeza.svg" width="24" height="24" valign="middle"> [Limpeza de Texto](https://github.com/MaykiSantos/Orange3-textAnalysis/blob/cb6612d38868a4244e4626dc78b17072cefc119a/doc/limpeza_texto.md)

Limpa o texto e cria uma nova coluna com o resultado, mostrando uma prévia
antes/depois. Opções: corrigir codificação quebrada, remover HTML, links,
e-mails e menções (remover ou trocar pelos marcadores `http`, `email` e
`@user`), hashtags, emojis (remover ou converter em texto), números,
stopwords, caracteres repetidos (`muuuuito` → `muuuito`), espaços
excessivos, minúsculas, acentos e pontuação, além de descartar textos
curtos e duplicados. É um pré-processamento de **texto corrido** (não
"palavra a palavra" como o Preprocess Text do Orange3-Text) — a forma
esperada pelos demais widgets deste add-on.

### <img src="https://raw.githubusercontent.com/MaykiSantos/Orange3-textAnalysis/cb6612d38868a4244e4626dc78b17072cefc119a/text_analysis/widgets/icons/traducao.svg" width="24" height="24" valign="middle"> [Tradução](https://github.com/MaykiSantos/Orange3-textAnalysis/blob/cb6612d38868a4244e4626dc78b17072cefc119a/doc/traducao.md)

Traduz textos com `unicamp-dl/translation-pt-en-t5` (português → inglês),
`unicamp-dl/translation-en-pt-t5` (inglês → português) ou
`facebook/m2m100_418M` (100 idiomas, com listas de idioma de origem e
destino). Pode criar uma nova coluna ou substituir o texto original. Divide
textos longos em frases, roda em segundo plano com barra de progresso e usa
a GPU, se houver.

### <img src="https://raw.githubusercontent.com/MaykiSantos/Orange3-textAnalysis/cb6612d38868a4244e4626dc78b17072cefc119a/text_analysis/widgets/icons/sentimento.svg" width="24" height="24" valign="middle"> [Análise de Sentimento](https://github.com/MaykiSantos/Orange3-textAnalysis/blob/cb6612d38868a4244e4626dc78b17072cefc119a/doc/analise_sentimento.md)

Transforma um modelo de classificação de texto do Hugging Face (ex.:
`cardiffnlp/twitter-xlm-roberta-base-sentiment`) em um modelo do Orange,
compatível com *Predictions*, *Save Model* e *Load Model*. Lê os rótulos do
próprio modelo, aceita parâmetros avançados em JSON e, ao receber o corpus
rotulado na entrada, usa a variável-alvo dele para que o *Predictions*
calcule as métricas de avaliação.

### <img src="https://raw.githubusercontent.com/MaykiSantos/Orange3-textAnalysis/cb6612d38868a4244e4626dc78b17072cefc119a/text_analysis/widgets/icons/zeroshot.svg" width="24" height="24" valign="middle"> [Zero-shot Classification](https://github.com/MaykiSantos/Orange3-textAnalysis/blob/cb6612d38868a4244e4626dc78b17072cefc119a/doc/zeroshot.md)

Classifica textos por tópicos definidos pelo próprio usuário (ex.:
`esporte, política, tecnologia, saúde`), usando um modelo zero-shot
multilíngue (NLI) do Hugging Face — não precisa de dados rotulados nem de
um modelo já treinado para as classes desejadas. Também produz um modelo
do Orange, usado junto com *Predictions*.

### <img src="https://raw.githubusercontent.com/MaykiSantos/Orange3-textAnalysis/cb6612d38868a4244e4626dc78b17072cefc119a/text_analysis/widgets/icons/embeddings.svg" width="24" height="24" valign="middle"> [Embeddings de Texto](https://github.com/MaykiSantos/Orange3-textAnalysis/blob/cb6612d38868a4244e4626dc78b17072cefc119a/doc/embeddings_texto.md)

Gera embeddings de sentença com um modelo do Hugging Face, prontos para uso
com o widget nativo *Distances* (ex.: para clusterização, mapas de
similaridade, projeções como t-SNE/MDS).

### <img src="https://raw.githubusercontent.com/MaykiSantos/Orange3-textAnalysis/cb6612d38868a4244e4626dc78b17072cefc119a/text_analysis/widgets/icons/topicosbert.svg" width="24" height="24" valign="middle"> [Extração de Tópicos (BERT)](https://github.com/MaykiSantos/Orange3-textAnalysis/blob/cb6612d38868a4244e4626dc78b17072cefc119a/doc/extracao_topicos_bert.md)

Descobre tópicos automaticamente a partir de embeddings de sentença do
Hugging Face, agrupados com o BERTopic (UMAP + HDBSCAN) — não é preciso
definir os tópicos antecipadamente. Mostra o resultado num gráfico de
dispersão interativo, com seleção de documentos por tópico.

## Requisitos

- **Orange 3** (testado com a versão 3.40) e **Python 3.12 ou superior**
- **Orange3-Text** (opcional, recomendado): fornece o widget Corpus e os
  demais widgets de texto

## Instalação

1. Na interface do Orange, acesse o menu superior e clique em
   **Options → Add-ons...**.
2. Na nova tela de instalação de complementos, clique em **Add more...**
   e digite `orange3-text-analysis`.
3. Marque a caixa do add-on na lista e clique em **OK** para instalar.
4. **Reinicie o Orange**.


## Observações

- Cada modelo é baixado do Hugging Face no primeiro uso (de centenas de MB
  a cerca de 2 GB) e fica em cache em `~/.cache/huggingface`.
- Em sistemas com vírgula decimal (ex.: português do Brasil), o add-on
  ajusta temporariamente a formatação numérica ao carregar os modelos,
  evitando o erro do `protobuf` com tokenizadores SentencePiece.

---

Para a estrutura do código e instruções de instalação em modo
desenvolvimento, veja [`doc/desenvolvimento.md`](https://github.com/MaykiSantos/Orange3-textAnalysis/blob/cb6612d38868a4244e4626dc78b17072cefc119a/doc/desenvolvimento.md).
