"""Descoberta de tópicos com BERTopic, a partir de embeddings de sentença
gerados com um modelo do Hugging Face (reaproveita text_analysis.embeddings).
"""
import numpy as np

from text_analysis import embeddings as E
from text_analysis.modelo import liberar_memoria, numeros_formato_c

Cancelado = E.Cancelado


def extrair_topicos(textos, nome_modelo, kwargs_carregamento=None,
                    kwargs_bertopic=None, tamanho_lote=16,
                    comprimento_maximo=None, min_topic_size=10,
                    nr_topics=None, progresso=None, cancelado=None,
                    modelo_carregado=None, liberar_depois=True):
    """Agrupa os textos em tópicos com o BERTopic.

    Os embeddings são gerados antes (reaproveitando text_analysis.embeddings,
    com a mesma barra de progresso e cancelamento) e repassados prontos ao
    BERTopic, que assim só cuida do agrupamento (UMAP + HDBSCAN) e da
    extração das palavras-chave de cada tópico (c-TF-IDF) — o modelo de
    linguagem não é carregado nem executado duas vezes.

    Retorna (topicos, probabilidades, info_topicos, embeddings):
    - topicos: array com o id do tópico de cada texto (-1 = sem tópico).
    - probabilidades: confiança da atribuição de cada texto ao seu tópico.
    - info_topicos: lista de dicts (id, quantidade, palavras_chave), um por
      tópico (incluindo o -1, quando há textos sem tópico).
    - embeddings: vetor de sentença de cada texto (array (n, dim)), para uso
      em widgets de projeção/visualização (ex.: t-SNE).
    """
    modelo = modelo_carregado or E.carregar(nome_modelo, kwargs_carregamento)
    modelo_bertopic = None
    try:
        def progresso_embeddings(p):
            if progresso is not None:
                progresso(p * 0.7)

        vetores = E.gerar_embeddings(
            textos, nome_modelo, kwargs_carregamento, None, tamanho_lote,
            comprimento_maximo, True, progresso_embeddings, cancelado,
            modelo_carregado=modelo, liberar_depois=False)
        if cancelado is not None and cancelado():
            raise Cancelado()
        if progresso is not None:
            progresso(75)

        from bertopic import BERTopic
        kw = dict(kwargs_bertopic or {})
        kw.setdefault("min_topic_size", min_topic_size)
        if nr_topics:
            kw["nr_topics"] = nr_topics
        with numeros_formato_c():
            modelo_bertopic = BERTopic(embedding_model=modelo, **kw)
            topicos, probs = modelo_bertopic.fit_transform(textos,
                                                            embeddings=vetores)

        topicos = np.asarray(topicos)
        if probs is None:
            probs = np.ones(len(topicos))
        else:
            probs = np.asarray(probs)
            if probs.ndim > 1:
                probs = probs.max(axis=1)

        info = modelo_bertopic.get_topic_info()
        info_topicos = [
            {
                "id": int(linha["Topic"]),
                "quantidade": int(linha["Count"]),
                "palavras_chave": ", ".join(
                    palavra for palavra, _ in
                    modelo_bertopic.get_topic(linha["Topic"]) or []),
            }
            for _, linha in info.iterrows()
        ]
        if progresso is not None:
            progresso(100)
        return topicos, probs, info_topicos, vetores
    finally:
        del modelo_bertopic  # solta a referência ao modelo de embeddings
        if liberar_depois:
            del modelo
            liberar_memoria()


def projetar_2d(embeddings, semente=42):
    """Reduz os embeddings a 2 dimensões (UMAP), só para visualização.

    É uma redução independente da usada internamente pelo BERTopic para o
    agrupamento (que normalmente fica em mais de 2 dimensões, para não
    perder separação entre tópicos).
    """
    n = len(embeddings)
    if n < 4:
        # UMAP exige n_neighbors < n_amostras; com poucos documentos a
        # projeção não ajudaria a visualizar agrupamentos de qualquer forma.
        return np.zeros((n, 2))
    from umap import UMAP
    n_neighbors = max(2, min(15, n - 1))
    return UMAP(n_components=2, n_neighbors=n_neighbors,
               random_state=semente).fit_transform(embeddings)
