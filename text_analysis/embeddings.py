"""Embeddings de sentença com modelos do Hugging Face.

Usa a biblioteca sentence-transformers em vez de implementar o pooling à
mão: modelos diferentes usam estratégias de pooling bem diferentes (mean,
cls+dense, último token, com ou sem normalização) e a biblioteca já lê essa
configuração do próprio repositório do modelo no Hub.
"""
import json

import numpy as np

from text_analysis.modelo import numeros_formato_c

_CACHE = {}   # modelos já carregados nesta sessão do Orange


class Cancelado(Exception):
    pass


def carregar(nome_modelo, kwargs_carregamento=None):
    """Carrega (uma vez por sessão) o modelo; usa GPU se houver."""
    kwargs_carregamento = dict(kwargs_carregamento or {})
    chave = (nome_modelo, json.dumps(kwargs_carregamento, sort_keys=True))
    if chave not in _CACHE:
        from sentence_transformers import SentenceTransformer
        with numeros_formato_c():
            try:
                modelo = SentenceTransformer(nome_modelo, **kwargs_carregamento)
            except Exception:
                if "device" in kwargs_carregamento:
                    raise  # usuário escolheu o dispositivo; não mascara o erro
                modelo = SentenceTransformer(
                    nome_modelo, device="cpu", **kwargs_carregamento)
        _CACHE[chave] = modelo
    return _CACHE[chave]


def gerar_embeddings(textos, nome_modelo, kwargs_carregamento=None,
                     kwargs_inferencia=None, tamanho_lote=16,
                     comprimento_maximo=None, normalizar=True,
                     progresso=None, cancelado=None):
    """Gera um vetor de embedding por texto. Retorna um array (n, dim)."""
    modelo = carregar(nome_modelo, kwargs_carregamento)
    if comprimento_maximo:
        modelo.max_seq_length = comprimento_maximo

    kw = dict(kwargs_inferencia or {})
    kw.setdefault("normalize_embeddings", normalizar)
    kw.setdefault("show_progress_bar", False)
    kw.setdefault("convert_to_numpy", True)

    vetores = []
    n = len(textos)
    for ini in range(0, n, tamanho_lote):
        if cancelado is not None and cancelado():
            raise Cancelado()
        lote = textos[ini:ini + tamanho_lote]
        vetores.append(modelo.encode(lote, **kw))
        if progresso is not None:
            progresso(100 * min(ini + tamanho_lote, n) / max(n, 1))

    if hasattr(modelo, "get_embedding_dimension"):
        dim = modelo.get_embedding_dimension()
    else:
        dim = modelo.get_sentence_embedding_dimension()
    return np.vstack(vetores) if vetores else np.zeros((0, dim), dtype=float)
