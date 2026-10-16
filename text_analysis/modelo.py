"""Adaptador que transforma um modelo de classificação de texto do
Hugging Face em um modelo do Orange.

Como esta classe vive dentro de um pacote instalado, qualquer .pkcls que a
contenha pode ser aberto pelo widget nativo "Load Model".
"""
import locale
import logging
import re
from contextlib import contextmanager

import numpy as np
from Orange.base import Model
from Orange.data import Domain, DiscreteVariable, StringVariable
from Orange.data.table import DomainTransformationError

# Parâmetros de carregamento que também valem para ler o config do modelo
CHAVES_CONFIG = ("revision", "token", "trust_remote_code", "subfolder",
                 "cache_dir", "local_files_only")

# Valores padrão da inferência; o usuário pode sobrescrever (exceto top_k)
INFERENCIA_PADRAO = {"truncation": True, "batch_size": 16}

RE_USUARIO = re.compile(r"@\w+")
RE_URL = re.compile(r"(https?://|www\.)\S+")


def limpar(texto):
    """Troca menções por @user e links por http (formato usado em tweets)."""
    texto = RE_URL.sub("http", texto)
    return RE_USUARIO.sub("@user", texto)


def ler_config(nome_modelo, kwargs_carregamento=None):
    """Lê do config do modelo os rótulos (id2label, em ordem) e a arquitetura.

    Baixa só o config.json, não os pesos.
    """
    from transformers import AutoConfig
    kw = {k: v for k, v in (kwargs_carregamento or {}).items()
          if k in CHAVES_CONFIG}
    cfg = AutoConfig.from_pretrained(nome_modelo, **kw)
    id2label = getattr(cfg, "id2label", None) or {}
    rotulos = [str(id2label[i]) for i in sorted(id2label, key=int)]
    arquiteturas = list(getattr(cfg, "architectures", None) or [])
    return rotulos, arquiteturas


@contextmanager
def numeros_formato_c():
    """Usa ponto decimal enquanto o modelo/tokenizador é carregado.

    Em sistemas com locale de vírgula decimal (ex.: pt_BR), o Orange ativa o
    locale do sistema e o protobuf falha ao ler valores como 0.9995 da
    definição do SentencePiece ("Invalid default '0.9995' ...").
    """
    try:
        anterior = locale.setlocale(locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")
    except locale.Error:
        anterior = None
    try:
        yield
    finally:
        if anterior is not None:
            try:
                locale.setlocale(locale.LC_NUMERIC, anterior)
            except locale.Error:
                pass


class _ColetorAvisos(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.mensagens = []

    def emit(self, record):
        self.mensagens.append(record.getMessage())


class ErroTokenizador(RuntimeError):
    pass


def carregar_tokenizador(nome, kwargs_carregamento=None):
    """Carrega o tokenizador e, se falhar, explica a causa real.

    O transformers esconde a causa original em um aviso de log (ex.: falta do
    sentencepiece) e mostra só o erro do plano B (TikToken), que é enganoso.
    """
    from transformers import AutoTokenizer
    kw = {k: v for k, v in (kwargs_carregamento or {}).items()
          if k in CHAVES_CONFIG + ("use_fast",)}
    coletor = _ColetorAvisos()
    log = logging.getLogger("transformers")
    log.addHandler(coletor)
    try:
        with numeros_formato_c():
            return AutoTokenizer.from_pretrained(nome, **kw)
    except Exception as ex:
        avisos = " ".join(coletor.mensagens)
        try:
            from transformers.utils import is_sentencepiece_available
            tem_sp = is_sentencepiece_available()
        except ImportError:
            tem_sp = True
        if not tem_sp and ("SentencePiece" in avisos
                           or "parsing line" in str(ex)):
            msg = (f"O tokenizador de '{nome}' precisa da biblioteca "
                   "'sentencepiece', que não está disponível no Python do "
                   "Orange. Instale 'sentencepiece' e 'protobuf' e reinicie "
                   "o Orange, ou preencha o campo Tokenizador com um "
                   "equivalente que tenha tokenizer.json (para modelos "
                   "XLM-RoBERTa: FacebookAI/xlm-roberta-base).")
        else:
            msg = f"Falha ao carregar o tokenizador de '{nome}': {ex}"
            if avisos:
                msg += f" | Avisos do transformers: {avisos[:600]}"
        raise ErroTokenizador(msg) from ex
    finally:
        log.removeHandler(coletor)


class SentimentoHF(Model):
    def __init__(self, nome_modelo, rotulos_modelo, coluna_texto="text",
                 nomes_classes=None, nome_tokenizador=None,
                 kwargs_carregamento=None, kwargs_inferencia=None,
                 limpar_texto=False, nome_alvo="sentimento",
                 variavel_alvo=None):
        """
        variavel_alvo: a variável-alvo (categórica) do corpus rotulado. Se for
        dada, o modelo usa exatamente essa variável, para o Orange reconhecer
        que ambos falam do mesmo alvo (o Orange compara nome E origem). Cada
        rótulo do modelo é ligado ao valor do alvo de mesmo nome, depois de
        aplicado nomes_classes.
        """
        rotulos_modelo = [str(r) for r in rotulos_modelo]
        nomes = list(nomes_classes) if nomes_classes else rotulos_modelo
        if len(nomes) != len(rotulos_modelo):
            raise ValueError("O número de nomes de classe deve ser igual ao "
                             "número de rótulos do modelo.")
        if variavel_alvo is not None:
            alvo = variavel_alvo
            valores = list(alvo.values)
            self._indices_alvo = [valores.index(n) if n in valores else -1
                                  for n in nomes]
        else:
            alvo = DiscreteVariable(nome_alvo or "sentimento", values=nomes)
            self._indices_alvo = list(range(len(nomes)))
        domain = Domain([], class_vars=alvo,
                        metas=[StringVariable(coluna_texto)])
        super().__init__(domain)
        self.nome_modelo = nome_modelo
        self.rotulos_modelo = rotulos_modelo
        self.coluna_texto = coluna_texto
        self.nome_tokenizador = nome_tokenizador or None
        self.kwargs_carregamento = dict(kwargs_carregamento or {})
        self.kwargs_inferencia = dict(kwargs_inferencia or {})
        self.limpar_texto = bool(limpar_texto)
        self.name = "HF: " + nome_modelo.split("/")[-1]
        self._pipe = None  # carregado sob demanda; não vai para o pickle

    # ---- carregamento ---------------------------------------------------
    def _pipeline(self):
        if self._pipe is None:
            from transformers import pipeline
            kw = dict(self.kwargs_carregamento)
            if "tokenizer" not in kw:
                kw["tokenizer"] = carregar_tokenizador(
                    self.nome_tokenizador or self.nome_modelo,
                    self.kwargs_carregamento)
            kw.pop("use_fast", None)
            with numeros_formato_c():
                self._pipe = pipeline("text-classification",
                                      model=self.nome_modelo, **kw)
        return self._pipe

    # ---- dados ----------------------------------------------------------
    def get_backmappers(self, data):
        # Se o corpus tem uma variável-alvo com o mesmo nome da do modelo, o
        # Orange alinha as classes pelo nome dos valores (permite avaliar).
        # Se o alvo for outro, apenas prevê, em vez de falhar.
        try:
            return super().get_backmappers(data)
        except DomainTransformationError:
            return None, []

    def data_to_model_domain(self, data, *args, **kwargs):
        # O texto é lido direto da coluna original (ver _textos); a conversão
        # de domínio padrão do Orange falharia, pois o modelo não tem
        # colunas numéricas.
        return data

    def _textos(self, data):
        nomes = [v.name for v in data.domain.variables + data.domain.metas]
        if self.coluna_texto not in nomes:
            raise ValueError(
                f"Coluna de texto '{self.coluna_texto}' não encontrada. "
                f"Colunas disponíveis: {', '.join(nomes)}")
        var = data.domain[self.coluna_texto]
        if hasattr(data, "get_column"):          # Orange >= 3.34
            valores = data.get_column(var)
        else:
            valores = data.get_column_view(var)[0]
        textos = ["" if v is None else str(v) for v in valores]
        if self.limpar_texto:
            textos = [limpar(t) for t in textos]
        return textos

    # ---- previsão -------------------------------------------------------
    def predict_storage(self, data):
        textos = self._textos(data)
        kw = dict(INFERENCIA_PADRAO)
        kw.update(self.kwargs_inferencia)
        kw["top_k"] = None  # sempre todas as classes, para obter probabilidades
        saidas = self._pipeline()(textos, **kw)

        indice = {r.lower(): j for j, r in enumerate(self.rotulos_modelo)}
        n_classes = len(self.domain.class_var.values)
        probs = np.zeros((len(textos), n_classes))
        for i, scores in enumerate(saidas):
            if isinstance(scores, dict):
                scores = [scores]
            for s in scores:
                j = indice.get(str(s["label"]).lower())
                if j is not None and self._indices_alvo[j] >= 0:
                    probs[i, self._indices_alvo[j]] += s["score"]
        # Se algum rótulo do modelo não existe no alvo do corpus, redistribui
        # a probabilidade entre as classes que existem.
        somas = probs.sum(axis=1, keepdims=True)
        np.divide(probs, somas, out=probs, where=somas > 0)
        return probs  # o Orange deriva a classe prevista das probabilidades

    # ---- pickle ---------------------------------------------------------
    def __getstate__(self):
        estado = self.__dict__.copy()
        estado["_pipe"] = None
        return estado

    def __setstate__(self, estado):
        # Compatibilidade com .pkcls gerados pelas versões 0.1.x
        estado.setdefault("rotulos_modelo", estado.get("classes", []))
        estado.setdefault("nome_tokenizador", None)
        estado.setdefault("kwargs_carregamento", {})
        estado.setdefault("kwargs_inferencia", {})
        estado.setdefault("limpar_texto", False)
        estado.setdefault("_indices_alvo",
                          list(range(len(estado["rotulos_modelo"]))))
        estado["_pipe"] = None
        self.__dict__.update(estado)
