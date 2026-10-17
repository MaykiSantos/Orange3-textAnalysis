"""Adaptador que transforma um modelo zero-shot NLI do Hugging Face em um
modelo de classificação do Orange, com os tópicos (classes) definidos pelo
usuário em vez de lidos do modelo.

Como esta classe vive dentro de um pacote instalado, qualquer .pkcls que a
contenha pode ser aberto pelo widget nativo "Load Model".
"""
import numpy as np
from Orange.base import Model
from Orange.data import Domain, DiscreteVariable, StringVariable
from Orange.data.table import DomainTransformationError

from text_analysis.modelo import carregar_tokenizador, numeros_formato_c

INFERENCIA_PADRAO = {"multi_label": False}


class TopicosZeroShot(Model):
    def __init__(self, nome_modelo, topicos, coluna_texto="text",
                 nome_tokenizador=None, kwargs_carregamento=None,
                 kwargs_inferencia=None, nome_alvo="topico"):
        topicos = [str(t) for t in topicos]
        if len(topicos) < 2:
            raise ValueError("Informe ao menos dois tópicos candidatos.")
        alvo = DiscreteVariable(nome_alvo or "topico", values=topicos)
        domain = Domain([], class_vars=alvo,
                        metas=[StringVariable(coluna_texto)])
        super().__init__(domain)
        self.nome_modelo = nome_modelo
        self.topicos = topicos
        self.coluna_texto = coluna_texto
        self.nome_tokenizador = nome_tokenizador or None
        self.kwargs_carregamento = dict(kwargs_carregamento or {})
        self.kwargs_inferencia = dict(kwargs_inferencia or {})
        self.name = "Tópicos: " + nome_modelo.split("/")[-1]
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
                self._pipe = pipeline("zero-shot-classification",
                                      model=self.nome_modelo, **kw)
        return self._pipe

    # ---- dados ----------------------------------------------------------
    def get_backmappers(self, data):
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
        if hasattr(data, "get_column"):
            valores = data.get_column(var)
        else:
            valores = data.get_column_view(var)[0]
        return ["" if v is None else str(v) for v in valores]

    # ---- previsão -------------------------------------------------------
    def predict_storage(self, data):
        textos = self._textos(data)
        kw = dict(INFERENCIA_PADRAO)
        kw.update(self.kwargs_inferencia)
        saidas = self._pipeline()(textos, candidate_labels=self.topicos, **kw)
        if isinstance(saidas, dict):
            saidas = [saidas]

        indice = {t: j for j, t in enumerate(self.topicos)}
        probs = np.zeros((len(textos), len(self.topicos)))
        for i, saida in enumerate(saidas):
            for rotulo, score in zip(saida["labels"], saida["scores"]):
                j = indice.get(rotulo)
                if j is not None:
                    probs[i, j] = score
        return probs  # o Orange deriva a classe prevista das probabilidades

    # ---- pickle ---------------------------------------------------------
    def __getstate__(self):
        estado = self.__dict__.copy()
        estado["_pipe"] = None
        return estado

    def __setstate__(self, estado):
        estado["_pipe"] = None
        self.__dict__.update(estado)
