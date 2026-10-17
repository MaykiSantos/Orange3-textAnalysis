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

from text_analysis.modelo import (
    carregar_tokenizador, comprimento_maximo_seguro, eh_erro_cuda,
    liberar_memoria, numeros_formato_c)

# Tokens reservados para a hipótese (candidate label + modelo de frase) e os
# tokens especiais de par (ex.: [CLS]/[SEP][SEP]), já que o texto é
# tokenizado em par com cada hipótese.
MARGEM_HIPOTESE = 40

INFERENCIA_PADRAO = {"multi_label": False}


class TopicosZeroShot(Model):
    def __init__(self, nome_modelo, topicos, coluna_texto="text",
                 nome_tokenizador=None, kwargs_carregamento=None,
                 kwargs_inferencia=None, nome_alvo="topico",
                 manter_vivo=False):
        """
        manter_vivo: se True, o pipeline fica carregado em memória entre uma
        previsão e outra (mais rápido, usa mais memória); chame liberar()
        para soltá-lo manualmente. Se False (padrão), cada previsão carrega
        e libera o pipeline na hora.
        """
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
        self.manter_vivo = bool(manter_vivo)
        self.name = "Tópicos: " + nome_modelo.split("/")[-1]
        self._pipe = None  # só é preenchido quando manter_vivo=True

    # ---- carregamento ---------------------------------------------------
    def _carregar_pipeline(self, forcar_cpu=False):
        from transformers import pipeline
        kw = dict(self.kwargs_carregamento)
        if forcar_cpu:
            kw["device"] = "cpu"
        if "tokenizer" not in kw:
            kw["tokenizer"] = carregar_tokenizador(
                self.nome_tokenizador or self.nome_modelo,
                self.kwargs_carregamento)
        kw.pop("use_fast", None)
        with numeros_formato_c():
            return pipeline("zero-shot-classification",
                            model=self.nome_modelo, **kw)

    def _carregar_pipeline_com_fallback(self):
        try:
            return self._carregar_pipeline()
        except Exception as ex:
            if not eh_erro_cuda(ex) or "device" in self.kwargs_carregamento:
                raise
            return self._carregar_pipeline(forcar_cpu=True)

    def _pipeline(self):
        """Com manter_vivo=True, carrega o pipeline uma vez e o mantém em
        cache no objeto (reutilizado entre previsões); sem isso, cria um
        pipeline novo a cada chamada e quem chama deve soltar a referência e
        chamar liberar_memoria() assim que terminar (ver predict_storage),
        para o modelo não ficar alocado indefinidamente.
        """
        if self.manter_vivo:
            if self._pipe is None:
                self._pipe = self._carregar_pipeline_com_fallback()
            return self._pipe
        return self._carregar_pipeline_com_fallback()

    def liberar(self):
        """Libera o pipeline mantido em memória (modo manter_vivo), se houver."""
        if self._pipe is not None:
            self._pipe = None
            liberar_memoria()

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
        pipe = self._pipeline()
        # O pipeline zero-shot-classification não repassa max_length para o
        # tokenizador (só aceita truncation), então, se o
        # tokenizer.model_max_length do modelo estiver mal configurado no
        # Hub, truncation=True não trunca nada e um texto longo estoura a
        # tabela de posições do modelo. Como rede de segurança, truncamos o
        # texto manualmente antes de enviá-lo ao pipeline.
        limite = max(1, comprimento_maximo_seguro(pipe.tokenizer, pipe.model.config)
                     - MARGEM_HIPOTESE)
        textos = [
            pipe.tokenizer.decode(
                pipe.tokenizer.encode(t, truncation=True, max_length=limite,
                                      add_special_tokens=False),
                skip_special_tokens=True)
            for t in textos
        ]
        try:
            saidas = pipe(textos, candidate_labels=self.topicos, **kw)
        except Exception:
            if self.manter_vivo:
                # Não reaproveita um pipeline que pode ter deixado a GPU
                # num estado inválido; a próxima chamada carrega de novo.
                self._pipe = None
            raise
        finally:
            if not self.manter_vivo:
                del pipe
                liberar_memoria()
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
        estado["_pipe"] = None  # nunca serializa o pipeline carregado
        return estado

    def __setstate__(self, estado):
        estado.setdefault("manter_vivo", False)
        estado["_pipe"] = None
        self.__dict__.update(estado)
