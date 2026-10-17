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


def comprimento_maximo_seguro(tokenizer, config, margem=2):
    """Comprimento máximo de sequência (em tokens) seguro para o modelo.

    O `tokenizer.model_max_length` de vários modelos publicados no Hub não
    é configurado corretamente (fica com um valor-sentinela enorme, tipo
    "sem limite"). Confiar só em `truncation=True` nesse caso não trunca
    nada, e um texto longo estoura `max_position_embeddings` do modelo com
    um "index out of bounds" — que, quando roda na GPU, costuma aparecer
    como um confuso "device-side assert triggered" em vez do erro real.
    Por isso sempre limitamos pelo tamanho real do modelo também.
    """
    candidatos = []
    max_pos = getattr(config, "max_position_embeddings", None)
    if max_pos:
        candidatos.append(max(1, max_pos - margem))
    limite_tokenizer = getattr(tokenizer, "model_max_length", None)
    if limite_tokenizer and limite_tokenizer < 1_000_000:
        candidatos.append(limite_tokenizer)
    return min(candidatos) if candidatos else 512


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


def eh_erro_cuda(ex):
    """Indica se a exceção parece vir de um erro de CUDA/GPU.

    Um erro desses (ex.: "device-side assert triggered", "out of memory")
    pode deixar o contexto da GPU num estado inválido pelo resto do
    processo: até operações não relacionadas voltam a falhar depois disso,
    e só reiniciar o Orange resolve de vez. Por isso, ao detectar um,
    tentamos recarregar o modelo na CPU em vez de insistir na GPU.
    """
    return "cuda" in str(ex).lower()


def mensagem_erro(ex):
    """Formata uma exceção para exibir ao usuário, acrescentando uma dica
    quando parece ser um erro de CUDA/GPU (ver eh_erro_cuda)."""
    texto = str(ex)
    if eh_erro_cuda(ex):
        texto += (" | Isso parece um erro de GPU/CUDA: depois que ocorre, a "
                  "GPU pode ficar num estado inválido pelo resto desta "
                  "sessão do Orange. Tentamos recarregar na CPU "
                  "automaticamente; se o erro persistir mesmo trocando de "
                  "modelo, reinicie o Orange.")
    return texto


def liberar_memoria():
    """Força a liberação de memória do modelo carregado mais recentemente.

    Chame logo depois de soltar a última referência ao pipeline/modelo
    (ex.: `del pipe`), para que a memória (RAM e, se houver, VRAM) seja
    devolvida de imediato, em vez de ficar alocada até o coletor de lixo do
    Python rodar por conta própria. Sem isso, cada inferência deixaria o
    modelo (de centenas de MB a alguns GB) alocado mesmo depois do widget
    que o criou ser removido do workflow.
    """
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # Uma GPU com o contexto CUDA corrompido (erro "sticky", ex.: um
        # "device-side assert" anterior) pode falhar até numa chamada de
        # limpeza como empty_cache(). Esta função só faz uma otimização de
        # limpeza; nunca deve propagar erro — principalmente porque é
        # chamada ao remover um widget (onDeleteWidget), e uma exceção ali
        # aparece como um crash do canvas do Orange.
        pass


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
                 variavel_alvo=None, manter_vivo=False):
        """
        variavel_alvo: a variável-alvo (categórica) do corpus rotulado. Se for
        dada, o modelo usa exatamente essa variável, para o Orange reconhecer
        que ambos falam do mesmo alvo (o Orange compara nome E origem). Cada
        rótulo do modelo é ligado ao valor do alvo de mesmo nome, depois de
        aplicado nomes_classes.

        manter_vivo: se True, o pipeline fica carregado em memória entre uma
        previsão e outra (mais rápido, usa mais memória); chame liberar()
        para soltá-lo manualmente. Se False (padrão), cada previsão carrega
        e libera o pipeline na hora.
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
        self.manter_vivo = bool(manter_vivo)
        self.name = "HF: " + nome_modelo.split("/")[-1]
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
            return pipeline("text-classification",
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
        pipe = self._pipeline()
        kw.setdefault("max_length",
                      comprimento_maximo_seguro(pipe.tokenizer, pipe.model.config))
        try:
            saidas = pipe(textos, **kw)
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
        estado["_pipe"] = None  # nunca serializa o pipeline carregado
        return estado

    def __setstate__(self, estado):
        # Compatibilidade com .pkcls gerados pelas versões 0.1.x
        estado.setdefault("rotulos_modelo", estado.get("classes", []))
        estado.setdefault("nome_tokenizador", None)
        estado.setdefault("kwargs_carregamento", {})
        estado.setdefault("kwargs_inferencia", {})
        estado.setdefault("limpar_texto", False)
        estado.setdefault("manter_vivo", False)
        estado.setdefault("_indices_alvo",
                          list(range(len(estado["rotulos_modelo"]))))
        estado["_pipe"] = None
        self.__dict__.update(estado)
