import json
import re

from Orange.base import Model
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.widget import OWWidget, Input, Output, Msg

from text_analysis.modelo import (
    SentimentoHF, ler_config, carregar_tokenizador, numeros_formato_c,
    INFERENCIA_PADRAO, ErroTokenizador)

TEXTO_TESTE = "Este é um texto de teste para validar a inferência do modelo."

PREDEFINIDO, AVANCADO = 0, 1

MODELOS_PREDEFINIDOS = [
    {
        "id": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "rotulo": "CardiffNLP — Twitter XLM-RoBERTa (multilíngue)",
        "descricao": "Modelo multilíngue (8 idiomas, incluindo português) "
                     "treinado em tweets; classifica em negativo, neutro ou "
                     "positivo.",
    },
    {
        "id": "nlptown/bert-base-multilingual-uncased-sentiment",
        "rotulo": "NLPTown — BERT multilíngue (avaliações de produtos)",
        "descricao": "Modelo multilíngue (inglês, holandês, alemão, francês, "
                     "italiano e espanhol) treinado em avaliações de "
                     "produtos; classifica de 1 a 5 estrelas.",
    },
    {
        "id": "pysentimiento/bertweet-pt-sentiment",
        "rotulo": "pysentimiento — BERTweet em português",
        "descricao": "Modelo em português treinado em tweets (biblioteca "
                     "pysentimiento); classifica em positivo, neutro ou "
                     "negativo.",
    },
]


class ErroSemRotulos(Exception):
    pass


class ErroCarregamentoPesos(Exception):
    pass


class ErroInferenciaTeste(Exception):
    pass


def _tarefa(nome, nome_tokenizador, kw_carga, kw_inf, state: TaskState):
    """Baixa o modelo (config, tokenizador e pesos) e faz uma inferência de
    teste, reportando o progresso de cada etapa em `state`."""
    state.set_status("Baixando a configuração do modelo...")
    state.set_progress_value(5)
    rotulos, arquiteturas = ler_config(nome, kw_carga)
    if len(rotulos) < 2:
        raise ErroSemRotulos
    state.set_progress_value(20)

    state.set_status("Baixando o tokenizador...")
    tokenizador = carregar_tokenizador(nome_tokenizador or nome, kw_carga)
    state.set_progress_value(45)

    state.set_status("Baixando os pesos do modelo...")
    try:
        from transformers import pipeline
        kw = dict(kw_carga)
        if "tokenizer" not in kw:
            kw["tokenizer"] = tokenizador
        kw.pop("use_fast", None)
        with numeros_formato_c():
            pipe = pipeline("text-classification", model=nome, **kw)
    except Exception as ex:
        raise ErroCarregamentoPesos(str(ex)) from ex
    state.set_progress_value(80)

    state.set_status("Fazendo uma inferência de teste...")
    try:
        kw_teste = dict(INFERENCIA_PADRAO)
        kw_teste.update(kw_inf)
        kw_teste["top_k"] = None
        pipe([TEXTO_TESTE], **kw_teste)
    except Exception as ex:
        raise ErroInferenciaTeste(str(ex)) from ex
    state.set_progress_value(100)

    return rotulos, arquiteturas, pipe


class OWSentimentoHF(OWWidget, ConcurrentWidgetMixin):
    name = "Análise de Sentimento"
    description = ("Carrega um modelo de classificação de sentimento"
                   "a partir de um modelo do Hugging Face.")
    icon = "icons/sentimento.svg"
    priority = 10
    keywords = ["sentiment", "huggingface", "transformers", "sentimento"]
    want_main_area = False
    resizing_enabled = False

    modo_modelo = Setting(PREDEFINIDO)
    modelo_idx = Setting(0)
    nome_modelo = Setting("cardiffnlp/twitter-xlm-roberta-base-sentiment")
    nome_tokenizador = Setting("")
    nomes_classes = Setting("")
    nome_alvo = Setting("sentimento")
    coluna_texto = Setting("text")
    json_carregamento = Setting("")
    json_inferencia = Setting("")

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        model = Output("Model", Model)

    class Error(OWWidget.Error):
        sem_nome = Msg("Informe o nome do modelo e da coluna de texto.")
        json_invalido = Msg("Parâmetros de {}: JSON inválido ({}). "
                            "Exemplo de formato válido: \"max_length\": 128 "
                            "entre chaves.")
        config = Msg("Não foi possível ler o modelo: {}")
        carregamento = Msg("Não foi possível carregar os pesos do modelo: {}")
        inferencia_teste = Msg("Falha ao testar a inferência do modelo: {}")
        sem_rotulos = Msg("O modelo não informa rótulos de classe (id2label); "
                          "ele não parece ser um classificador.")
        nomes_classes = Msg("Informe {} nomes de classe (um por rótulo do "
                            "modelo, na mesma ordem) ou deixe o campo vazio.")
        nomes_repetidos = Msg("Os nomes de classe não podem se repetir.")
        tokenizador = Msg("{}")
        alvo_numerico = Msg("O alvo do corpus ('{}') é numérico. Converta-o em "
                            "categórico (ex.: com o Edit Domain).")
        sem_correspondencia = Msg("Nenhuma classe do modelo ({}) corresponde aos "
                                  "valores do alvo do corpus ({}). Use o campo "
                                  "Renomear classes.")

    class Warning(OWWidget.Warning):
        rotulos_genericos = Msg("O modelo usa rótulos genéricos ({}). Consulte "
                                "o model card e dê nomes às classes.")
        nao_classificacao = Msg("Arquitetura '{}' não é de classificação de "
                                "sequência; o resultado pode não fazer sentido.")
        classes_faltando = Msg("Classes do modelo sem correspondência no alvo do "
                               "corpus: {}. A probabilidade delas será "
                               "redistribuída entre as demais.")
        top_k = Msg("O parâmetro top_k é ignorado: o add-on sempre usa todas "
                    "as classes para calcular probabilidades.")

    class Information(OWWidget.Information):
        pendente = Msg("Há alterações não aplicadas: clique em Aplicar para "
                       "carregar/atualizar o modelo.")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.data = None
        self._chave_cache = None
        self._cache_resultado = None
        self._contexto = None

        box = gui.widgetBox(self.controlArea, "Modelo")
        rb = gui.radioButtons(box, self, "modo_modelo",
                              callback=self._atualizar_estado_modelo)
        gui.appendRadioButton(rb, "Modelo pré-definido")
        self.caixa_lista = gui.indentedBox(rb)
        gui.comboBox(self.caixa_lista, self, "modelo_idx",
                     items=[m["rotulo"] for m in MODELOS_PREDEFINIDOS],
                     callback=self._atualizar_estado_modelo)
        self.lbl_descricao_modelo = gui.widgetLabel(self.caixa_lista, "")
        self.lbl_descricao_modelo.setWordWrap(True)

        gui.appendRadioButton(rb, "Avançado")
        self.caixa_avancado = gui.indentedBox(rb)
        a = gui.lineEdit(self.caixa_avancado, self, "nome_modelo",
                     label="Nome no Hugging Face:",
                     callback=self._marcar_pendente)
        a.setPlaceholderText("ex.: cardiffnlp/twitter-xlm-roberta-base-sentiment")
        e = gui.lineEdit(self.caixa_avancado, self, "nome_tokenizador",
                         label="Tokenizador (opcional; vazio = o do modelo):",
                         callback=self._marcar_pendente)
        e.setPlaceholderText("ex.: FacebookAI/xlm-roberta-base")
        e = gui.lineEdit(self.caixa_avancado, self, "json_carregamento",
                         label="Carregamento (from_pretrained / pipeline):",
                         callback=self._marcar_pendente)
        e.setPlaceholderText('ex.: {"revision": "main", "device": 0}')
        e = gui.lineEdit(self.caixa_avancado, self, "json_inferencia",
                         label="Inferência (chamada do pipeline):",
                         callback=self._marcar_pendente)
        e.setPlaceholderText('ex.: {"max_length": 128, "batch_size": 8}')

        box = gui.widgetBox(self.controlArea, "Classes")
        self.lbl_rotulos = gui.widgetLabel(
            box, "Rótulos do modelo: (clique em Aplicar)")
        self.lbl_rotulos.setWordWrap(True)
        e = gui.lineEdit(box, self, "nomes_classes",
                         label="Renomear classes (opcional, mesma ordem):",
                         callback=self._marcar_pendente)
        e.setPlaceholderText("ex.: negativo, neutro, positivo")
        self.campo_alvo = gui.lineEdit(
            box, self, "nome_alvo",
            label="Nome da variável alvo (sem corpus conectado):",
            callback=self._marcar_pendente)
        self.lbl_alvo = gui.widgetLabel(
            box, "Dica: conecte o corpus rotulado à entrada deste widget para "
                 "usar o alvo dele automaticamente.")
        self.lbl_alvo.setWordWrap(True)

        box = gui.widgetBox(self.controlArea, "Texto")
        gui.lineEdit(box, self, "coluna_texto", label="Coluna de texto do corpus:",
                     callback=self._marcar_pendente)

        self.botao = gui.button(self.controlArea, self, "Aplicar",
                                callback=self.commit)
        self._atualizar_estado_modelo()
        self._marcar_pendente()

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    @Inputs.data
    def set_data(self, data):
        self.data = data
        self._alvo_do_corpus()
        self._marcar_pendente()

    def _marcar_pendente(self):
        self.Information.pendente()

    def _atualizar_estado_modelo(self):
        predefinido = self.modo_modelo == PREDEFINIDO
        self.caixa_lista.setVisible(predefinido)
        self.caixa_avancado.setVisible(not predefinido)
        self.lbl_descricao_modelo.setText(
            MODELOS_PREDEFINIDOS[self.modelo_idx]["descricao"])
        if self.controlArea.layout() is not None:
            self.controlArea.layout().activate()
        self.adjustSize()
        self._marcar_pendente()

    def _modelo_e_tokenizador(self):
        if self.modo_modelo == PREDEFINIDO:
            return MODELOS_PREDEFINIDOS[self.modelo_idx]["id"], None
        return self.nome_modelo.strip(), self.nome_tokenizador.strip() or None

    def _alvo_do_corpus(self):
        """Variável-alvo do corpus conectado (ou None)."""
        alvo = self.data.domain.class_var if self.data is not None else None
        tem = alvo is not None
        self.campo_alvo.setEnabled(not tem)
        if tem:
            self.lbl_alvo.setText(f"Usando o alvo do corpus: '{alvo.name}'"
                                  + (f" ({', '.join(alvo.values)})"
                                     if alvo.is_discrete else ""))
        else:
            self.lbl_alvo.setText(
                "Dica: conecte o corpus rotulado à entrada deste widget para "
                "usar o alvo dele automaticamente.")
        return alvo

    def _ler_json(self, texto, onde):
        texto = (texto or "").strip()
        if not texto:
            return {}
        try:
            valor = json.loads(texto)
        except ValueError as ex:
            self.Error.json_invalido(onde, ex)
            return None
        if not isinstance(valor, dict):
            self.Error.json_invalido(onde, "use um objeto entre chaves")
            return None
        return valor

    def commit(self):
        self.Error.clear()
        self.Warning.clear()
        self.Information.pendente.clear()
        self.cancel()
        self.Outputs.model.send(None)

        nome, nome_tokenizador = self._modelo_e_tokenizador()
        coluna = self.coluna_texto.strip()
        if not nome or not coluna:
            self.Error.sem_nome()
            return

        if self.modo_modelo == AVANCADO:
            kw_carga = self._ler_json(self.json_carregamento, "carregamento")
            kw_inf = self._ler_json(self.json_inferencia, "inferência")
            if kw_carga is None or kw_inf is None:
                return
        else:
            kw_carga, kw_inf = {}, {}
        if "top_k" in kw_inf:
            kw_inf.pop("top_k")
            self.Warning.top_k()

        self._contexto = dict(nome=nome, coluna=coluna, kw_carga=kw_carga,
                              kw_inf=kw_inf, nome_tokenizador=nome_tokenizador)

        chave = (nome, nome_tokenizador, json.dumps(kw_carga, sort_keys=True),
                 json.dumps(kw_inf, sort_keys=True))
        if chave == self._chave_cache and self._cache_resultado is not None:
            self._finalizar(self._cache_resultado)
            return

        self.lbl_rotulos.setText("Rótulos do modelo: (baixando...)")
        self._chave_cache = chave
        self.botao.setEnabled(False)
        self.start(_tarefa, nome, nome_tokenizador, kw_carga, kw_inf)

    def on_done(self, resultado):
        self.botao.setEnabled(True)
        self._cache_resultado = resultado
        self._finalizar(resultado)

    def on_exception(self, ex):
        self.botao.setEnabled(True)
        self._chave_cache = None
        self.lbl_rotulos.setText("Rótulos do modelo: —")
        if isinstance(ex, ErroSemRotulos):
            self.Error.sem_rotulos()
        elif isinstance(ex, ErroTokenizador):
            self.Error.tokenizador(ex)
        elif isinstance(ex, ErroCarregamentoPesos):
            self.Error.carregamento(ex)
        elif isinstance(ex, ErroInferenciaTeste):
            self.Error.inferencia_teste(ex)
        else:
            self.Error.config(ex)

    def on_partial_result(self, _):
        pass

    def _finalizar(self, resultado):
        rotulos, arquiteturas, pipe = resultado
        ctx = self._contexto
        self.lbl_rotulos.setText("Rótulos do modelo: " + ", ".join(rotulos))

        if arquiteturas and not any("SequenceClassification" in a
                                    for a in arquiteturas):
            self.Warning.nao_classificacao(", ".join(arquiteturas))

        nomes = [c.strip() for c in self.nomes_classes.split(",") if c.strip()]
        if nomes and len(nomes) != len(rotulos):
            self.Error.nomes_classes(len(rotulos))
            return
        if len(set(nomes)) != len(nomes):
            self.Error.nomes_repetidos()
            return
        if not nomes and all(re.fullmatch(r"LABEL_\d+", r) for r in rotulos):
            self.Warning.rotulos_genericos(", ".join(rotulos))

        alvo = self._alvo_do_corpus()
        if alvo is not None:
            if not alvo.is_discrete:
                self.Error.alvo_numerico(alvo.name)
                return
            efetivos = nomes or rotulos
            faltando = [n for n in efetivos if n not in alvo.values]
            if len(faltando) == len(efetivos):
                self.Error.sem_correspondencia(", ".join(efetivos),
                                               ", ".join(alvo.values))
                return
            if faltando:
                self.Warning.classes_faltando(", ".join(faltando))

        modelo = SentimentoHF(
            ctx["nome"], rotulos, coluna_texto=ctx["coluna"],
            nomes_classes=nomes or None,
            nome_tokenizador=ctx["nome_tokenizador"],
            kwargs_carregamento=ctx["kw_carga"], kwargs_inferencia=ctx["kw_inf"],
            nome_alvo=self.nome_alvo.strip() or "sentimento",
            variavel_alvo=alvo)
        modelo._pipe = pipe  # já carregado e testado; evita baixar de novo
        self.Outputs.model.send(modelo)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWSentimentoHF).run()
