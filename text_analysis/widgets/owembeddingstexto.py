import json

from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler, Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import Input, Msg, OWWidget, Output

from text_analysis import embeddings as E

PREDEFINIDO, AVANCADO = 0, 1

MODELOS_PREDEFINIDOS = [
    {
        "id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "rotulo": "MiniLM multilíngue — rápido (384 dimensões)",
        "descricao": "Modelo leve e rápido, bom equilíbrio entre qualidade e "
                     "velocidade; 384 dimensões, mais de 50 idiomas.",
    },
    {
        "id": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "rotulo": "MPNet multilíngue — mais preciso (768 dimensões)",
        "descricao": "Mais preciso que o MiniLM, porém mais lento; 768 "
                     "dimensões, mais de 50 idiomas.",
    },
    {
        "id": "sentence-transformers/LaBSE",
        "rotulo": "LaBSE (109 idiomas)",
        "descricao": "Modelo do Google voltado para comparar frases entre "
                     "idiomas diferentes; 768 dimensões, 109 idiomas.",
    },
    {
        "id": "Qwen/Qwen3-Embedding-0.6B",
        "rotulo": "Qwen3 Embedding 0.6B — maior qualidade (mais pesado)",
        "descricao": "Modelo grande baseado em LLM (0.6B parâmetros); maior "
                     "qualidade, porém bem mais lento e pesado para baixar "
                     "e processar; 1024 dimensões.",
    },
]


def tarefa(textos, nome_modelo, kw_carga, kw_inf, tamanho_lote,
          comprimento_maximo, normalizar, state: TaskState):
    state.set_status("Baixando/carregando o modelo...")
    def progresso(p):
        state.set_status("Gerando embeddings...")
        state.set_progress_value(p)
    return E.gerar_embeddings(textos, nome_modelo, kw_carga, kw_inf,
                              tamanho_lote, comprimento_maximo, normalizar,
                              progresso, state.is_interruption_requested)


class OWEmbeddingsTexto(OWWidget, ConcurrentWidgetMixin):
    name = "Embeddings de Texto"
    description = ("Gera embeddings de sentença com um modelo do Hugging "
                   "Face, prontos para uso com o widget Distances (ex.: "
                   "para clusterização).")
    icon = "icons/embeddings.svg"
    priority = 15
    keywords = ["embedding", "embeddings", "clusterização", "clustering",
               "sentence-transformers", "distances"]
    want_main_area = False
    resizing_enabled = False

    settingsHandler = DomainContextHandler()
    coluna = ContextSetting(None)

    modo_modelo = Setting(PREDEFINIDO)
    modelo_idx = Setting(0)
    nome_modelo_avancado = Setting("")
    json_carregamento = Setting("")
    json_inferencia = Setting("")
    tamanho_lote = Setting(16)
    comprimento_maximo = Setting(0)
    normalizar = Setting(True)
    auto_gerar = Setting(False)

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Embeddings", Table)

    class Error(OWWidget.Error):
        sem_texto = Msg("Os dados não têm nenhuma coluna de texto.")
        sem_nome_modelo = Msg("Informe o nome do modelo (Hugging Face).")
        json_invalido = Msg("Parâmetros de {}: JSON inválido ({}). "
                            "Exemplo de formato válido: \"device\": \"cpu\" "
                            "entre chaves.")
        falha = Msg("Falha ao gerar embeddings: {}")

    class Information(OWWidget.Information):
        pendente = Msg("Configurações alteradas: clique em Gerar.")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.data = None
        self._chave_cache = None
        self._embeddings = None
        self._chave_em_curso = None

        box = gui.widgetBox(self.controlArea, "Coluna de texto")
        self.modelo_colunas = DomainModel(order=DomainModel.METAS,
                                          valid_types=(StringVariable,))
        gui.comboBox(box, self, "coluna", model=self.modelo_colunas,
                     callback=self._mudou_config)

        box = gui.widgetBox(self.controlArea, "Modelo")
        rb = gui.radioButtons(box, self, "modo_modelo",
                              callback=self._mudou_modo_modelo)
        gui.appendRadioButton(rb, "Modelo pré-definido")
        self.caixa_lista = gui.indentedBox(rb)
        gui.comboBox(self.caixa_lista, self, "modelo_idx",
                     items=[m["rotulo"] for m in MODELOS_PREDEFINIDOS],
                     callback=self._mudou_modo_modelo)
        self.lbl_descricao_modelo = gui.widgetLabel(self.caixa_lista, "")
        self.lbl_descricao_modelo.setWordWrap(True)

        gui.appendRadioButton(rb, "Avançado")
        self.caixa_avancado = gui.indentedBox(rb)
        e = gui.lineEdit(self.caixa_avancado, self, "nome_modelo_avancado",
                         label="Nome no Hugging Face:",
                         callback=self._mudou_config)
        e.setPlaceholderText(
            "ex.: sentence-transformers/all-MiniLM-L6-v2")
        e = gui.lineEdit(self.caixa_avancado, self, "json_carregamento",
                         label="Carregamento (SentenceTransformer):",
                         callback=self._mudou_config)
        e.setPlaceholderText(
            'ex.: {"trust_remote_code": true, "revision": "main"}')
        e = gui.lineEdit(self.caixa_avancado, self, "json_inferencia",
                         label="Inferência (encode):",
                         callback=self._mudou_config)
        e.setPlaceholderText('ex.: {"prompt_name": "query"}')

        box = gui.widgetBox(self.controlArea, "Processamento")
        gui.checkBox(box, self, "normalizar",
                     "Normalizar vetores (recomendado para distância do "
                     "cosseno)", callback=self._mudou_config)
        e = gui.spin(box, self, "comprimento_maximo", 0, 4096, step=16,
                     label="Tamanho máximo do texto (tokens; 0 = padrão do "
                           "modelo):",
                     callback=self._mudou_config)
        gui.spin(box, self, "tamanho_lote", 1, 128,
                 label="Textos por lote (maior = mais rápido, mais memória):")

        gui.checkBox(self.buttonsArea, self, "auto_gerar",
                     "Gerar automaticamente", callback=self._mudou_config)
        self.botao = gui.button(self.buttonsArea, self, "Gerar",
                                callback=self.gerar)

        self.lbl_resumo = gui.widgetLabel(self.controlArea, "")
        self.lbl_resumo.setWordWrap(True)

        self._atualizar_estado_modelo()

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    @Inputs.data
    def set_data(self, data):
        self.cancel()
        self.closeContext()
        self.Error.clear()
        self.Information.clear()
        self.data = data
        self._chave_cache = self._embeddings = None
        self.modelo_colunas.set_domain(data.domain if data else None)
        self.coluna = None
        self.lbl_resumo.setText("")
        if data is not None:
            if not len(self.modelo_colunas):
                self.Error.sem_texto()
                self.data = None
            else:
                padrao = list(getattr(data, "text_features", None) or [])
                self.coluna = (padrao[0] if padrao and
                               padrao[0] in self.modelo_colunas
                               else self.modelo_colunas[0])
                self.openContext(data)
        self.Outputs.data.send(None)
        if self.data is not None:
            if self.auto_gerar:
                self.gerar()
            else:
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

    def _mudou_modo_modelo(self):
        self._atualizar_estado_modelo()
        self._mudou_config()

    def _nome_modelo(self):
        if self.modo_modelo == PREDEFINIDO:
            return MODELOS_PREDEFINIDOS[self.modelo_idx]["id"]
        return self.nome_modelo_avancado.strip()

    def _mudou_config(self):
        if self.data is None:
            return
        if self.auto_gerar:
            self.gerar()
        elif self._chave() != self._chave_cache:
            self.Information.pendente()

    def _textos(self):
        if hasattr(self.data, "get_column"):
            valores = self.data.get_column(self.coluna)
        else:
            valores = self.data.get_column_view(self.coluna)[0]
        return ["" if v is None else str(v) for v in valores]

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

    def _chave(self):
        return (id(self.data), self.coluna and self.coluna.name,
                self._nome_modelo(), self.json_carregamento,
                self.json_inferencia, self.comprimento_maximo,
                self.normalizar)

    def gerar(self):
        self.Error.clear()
        self.Information.pendente.clear()
        if self.data is None or self.coluna is None:
            return
        nome = self._nome_modelo()
        if not nome:
            self.Error.sem_nome_modelo()
            return

        if self.modo_modelo == AVANCADO:
            kw_carga = self._ler_json(self.json_carregamento, "carregamento")
            kw_inf = self._ler_json(self.json_inferencia, "inferência")
            if kw_carga is None or kw_inf is None:
                return
        else:
            kw_carga, kw_inf = {}, {}

        chave = self._chave()
        if chave == self._chave_cache and self._embeddings is not None:
            self._enviar(self._embeddings)
            return
        self._chave_em_curso = chave
        self.lbl_resumo.setText(
            f"Gerando embeddings de {len(self.data)} documentos...")
        self.botao.setEnabled(False)
        self.start(tarefa, self._textos(), nome, kw_carga, kw_inf,
                   self.tamanho_lote, self.comprimento_maximo or None,
                   self.normalizar)

    def on_done(self, embeddings):
        self.botao.setEnabled(True)
        self._chave_cache = self._chave_em_curso
        self._embeddings = embeddings
        self._enviar(embeddings)

    def on_exception(self, ex):
        self.botao.setEnabled(True)
        if isinstance(ex, E.Cancelado):
            return
        self.lbl_resumo.setText("A geração de embeddings falhou.")
        self.Error.falha(ex)

    def on_partial_result(self, _):
        pass

    def _enviar(self, embeddings):
        dominio = self.data.domain
        dim = embeddings.shape[1]
        atributos = [ContinuousVariable(f"Dim{i + 1}") for i in range(dim)]
        novo_dominio = Domain(atributos, dominio.class_vars, dominio.metas)
        saida = Table.from_numpy(novo_dominio, embeddings, self.data.Y,
                                 self.data.metas)
        self.lbl_resumo.setText(
            f"{len(saida)} documentos, {dim} dimensões, modelo "
            f"'{self._nome_modelo()}'.")
        self.Outputs.data.send(saida)

    def send_report(self):
        self.report_items((
            ("Modelo", self._nome_modelo()),
            ("Dimensões por texto",
             self._embeddings.shape[1] if self._embeddings is not None
             else "—"),
        ))


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWEmbeddingsTexto).run()
