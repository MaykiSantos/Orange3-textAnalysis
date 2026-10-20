import html
import json

import numpy as np
import pyqtgraph as pg
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QAbstractItemView, QListWidget, QListWidgetItem
from scipy.spatial import ConvexHull, QhullError
from Orange.data import (
    ContinuousVariable, DiscreteVariable, Domain, StringVariable, Table)
from Orange.data.util import get_unique_names
from Orange.widgets import gui
from Orange.widgets.settings import (
    ContextSetting, DomainContextHandler, Setting, SettingProvider)
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.plot import OWPlotGUI
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotBase
from Orange.widgets.widget import Input, Msg, OWWidget, Output

from text_analysis import embeddings as E
from text_analysis import topicos_bertopic as BT

PREDEFINIDO, AVANCADO = 0, 1

MODELOS_PREDEFINIDOS = [
    {
        "id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "rotulo": "MiniLM multilíngue — rápido (recomendado)",
        "descricao": "Modelo leve e rápido, bom equilíbrio entre qualidade e "
                     "velocidade; 384 dimensões, mais de 50 idiomas.",
    },
    {
        "id": "BAAI/bge-m3",
        "rotulo": "BGE-M3 (multilíngue, maior qualidade)",
        "descricao": "Modelo multilíngue de alta qualidade (mais de 100 "
                     "idiomas), com suporte a textos longos (até 8192 "
                     "tokens); 1024 dimensões, porém mais pesado e lento.",
    },
    {
        "id": "intfloat/multilingual-e5-large",
        "rotulo": "Multilingual E5-large (multilíngue, preciso)",
        "descricao": "Modelo multilíngue grande (100 idiomas), bom "
                     "equilíbrio entre qualidade e custo computacional; "
                     "1024 dimensões.",
    },
]


class OWTopicosGraph(OWScatterPlotBase):
    """Gráfico de dispersão com contorno e rótulo por tópico, no estilo do
    widget "Annotated Corpus Map" (orangecontrib.text)."""
    show_cluster_hull = Setting(True)

    def __init__(self, scatter_widget, parent):
        super().__init__(scatter_widget, parent)
        self.cluster_hulls_items = []
        self.cluster_labels_items = []

    def clear(self):
        super().clear()
        for item in self.cluster_hulls_items:
            self.plot_widget.removeItem(item)
        for item in self.cluster_labels_items:
            self.plot_widget.removeItem(item)
        self.cluster_hulls_items = []
        self.cluster_labels_items = []

    def update_coordinates(self):
        super().update_coordinates()
        self.update_clusters()

    def update_clusters(self):
        self._update_cluster_hull()
        self._update_cluster_labels()

    def _update_cluster_hull(self):
        for item in self.cluster_hulls_items:
            self.plot_widget.removeItem(item)
        self.cluster_hulls_items = []
        if not self.show_cluster_hull:
            return
        for contorno, cor in self.master.get_cluster_hulls():
            pen = pg.mkPen(color=cor, width=2, style=Qt.DashLine)
            x = np.hstack([contorno[:, 0], contorno[:1, 0]])
            y = np.hstack([contorno[:, 1], contorno[:1, 1]])
            item = pg.PlotCurveItem(x=x, y=y, pen=pen, antialias=True)
            item.setZValue(-10)  # atrás dos pontos
            self.plot_widget.addItem(item)
            self.cluster_hulls_items.append(item)

    def _update_cluster_labels(self):
        for item in self.cluster_labels_items:
            self.plot_widget.removeItem(item)
        self.cluster_labels_items = []
        for texto, (x, y), tooltip, cor in self.master.get_cluster_labels():
            item = pg.TextItem(
                html=('<div style="background-color: rgba(255, 255, 255, '
                     f'200); border: 1px solid {cor.name()}; padding: 2px '
                     f'4px; color: black; font-size: 9pt;">'
                     f'{html.escape(texto)}</div>'),
                anchor=(0.5, 0.5))
            item.textItem.setToolTip(html.escape(tooltip).replace("\n", "<br/>"))
            item.setPos(x, y)
            self.plot_widget.addItem(item)
            self.cluster_labels_items.append(item)


def tarefa(textos, nome_modelo, kw_carga, kw_bertopic, tamanho_lote,
          comprimento_maximo, min_topic_size, nr_topics, modelo_vivo,
          manter_vivo, state: TaskState):
    state.set_status("Baixando/carregando o modelo de embeddings...")

    def progresso(p):
        state.set_status("Extraindo tópicos...")
        state.set_progress_value(min(p, 90))

    modelo_carregado = modelo_vivo or E.carregar(nome_modelo, kw_carga)
    topicos, probs, info, embeddings = BT.extrair_topicos(
        textos, nome_modelo, kw_carga, kw_bertopic, tamanho_lote,
        comprimento_maximo, min_topic_size, nr_topics, progresso,
        state.is_interruption_requested, modelo_carregado=modelo_carregado,
        liberar_depois=not manter_vivo)

    state.set_status("Gerando projeção para o gráfico...")
    coords2d = BT.projetar_2d(embeddings)
    state.set_progress_value(100)
    return (topicos, probs, info, embeddings, coords2d,
           (modelo_carregado if manter_vivo else None))


class OWTopicosBERTopic(OWWidget, ConcurrentWidgetMixin):
    name = "Extração de Tópicos (BERT)"
    description = ("Descobre tópicos automaticamente a partir de embeddings "
                   "de sentença do Hugging Face, agrupados com o BERTopic — "
                   "não é preciso definir os tópicos antecipadamente.")
    icon = "icons/topicosbert.svg"
    priority = 20
    keywords = ["bertopic", "tópicos", "topicos", "topic modeling",
               "clusterização", "clustering", "embeddings", "hdbscan",
               "umap"]
    want_main_area = True

    settingsHandler = DomainContextHandler()
    coluna = ContextSetting(None)

    modo_modelo = Setting(PREDEFINIDO)
    modelo_idx = Setting(0)
    nome_modelo_avancado = Setting("")
    json_carregamento = Setting("")
    json_bertopic = Setting("")
    tamanho_lote = Setting(16)
    comprimento_maximo = Setting(0)
    min_topic_size = Setting(10)
    nr_topics = Setting(0)
    auto_gerar = Setting(False)
    manter_vivo = Setting(False)
    colorir_por_topico = Setting(True)

    graph = SettingProvider(OWTopicosGraph)

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)
        selecionados = Output("Selected Data", Table)
        topicos = Output("Topics", Table)

    class Error(OWWidget.Error):
        sem_texto = Msg("Os dados não têm nenhuma coluna de texto.")
        sem_nome_modelo = Msg("Informe o nome do modelo de embeddings "
                              "(Hugging Face).")
        json_invalido = Msg("Parâmetros de {}: JSON inválido ({}). "
                            "Exemplo de formato válido: \"min_topic_size\": "
                            "15 entre chaves.")
        poucos_documentos = Msg("É preciso de pelo menos 2 documentos para "
                                "extrair tópicos.")
        falha = Msg("Falha ao extrair tópicos: {}")

    class Information(OWWidget.Information):
        pendente = Msg("Configurações alteradas: clique em Extrair.")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.data = None
        self._chave_cache = None
        self._resultado = None
        self._chave_em_curso = None
        self._chave_modelo_em_curso = None
        self._modelo_vivo = None
        self._modelo_vivo_chave = None
        self._textos_em_curso = None

        # Estado do gráfico/seleção (ver métodos "gráfico" mais abaixo)
        self._saida_atual = None
        self._coords2d = None
        self._codigos_atual = None
        self._var_topico = None
        self._ids_ordenados = None
        self._info_atual = None

        box = gui.widgetBox(self.controlArea, "Coluna de texto")
        self.modelo_colunas = DomainModel(order=DomainModel.METAS,
                                          valid_types=(StringVariable,))
        gui.comboBox(box, self, "coluna", model=self.modelo_colunas,
                     callback=self._mudou_config)

        box = gui.widgetBox(self.controlArea, "Modelo de embeddings")
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
        e = gui.lineEdit(self.caixa_avancado, self, "json_bertopic",
                         label="Parâmetros do BERTopic:",
                         callback=self._mudou_config)
        e.setPlaceholderText('ex.: {"top_n_words": 15}')

        box = gui.widgetBox(self.controlArea, "Tópicos")
        gui.spin(box, self, "min_topic_size", 2, 1000,
                 label="Tamanho mínimo de um tópico (documentos):",
                 callback=self._mudou_config)
        gui.spin(box, self, "nr_topics", 0, 1000,
                 label="Número de tópicos a manter (0 = automático):",
                 callback=self._mudou_config)

        box = gui.widgetBox(self.controlArea, "Processamento")
        gui.spin(box, self, "comprimento_maximo", 0, 4096, step=16,
                 label="Tamanho máximo do texto (tokens; 0 = padrão do "
                       "modelo):",
                 callback=self._mudou_config)
        gui.spin(box, self, "tamanho_lote", 1, 128,
                 label="Textos por lote (maior = mais rápido, mais memória):")

        box = gui.widgetBox(self.controlArea, "Memória")
        gui.checkBox(box, self, "manter_vivo",
                     "Manter o modelo de embeddings carregado em memória "
                     "enquanto este widget existir",
                     callback=self._mudou_manter_vivo)
        lbl = gui.widgetLabel(
            box, "Desligado (padrão): cada extração carrega o modelo e o "
                "libera da memória em seguida — usa menos memória, porém é "
                "mais lento a cada uso. Ligado: o modelo fica carregado até "
                "este widget ser removido ou a opção ser desligada — mais "
                "rápido, mas usa mais memória.")
        lbl.setWordWrap(True)

        # ---- gráfico (ver protocolo de OWScatterPlotBase mais abaixo) ----
        caixa_grafico = gui.vBox(self.mainArea, True, margin=0)
        self.graph = OWTopicosGraph(self, caixa_grafico)
        caixa_grafico.layout().addWidget(self.graph.plot_widget)
        plot_gui = OWPlotGUI(self)
        plot_gui.box_zoom_select(self.buttonsArea)

        box = gui.widgetBox(self.controlArea, "Gráfico")
        gui.checkBox(box, self.graph, "show_legend", "Show legend",
                     callback=self._mudou_mostrar_legenda)
        gui.checkBox(box, self.graph, "show_cluster_hull",
                     "Show cluster hull",
                     callback=lambda: self.graph.update_clusters())
        gui.checkBox(box, self, "colorir_por_topico",
                     "Color point by topic",
                     callback=lambda: self.graph.update_colors())

        box = gui.widgetBox(self.controlArea, "Seleção por tópico")
        gui.widgetLabel(
            box, "Selecione um ou mais tópicos da lista e clique em "
                "\"Selecionar\" para passar todos os seus documentos "
                "adiante — ou selecione pontos direto no gráfico "
                "(retângulo/laço) para escolher documentos individuais."
        ).setWordWrap(True)
        self.lista_topicos = QListWidget()
        self.lista_topicos.setSelectionMode(QAbstractItemView.ExtendedSelection)
        box.layout().addWidget(self.lista_topicos)
        linha = gui.hBox(box)
        gui.button(linha, self, "Selecionar",
                  callback=self._selecionar_topicos)
        gui.button(linha, self, "Limpar seleção",
                  callback=self._limpar_selecao)

        gui.checkBox(self.buttonsArea, self, "auto_gerar",
                     "Extrair automaticamente", callback=self._mudou_config)
        self.botao = gui.button(self.buttonsArea, self, "Extrair",
                                callback=self.gerar)

        self.lbl_resumo = gui.widgetLabel(self.controlArea, "")
        self.lbl_resumo.setWordWrap(True)

        self._atualizar_estado_modelo()

    def onDeleteWidget(self):
        self.shutdown()
        self._liberar_modelo_vivo()
        self.graph.plot_widget.getViewBox().deleteLater()
        self.graph.plot_widget.clear()
        self.graph.clear()
        super().onDeleteWidget()

    def _liberar_modelo_vivo(self):
        if self._modelo_vivo is not None:
            self._modelo_vivo = None
            self._modelo_vivo_chave = None
            E.liberar_memoria()

    def _mudou_manter_vivo(self):
        if not self.manter_vivo:
            self._liberar_modelo_vivo()

    def _mudou_mostrar_legenda(self):
        self.graph.update_legend_visibility()

    @Inputs.data
    def set_data(self, data):
        self.cancel()
        self.closeContext()
        self.Error.clear()
        self.Information.clear()
        self.data = data
        self._chave_cache = self._resultado = None
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
        self._limpar_grafico()
        self.Outputs.data.send(None)
        self.Outputs.selecionados.send(None)
        self.Outputs.topicos.send(None)
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
                self.json_bertopic, self.comprimento_maximo,
                self.min_topic_size, self.nr_topics)

    def gerar(self):
        self.Error.clear()
        self.Information.pendente.clear()
        if self.data is None or self.coluna is None:
            return
        nome = self._nome_modelo()
        if not nome:
            self.Error.sem_nome_modelo()
            return
        if len(self.data) < 2:
            self.Error.poucos_documentos()
            return

        if self.modo_modelo == AVANCADO:
            kw_carga = self._ler_json(self.json_carregamento, "carregamento")
            kw_bertopic = self._ler_json(self.json_bertopic, "BERTopic")
            if kw_carga is None or kw_bertopic is None:
                return
        else:
            kw_carga, kw_bertopic = {}, {}

        chave = self._chave()
        if chave == self._chave_cache and self._resultado is not None:
            self._enviar(*self._resultado)
            return

        chave_modelo = (nome, json.dumps(kw_carga, sort_keys=True))
        if (self._modelo_vivo is not None
                and self._modelo_vivo_chave != chave_modelo):
            self._liberar_modelo_vivo()  # modelo mudou; solta o antigo antes
        modelo_para_reusar = (self._modelo_vivo if self.manter_vivo
                              and self._modelo_vivo_chave == chave_modelo
                              else None)

        self._chave_em_curso = chave
        self._chave_modelo_em_curso = chave_modelo
        self._textos_em_curso = self._textos()
        self.lbl_resumo.setText(
            f"Extraindo tópicos de {len(self.data)} documentos...")
        self.botao.setEnabled(False)
        self.start(tarefa, self._textos_em_curso, nome, kw_carga, kw_bertopic,
                   self.tamanho_lote, self.comprimento_maximo or None,
                   self.min_topic_size, self.nr_topics or None,
                   modelo_para_reusar, self.manter_vivo)

    def on_done(self, resultado):
        topicos, probs, info, embeddings, coords2d, modelo_vivo = resultado
        self.botao.setEnabled(True)
        self._chave_cache = self._chave_em_curso
        self._resultado = (topicos, probs, info, embeddings, coords2d)
        if self.manter_vivo:
            self._modelo_vivo = modelo_vivo
            self._modelo_vivo_chave = self._chave_modelo_em_curso
        self._enviar(topicos, probs, info, embeddings, coords2d)

    def on_exception(self, ex):
        self.botao.setEnabled(True)
        if isinstance(ex, BT.Cancelado):
            return
        # Não reaproveita um modelo que pode ter deixado a GPU num estado
        # inválido; a próxima extração carrega de novo.
        self._liberar_modelo_vivo()
        self.lbl_resumo.setText("A extração de tópicos falhou.")
        self.Error.falha(E.mensagem_erro(ex))

    def on_partial_result(self, _):
        pass

    @staticmethod
    def _rotulo_topico(id_topico):
        return "Sem tópico" if id_topico == -1 else f"Tópico {id_topico}"

    def _enviar(self, topicos, probs, info, embeddings, coords2d):
        info_por_id = {i["id"]: i for i in info}
        ids_ordenados = sorted(info_por_id)
        rotulos = [self._rotulo_topico(i) for i in ids_ordenados]
        indice = {i: j for j, i in enumerate(ids_ordenados)}
        codigos = np.array([indice[t] for t in topicos], dtype=float)

        dominio = self.data.domain
        dim = embeddings.shape[1]
        # Os embeddings (e não os atributos originais) viram os atributos
        # contínuos da saída, para a tabela poder ser ligada direto a um
        # widget de projeção (ex.: t-SNE), que calcula distâncias a partir
        # dos atributos.
        atributos = [ContinuousVariable(f"Dim{i + 1}") for i in range(dim)]
        var_topico = DiscreteVariable(get_unique_names(dominio, "Tópico"),
                                      values=rotulos)
        var_prob = ContinuousVariable(
            get_unique_names(dominio, "Tópico (confiança)"))
        novo_dominio = Domain(atributos, dominio.class_vars,
                              dominio.metas + (var_topico, var_prob))
        novas_metas = np.column_stack((self.data.metas, codigos, probs))
        saida = Table.from_numpy(novo_dominio, embeddings, self.data.Y,
                                 novas_metas)

        # Estado usado pelo gráfico e pela seleção (ver métodos "gráfico")
        self._saida_atual = saida
        self._coords2d = coords2d
        self._codigos_atual = codigos
        self._var_topico = var_topico
        self._ids_ordenados = ids_ordenados
        self._info_atual = info_por_id
        self._atualizar_lista_topicos(ids_ordenados, rotulos, info_por_id)
        self.graph.reset_graph()

        n_sem_topico = int(np.sum(topicos == -1))
        n_topicos = len(ids_ordenados) - (1 if -1 in info_por_id else 0)
        self.lbl_resumo.setText(
            f"{n_topicos} tópicos encontrados em {len(self.data)} "
            "documentos"
            + (f" ({n_sem_topico} sem tópico)" if n_sem_topico else "")
            + f". Modelo: '{self._nome_modelo()}'.")
        self.Outputs.data.send(saida)
        self.Outputs.topicos.send(self._tabela_topicos(info, ids_ordenados))
        self._enviar_selecionados()

    @staticmethod
    def _tabela_topicos(info, ids_ordenados):
        info_por_id = {i["id"]: i for i in info}
        dominio = Domain([ContinuousVariable("Documentos")],
                         metas=[StringVariable("Tópico"),
                               StringVariable("Palavras-chave")])
        linhas = [info_por_id[i] for i in ids_ordenados]
        X = np.array([[linha["quantidade"]] for linha in linhas], dtype=float)
        metas = np.array(
            [[OWTopicosBERTopic._rotulo_topico(linha["id"]),
             linha["palavras_chave"]] for linha in linhas], dtype=object)
        return Table.from_numpy(dominio, X, metas=metas)

    # ---- gráfico: lista de tópicos e seleção -----------------------------
    def _atualizar_lista_topicos(self, ids_ordenados, rotulos, info_por_id):
        self.lista_topicos.clear()
        for codigo, (topic_id, rotulo) in enumerate(zip(ids_ordenados,
                                                        rotulos)):
            qtd = info_por_id[topic_id]["quantidade"]
            item = QListWidgetItem(f"{rotulo} ({qtd} docs)")
            item.setData(Qt.UserRole, codigo)
            self.lista_topicos.addItem(item)

    def _limpar_grafico(self):
        self._saida_atual = None
        self._coords2d = None
        self._codigos_atual = None
        self._var_topico = None
        self._ids_ordenados = None
        self._info_atual = None
        self._textos_em_curso = None
        self.lista_topicos.clear()
        self.graph.selection = None
        self.graph.reset_graph()

    def _contorno_convexo(self, pontos):
        if len(pontos) < 2:
            return None
        if len(pontos) == 2:
            contorno = pontos
        else:
            try:
                contorno = pontos[ConvexHull(pontos).vertices]
            except QhullError:
                return None  # pontos colineares/degenerados: sem área
        return contorno

    def get_cluster_hulls(self):
        """Usado por OWTopicosGraph para desenhar o contorno de cada tópico."""
        if (self._coords2d is None or self._codigos_atual is None
                or self._var_topico is None or self._ids_ordenados is None):
            return []
        contornos = []
        for codigo, topic_id in enumerate(self._ids_ordenados):
            if topic_id == -1:
                continue  # "Sem tópico" é ruído, não um agrupamento de fato
            pontos = self._coords2d[self._codigos_atual == codigo]
            contorno = self._contorno_convexo(pontos)
            if contorno is None:
                continue
            cor = self._var_topico.palette.value_to_qcolor(codigo)
            contornos.append((contorno, cor))
        return contornos

    def get_cluster_labels(self):
        """Usado por OWTopicosGraph: rótulo curto sempre visível + descrição
        completa do tópico (palavras-chave) exibida ao passar o mouse."""
        if (self._coords2d is None or self._codigos_atual is None
                or self._var_topico is None or self._ids_ordenados is None):
            return []
        rotulos = []
        for codigo, topic_id in enumerate(self._ids_ordenados):
            if topic_id == -1:
                continue  # "Sem tópico" é ruído, não um agrupamento de fato
            pontos = self._coords2d[self._codigos_atual == codigo]
            if not len(pontos):
                continue
            cor = self._var_topico.palette.value_to_qcolor(codigo)
            rotulo = self._var_topico.values[codigo]
            info = self._info_atual.get(topic_id, {})
            descricao = info.get("palavras_chave", "")
            tooltip = f"{rotulo} ({info.get('quantidade', len(pontos))} docs)"
            if descricao:
                tooltip += f"\n{descricao}"
            cx, cy = pontos.mean(axis=0)
            rotulos.append((rotulo, (float(cx), float(cy)), tooltip, cor))
        return rotulos

    def _selecionar_topicos(self):
        if self._codigos_atual is None:
            return
        codigos_sel = {item.data(Qt.UserRole)
                       for item in self.lista_topicos.selectedItems()}
        if not codigos_sel:
            return
        self.graph.selection = np.isin(
            self._codigos_atual, list(codigos_sel)).astype(np.uint8)
        self.graph.update_selection_colors()
        self.selection_changed()

    def _limpar_selecao(self):
        self.lista_topicos.clearSelection()
        if self.graph.selection is not None:
            self.graph.selection = None
            self.graph.update_selection_colors()
            self.selection_changed()

    def _enviar_selecionados(self):
        selecao = self.graph.selection
        if (self._saida_atual is None or selecao is None
                or not np.any(selecao)):
            self.Outputs.selecionados.send(None)
            return
        self.Outputs.selecionados.send(self._saida_atual[selecao.astype(bool)])

    # ---- gráfico: protocolo esperado por OWScatterPlotBase ---------------
    # (self.graph chama estes métodos para obter o que desenhar; ver a
    # docstring de OWScatterPlotBase para a lista completa.)
    def get_coordinates_data(self):
        if self._coords2d is None or not len(self._coords2d):
            return None, None
        return (np.asarray(self._coords2d[:, 0], dtype=float),
               np.asarray(self._coords2d[:, 1], dtype=float))

    def get_size_data(self):
        return None

    def get_shape_data(self):
        return None

    def get_label_data(self):
        return None

    def get_shape_labels(self):
        return None

    def get_subset_mask(self):
        return None

    def is_continuous_color(self):
        return False

    def get_color_data(self):
        if not self.colorir_por_topico:
            return None
        return self._codigos_atual

    def get_color_labels(self):
        if not self.colorir_por_topico or self._var_topico is None:
            return None
        return list(self._var_topico.values)

    def get_palette(self):
        if not self.colorir_por_topico or self._var_topico is None:
            return None
        return self._var_topico.palette

    def get_tooltip(self, point_ids):
        if not point_ids or self._var_topico is None:
            return ""
        textos = self._textos_em_curso or []
        partes = []
        for i in list(point_ids)[:5]:
            i = int(i)
            rotulo = self._var_topico.values[int(self._codigos_atual[i])]
            texto = textos[i] if i < len(textos) else ""
            partes.append(f"<b>{rotulo}</b><br/>{texto[:200]}")
        return "<hr/>".join(partes)

    def selection_changed(self):
        self._enviar_selecionados()

    def send_report(self):
        if self.data is not None and self.coluna is not None:
            self.report_items((
                ("Coluna de texto", self.coluna.name),
                ("Modelo de embeddings", self._nome_modelo()),
            ))


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWTopicosBERTopic).run()
