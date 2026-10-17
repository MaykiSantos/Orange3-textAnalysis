import re

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView

from Orange.data import Domain, StringVariable, Table
from Orange.data.util import get_unique_names
from Orange.widgets import gui
from Orange.widgets.settings import (
    Setting, ContextSetting, DomainContextHandler)
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import OWWidget, Input, Output, Msg

from text_analysis import traducao as T

N_PREVIA = 20
NOVA_COLUNA, SUBSTITUIR = 0, 1
PREDEFINIDO, AVANCADO = 0, 1
TIPO_T5, TIPO_M2M100 = 0, 1

# Lista do combo de idiomas: "Português (pt)", ordenada pelo nome
ITENS_IDIOMAS = sorted(f"{nome} ({cod})"
                       for cod, nome in T.IDIOMAS_M2M100.items())


def codigo(item):
    return re.search(r"\((\w+)\)$", item).group(1)


def item_idioma(cod):
    return f"{T.IDIOMAS_M2M100[cod]} ({cod})"


def tarefa(textos, cfg, origem, destino, lote, beams, max_tokens,
           state: TaskState):
    state.set_status("Carregando o modelo...")
    def progresso(p):
        state.set_status("Traduzindo...")
        state.set_progress_value(p)
    return T.traduzir(textos, cfg, origem, destino, lote, beams,
                      max_tokens, progresso, state.is_interruption_requested)


class OWTraducao(OWWidget, ConcurrentWidgetMixin):
    name = "Tradução"
    description = "Traduz textos com modelos do Hugging Face."
    icon = "icons/traducao.svg"
    priority = 7
    keywords = ["tradução", "traduzir", "translate", "translation"]

    settingsHandler = DomainContextHandler()
    coluna = ContextSetting(None)

    modo_modelo = Setting(PREDEFINIDO)
    modelo_idx = Setting(0)
    nome_modelo_avancado = Setting("")
    tipo_modelo_avancado = Setting(TIPO_T5)
    prefixo_avancado = Setting("translate Portuguese to English: ")
    idioma_origem = Setting(item_idioma("pt"))
    idioma_destino = Setting(item_idioma("en"))
    num_beams = Setting(1)
    max_tokens = Setting(256)
    tamanho_lote = Setting(8)
    modo_saida = Setting(NOVA_COLUNA)
    nome_saida = Setting("texto_traduzido")
    usar_como_texto = Setting(True)
    auto_traduzir = Setting(False)

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    class Error(OWWidget.Error):
        sem_texto = Msg("Os dados não têm nenhuma coluna de texto.")
        falha = Msg("Falha na tradução: {}")
        mesmo_idioma = Msg("Os idiomas de origem e destino são iguais.")
        sem_nome_modelo = Msg("Informe o nome do modelo (Hugging Face).")

    class Information(OWWidget.Information):
        pendente = Msg("Configurações alteradas: clique em Traduzir.")
        nome_alterado = Msg("Já existe uma coluna '{}'; a nova coluna se "
                            "chama '{}'.")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.data = None
        self._chave_cache = None
        self._traducoes = None
        self._chave_em_curso = None

        box = gui.widgetBox(self.controlArea, "Coluna de texto")
        self.modelo_colunas = DomainModel(order=DomainModel.METAS,
                                          valid_types=(StringVariable,))
        gui.comboBox(box, self, "coluna", model=self.modelo_colunas,
                     callback=self._mudou_traducao)

        box = gui.widgetBox(self.controlArea, "Modelo")
        rb_modelo = gui.radioButtons(box, self, "modo_modelo",
                                     callback=self._mudou_modelo)
        gui.appendRadioButton(rb_modelo, "Modelo pré-definido")
        self.caixa_lista = gui.indentedBox(rb_modelo)
        gui.comboBox(self.caixa_lista, self, "modelo_idx",
                     items=[m["rotulo"] for m in T.MODELOS],
                     callback=self._mudou_modelo)

        gui.appendRadioButton(rb_modelo, "Avançado")
        self.caixa_avancado = gui.indentedBox(rb_modelo)
        gui.lineEdit(self.caixa_avancado, self, "nome_modelo_avancado",
                     label="Nome no Hugging Face:",
                     callback=self._mudou_modelo)
        rb_tipo = gui.radioButtons(self.caixa_avancado, self,
                                   "tipo_modelo_avancado",
                                   label="Tipo de modelo:",
                                   callback=self._mudou_modelo)
        gui.appendRadioButton(
            rb_tipo, "T5 / texto-para-texto (usa um prefixo de instrução)")
        self.caixa_prefixo = gui.indentedBox(rb_tipo)
        gui.lineEdit(self.caixa_prefixo, self, "prefixo_avancado",
                     label="Prefixo:", callback=self._mudou_modelo)
        gui.appendRadioButton(
            rb_tipo, "M2M100 (usa os idiomas selecionados abaixo)")

        self.box_idiomas = gui.widgetBox(self.controlArea, "Idiomas")
        for attr, rotulo in (("idioma_origem", "De:"),
                             ("idioma_destino", "Para:")):
            gui.comboBox(self.box_idiomas, self, attr, label=rotulo,
                         items=ITENS_IDIOMAS, sendSelectedValue=True,
                         orientation=Qt.Horizontal, searchable=True,
                         callback=self._mudou_traducao)

        box = gui.widgetBox(self.controlArea, "Geração")
        gui.spin(box, self, "num_beams", 1, 8,
                 label="Qualidade (num_beams; maior = melhor e mais lento):",
                 callback=self._mudou_traducao)
        gui.spin(box, self, "max_tokens", 16, 1024, step=16,
                 label="Tamanho máximo de cada trecho traduzido (tokens):",
                 callback=self._mudou_traducao)
        gui.spin(box, self, "tamanho_lote", 1, 64,
                 label="Textos por lote (maior = mais rápido, mais memória):")

        box = gui.widgetBox(self.controlArea, "Saída")
        rb = gui.radioButtons(box, self, "modo_saida",
                              callback=self._mudou_saida)
        gui.appendRadioButton(rb, "Criar uma nova coluna")
        self.campo_nome = gui.lineEdit(gui.indentedBox(rb), self,
                                       "nome_saida", label="Nome:",
                                       orientation=Qt.Horizontal,
                                       callback=self._mudou_saida)
        self.cb_texto = gui.checkBox(gui.indentedBox(rb), self,
                                     "usar_como_texto",
                                     "Usar como texto do corpus",
                                     callback=self._mudou_saida)
        gui.appendRadioButton(rb, "Substituir o texto original")

        gui.checkBox(self.buttonsArea, self, "auto_traduzir",
                     "Traduzir automaticamente",
                     callback=self._mudou_traducao)
        self.botao = gui.button(self.buttonsArea, self, "Traduzir",
                                callback=self.traduzir)

        self.lbl_resumo = gui.widgetLabel(
            self.mainArea, "A tradução pode levar alguns minutos (o modelo é "
                           "baixado no primeiro uso).")
        self.lbl_resumo.setWordWrap(True)
        self.tabela = QTableWidget(0, 2)
        self.tabela.setHorizontalHeaderLabels(["Original", "Tradução"])
        self.tabela.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.tabela.setWordWrap(True)
        self.tabela.setEditTriggers(QTableWidget.NoEditTriggers)
        self.mainArea.layout().addWidget(self.tabela)

        self._atualizar_controles()

    # ---- estado dos controles ---------------------------------------------
    @property
    def modelo(self):
        if self.modo_modelo == PREDEFINIDO:
            return T.MODELOS[self.modelo_idx]
        nome = self.nome_modelo_avancado.strip()
        if self.tipo_modelo_avancado == TIPO_M2M100:
            return {"id": nome, "tipo": "m2m100", "rotulo": nome}
        return {"id": nome, "tipo": "t5", "rotulo": nome,
               "prefixo": self.prefixo_avancado}

    def _atualizar_controles(self):
        predefinido = self.modo_modelo == PREDEFINIDO
        self.caixa_lista.setVisible(predefinido)
        self.caixa_avancado.setVisible(not predefinido)
        if not predefinido:
            self.caixa_prefixo.setVisible(
                self.tipo_modelo_avancado == TIPO_T5)
        self.box_idiomas.setVisible(self.modelo["tipo"] == "m2m100")
        self.adjustSize()
        nova = self.modo_saida == NOVA_COLUNA
        self.campo_nome.setEnabled(nova)
        self.cb_texto.setEnabled(nova)

    def _mudou_modelo(self):
        self._atualizar_controles()
        self._mudou_traducao()

    def _mudou_traducao(self):
        if self.data is None:
            return
        if self.auto_traduzir:
            self.traduzir()
        elif self._chave() != self._chave_cache:
            self.Information.pendente()

    def _mudou_saida(self):
        self._atualizar_controles()
        if self._traducoes is not None and self._chave() == self._chave_cache:
            self._enviar(self._traducoes)   # não precisa traduzir de novo

    # ---- entrada ------------------------------------------------------------
    @Inputs.data
    def set_data(self, data):
        self.cancel()
        self.closeContext()
        self.Error.clear()
        self.Information.clear()
        self.data = data
        self._chave_cache = self._traducoes = None
        self.modelo_colunas.set_domain(data.domain if data else None)
        self.coluna = None
        self.tabela.setRowCount(0)
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
            if self.auto_traduzir:
                self.traduzir()
            else:
                self.Information.pendente()

    # ---- tradução -----------------------------------------------------------
    def _origem_destino(self):
        if self.modelo["tipo"] == "m2m100":
            return codigo(self.idioma_origem), codigo(self.idioma_destino)
        return None, None

    def _chave(self):
        m = self.modelo
        return (id(self.data), self.coluna and self.coluna.name,
                m["id"], m["tipo"], m.get("prefixo"), self._origem_destino(),
                self.num_beams, self.max_tokens)

    def _textos(self):
        if hasattr(self.data, "get_column"):
            valores = self.data.get_column(self.coluna)
        else:
            valores = self.data.get_column_view(self.coluna)[0]
        return ["" if v is None else str(v) for v in valores]

    def traduzir(self):
        self.Error.clear()
        self.Information.pendente.clear()
        if self.data is None or self.coluna is None:
            return
        if self.modo_modelo == AVANCADO and not self.modelo["id"]:
            self.Error.sem_nome_modelo()
            return
        origem, destino = self._origem_destino()
        if origem is not None and origem == destino:
            self.Error.mesmo_idioma()
            return
        chave = self._chave()
        if chave == self._chave_cache and self._traducoes is not None:
            self._enviar(self._traducoes)
            return
        self._chave_em_curso = chave
        self.lbl_resumo.setText(f"Traduzindo {len(self.data)} documentos...")
        self.start(tarefa, self._textos(), self.modelo,
                   origem or "pt", destino or "en", self.tamanho_lote,
                   self.num_beams, self.max_tokens)

    def on_done(self, traducoes):
        self._chave_cache = self._chave_em_curso
        self._traducoes = traducoes
        self._enviar(traducoes)

    def on_exception(self, ex):
        if isinstance(ex, T.Cancelado):
            return
        self.lbl_resumo.setText("A tradução falhou.")
        self.Error.falha(ex)

    def on_partial_result(self, _):
        pass

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    # ---- saída --------------------------------------------------------------
    def _enviar(self, traducoes):
        self.Information.nome_alterado.clear()
        dominio = self.data.domain
        if self.modo_saida == SUBSTITUIR:
            saida = self.data.copy()
            indice = dominio.metas.index(self.coluna)
            with saida.unlocked(saida.metas):
                saida.metas[:, indice] = traducoes
        else:
            pedido = self.nome_saida.strip() or "texto_traduzido"
            nome = get_unique_names(dominio, pedido)
            if nome != pedido:
                self.Information.nome_alterado(pedido, nome)
            var = StringVariable(nome)
            saida = self.data.transform(
                Domain(dominio.attributes, dominio.class_vars,
                       dominio.metas + (var,)))
            with saida.unlocked(saida.metas):
                saida.metas[:, -1] = traducoes
            if self.usar_como_texto and hasattr(saida, "set_text_features"):
                saida.set_text_features([var])

        originais = self._textos()
        self.tabela.setRowCount(min(N_PREVIA, len(originais)))
        for i in range(self.tabela.rowCount()):
            for j, texto in enumerate((originais[i], traducoes[i])):
                item = QTableWidgetItem(texto[:400])
                item.setToolTip(texto[:2000])
                self.tabela.setItem(i, j, item)
        self.tabela.resizeRowsToContents()
        self.lbl_resumo.setText(
            f"{len(traducoes)} documentos traduzidos com "
            f"{self.modelo['id']}. Prévia dos primeiros "
            f"{self.tabela.rowCount()}.")
        self.Outputs.data.send(saida)

    def send_report(self):
        itens = [("Modelo", self.modelo["id"])]
        if self.modelo["tipo"] == "m2m100":
            itens.append(("Idiomas",
                          f"{self.idioma_origem} → {self.idioma_destino}"))
        self.report_items(itens)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWTraducao).run(Table("book-excerpts"))
