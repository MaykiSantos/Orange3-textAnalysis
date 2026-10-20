import os

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import (
    QFileDialog, QHeaderView, QTableWidget, QTableWidgetItem)

from Orange.data import Domain, StringVariable, Table
from Orange.data.util import get_unique_names
from Orange.widgets import gui
from Orange.widgets.settings import (
    Setting, ContextSetting, DomainContextHandler)
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import OWWidget, Input, Output, Msg

from text_analysis import limpeza as L

N_PREVIA = 30


class OWLimpezaTexto(OWWidget):
    name = "Limpeza de Texto"
    description = ("Limpa o texto (HTML, links, menções, emojis, acentos...) "
                   "e cria uma nova coluna com o resultado.")
    icon = "icons/limpeza.svg"
    priority = 5
    keywords = ["limpeza", "clean", "texto", "preprocess", "normalizar"]

    settingsHandler = DomainContextHandler()
    coluna = ContextSetting(None)

    # Ruído da web
    corrigir_codificacao = Setting(False)
    remover_html = Setting(True)
    links = Setting(L.SUBSTITUIR)
    emails = Setting(L.SUBSTITUIR)
    mencoes = Setting(L.SUBSTITUIR)
    hashtags = Setting(L.HASHTAG_MANTER)
    # Conteúdo
    emojis = Setting(L.EMOJI_MANTER)
    numeros = Setting(False)
    stopwords = Setting(False)
    stopwords_idioma = Setting(L.IDIOMA_PT)
    stopwords_idioma_nltk = Setting("")
    stopwords_extras = Setting("")
    stopwords_csv = Setting("")
    repeticoes = Setting(True)
    max_repeticoes = Setting(3)
    # Normalização
    minusculas = Setting(False)
    acentos = Setting(False)
    pontuacao = Setting(False)
    espacos = Setting(True)
    # Documentos
    min_caracteres = Setting(0)
    remover_duplicados = Setting(False)
    # Saída
    nome_saida = Setting("texto_limpo")
    usar_como_texto = Setting(True)
    auto_commit = Setting(True)

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    class Error(OWWidget.Error):
        sem_texto = Msg("Os dados não têm nenhuma coluna de texto.")
        stopwords_falha = Msg("Não foi possível carregar as stopwords: {}")

    class Warning(OWWidget.Warning):
        sem_ftfy = Msg("Para corrigir a codificação, instale a biblioteca "
                       "'ftfy' e reinicie o Orange.")
        sem_emoji = Msg("Para converter emojis em texto, instale a biblioteca "
                        "'emoji' e reinicie o Orange (por ora, eles são "
                        "removidos).")

    class Information(OWWidget.Information):
        nome_alterado = Msg("Já existe uma coluna '{}'; a nova coluna se "
                            "chama '{}'.")

    def __init__(self):
        super().__init__()
        self.data = None
        mudou = self._mudou

        def combo(caixa, attr, rotulo, itens):
            gui.comboBox(caixa, self, attr, label=rotulo, items=itens,
                         orientation=Qt.Horizontal, callback=mudou)

        box = gui.widgetBox(self.controlArea, "Coluna de texto")
        self.modelo_colunas = DomainModel(order=DomainModel.METAS,
                                          valid_types=(StringVariable,))
        gui.comboBox(box, self, "coluna", model=self.modelo_colunas,
                     callback=mudou)

        box = gui.widgetBox(self.controlArea, "Ruído da web")
        gui.checkBox(box, self, "corrigir_codificacao",
                     "Corrigir codificação quebrada (ex.: 'pÃ©ssimo')",
                     callback=mudou)
        gui.checkBox(box, self, "remover_html",
                     "Remover HTML (tags e entidades como &amp;)",
                     callback=mudou)
        tres = ["Manter", "Remover", "Substituir por marcador"]
        combo(box, "links", "Links:", tres)
        combo(box, "emails", "E-mails:", tres)
        combo(box, "mencoes", "Menções (@perfil):", tres)
        combo(box, "hashtags", "Hashtags:",
              ["Manter", "Remover só o #", "Remover inteira"])
        gui.widgetLabel(box, "Marcadores: http, email e @user (formato dos "
                             "modelos treinados com tweets).").setWordWrap(True)

        box = gui.widgetBox(self.controlArea, "Conteúdo")
        combo(box, "emojis", "Emojis:",
              ["Manter", "Remover", "Converter em texto"])
        gui.checkBox(box, self, "numeros", "Remover números", callback=mudou)
        linha = gui.hBox(box)
        gui.checkBox(linha, self, "stopwords",
                     "Remover stopwords (palavras muito comuns):",
                     callback=mudou)
        gui.comboBox(linha, self, "stopwords_idioma",
                     items=["Português", "English",
                           "Outro idioma (baixa via NLTK)"],
                     callback=mudou)
        e = gui.lineEdit(
            box, self, "stopwords_idioma_nltk",
            label="Nome do outro idioma (em inglês, só se selecionado "
                  "acima):", callback=mudou)
        e.setPlaceholderText("ex.: spanish, french, german")
        e = gui.lineEdit(box, self, "stopwords_extras",
                         label="Stopwords extras (separadas por vírgula):",
                         callback=mudou)
        e.setPlaceholderText("ex.: tipo, né, então")
        linha = gui.hBox(box)
        gui.widgetLabel(linha, "CSV com mais stopwords:")
        self.lbl_csv_stopwords = gui.widgetLabel(linha, "(nenhum)")
        self.lbl_csv_stopwords.setWordWrap(True)
        gui.button(linha, self, "Selecionar...",
                  callback=self._selecionar_csv_stopwords)
        gui.button(linha, self, "Limpar",
                  callback=self._limpar_csv_stopwords)
        linha = gui.hBox(box)
        gui.checkBox(linha, self, "repeticoes",
                     "Reduzir caracteres repetidos para no máximo",
                     callback=mudou)
        gui.spin(linha, self, "max_repeticoes", 2, 10, callback=mudou)

        box = gui.widgetBox(self.controlArea, "Normalização")
        gui.checkBox(box, self, "espacos",
                     "Remover espaços e quebras de linha excessivos",
                     callback=mudou)
        gui.checkBox(box, self, "minusculas", "Passar para minúsculas",
                     callback=mudou)
        gui.checkBox(box, self, "acentos", "Remover acentos", callback=mudou)
        gui.checkBox(box, self, "pontuacao", "Remover pontuação",
                     callback=mudou)
        aviso = gui.widgetLabel(
            box, "⚠ Minúsculas, acentos, pontuação e emojis carregam "
                 "sentimento: removê-los ajuda léxicos e Bag of Words, mas "
                 "costuma piorar modelos do Hugging Face.")
        aviso.setWordWrap(True)

        box = gui.widgetBox(self.controlArea, "Documentos")
        gui.spin(box, self, "min_caracteres", 0, 10000,
                 label="Descartar textos com menos caracteres que "
                       "(0 = não descartar):", callback=mudou)
        gui.checkBox(box, self, "remover_duplicados",
                     "Remover documentos duplicados (após a limpeza)",
                     callback=mudou)

        box = gui.widgetBox(self.controlArea, "Saída")
        gui.lineEdit(box, self, "nome_saida", label="Nome da nova coluna:",
                     callback=mudou)
        gui.checkBox(box, self, "usar_como_texto",
                     "Usar a nova coluna como texto do corpus",
                     callback=mudou)

        gui.auto_apply(self.buttonsArea, self, "auto_commit")

        # Área principal: prévia antes/depois
        self.lbl_resumo = gui.widgetLabel(self.mainArea, "Sem dados.")
        self.tabela = QTableWidget(0, 2)
        self.tabela.setHorizontalHeaderLabels(["Original", "Limpo"])
        self.tabela.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.tabela.setWordWrap(True)
        self.tabela.setEditTriggers(QTableWidget.NoEditTriggers)
        self.mainArea.layout().addWidget(self.tabela)

        self._atualizar_lbl_csv_stopwords()

    # ---- stopwords: CSV extra --------------------------------------------
    def _atualizar_lbl_csv_stopwords(self):
        self.lbl_csv_stopwords.setText(
            os.path.basename(self.stopwords_csv) if self.stopwords_csv
            else "(nenhum)")

    def _selecionar_csv_stopwords(self):
        caminho, _ = QFileDialog.getOpenFileName(
            self, "Selecionar CSV de stopwords", self.stopwords_csv or "",
            "CSV (*.csv);;Todos os arquivos (*)")
        if caminho:
            self.stopwords_csv = caminho
            self._atualizar_lbl_csv_stopwords()
            self._mudou()

    def _limpar_csv_stopwords(self):
        if self.stopwords_csv:
            self.stopwords_csv = ""
            self._atualizar_lbl_csv_stopwords()
            self._mudou()

    # ---- entrada --------------------------------------------------------
    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.Error.clear()
        self.data = data
        self.modelo_colunas.set_domain(data.domain if data else None)
        self.coluna = None
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
        self._atualizar_previa()
        self.commit.now()

    # ---- processamento --------------------------------------------------
    def _mudou(self):
        self._atualizar_previa()
        self.commit.deferred()

    def _resolver_stopwords(self):
        if not self.stopwords:
            self.Error.stopwords_falha.clear()
            return None
        try:
            palavras = L.resolver_stopwords(
                idioma=self.stopwords_idioma,
                idioma_nltk=self.stopwords_idioma_nltk,
                extras=self.stopwords_extras,
                arquivo_csv=self.stopwords_csv)
            self.Error.stopwords_falha.clear()
            return palavras
        except Exception as ex:
            self.Error.stopwords_falha(str(ex))
            return None

    def _opcoes(self):
        self.Warning.sem_ftfy(shown=self.corrigir_codificacao
                              and not L.TEM_FTFY)
        self.Warning.sem_emoji(shown=self.emojis == L.EMOJI_TEXTO
                               and not L.TEM_EMOJI)
        return dict(
            corrigir_codificacao=self.corrigir_codificacao,
            remover_html=self.remover_html, links=self.links,
            emails=self.emails, mencoes=self.mencoes,
            hashtags=self.hashtags, emojis=self.emojis,
            numeros=self.numeros, stopwords=self._resolver_stopwords(),
            repeticoes=self.repeticoes,
            max_repeticoes=self.max_repeticoes, minusculas=self.minusculas,
            acentos=self.acentos, pontuacao=self.pontuacao,
            espacos=self.espacos)

    def _textos_originais(self, data, n=None):
        if n is not None:
            data = data[:n]
        if hasattr(data, "get_column"):          # Orange >= 3.34
            valores = data.get_column(self.coluna)
        else:
            valores = data.get_column_view(self.coluna)[0]
        return ["" if v is None else str(v) for v in valores]

    def _filtrar(self, limpos):
        """Máscara de documentos mantidos (tamanho mínimo e duplicados)."""
        vistos, manter = set(), []
        for t in limpos:
            ok = len(t.strip()) >= self.min_caracteres
            if ok and self.remover_duplicados:
                ok = t not in vistos
                vistos.add(t)
            manter.append(ok)
        return manter

    def _atualizar_previa(self):
        self.tabela.setRowCount(0)
        if self.data is None or self.coluna is None:
            self.lbl_resumo.setText("Sem dados.")
            return
        opcoes = self._opcoes()
        originais = self._textos_originais(self.data, N_PREVIA)
        self.tabela.setRowCount(len(originais))
        for i, original in enumerate(originais):
            limpo = L.limpar_texto(original, **opcoes)
            for j, texto in enumerate((original, limpo)):
                item = QTableWidgetItem(texto[:400])
                item.setToolTip(texto[:2000])
                self.tabela.setItem(i, j, item)
        self.tabela.resizeRowsToContents()
        self.lbl_resumo.setText(
            f"Prévia dos primeiros {len(originais)} de {len(self.data)} "
            "documentos.")

    @gui.deferred
    def commit(self):
        self.Information.clear()
        if self.data is None or self.coluna is None:
            self.Outputs.data.send(None)
            return
        opcoes = self._opcoes()
        originais = self._textos_originais(self.data)
        limpos = [L.limpar_texto(t, **opcoes) for t in originais]

        dominio = self.data.domain
        pedido = self.nome_saida.strip() or "texto_limpo"
        nome = get_unique_names(dominio, pedido)
        if nome != pedido:
            self.Information.nome_alterado(pedido, nome)
        var = StringVariable(nome)
        novo_dominio = Domain(dominio.attributes, dominio.class_vars,
                              dominio.metas + (var,))
        saida = self.data.transform(novo_dominio)
        with saida.unlocked(saida.metas):
            saida.metas[:, -1] = limpos

        manter = self._filtrar(limpos)
        descartados = manter.count(False)
        if descartados:
            saida = saida[[i for i, ok in enumerate(manter) if ok]]
        if self.usar_como_texto and hasattr(saida, "set_text_features"):
            saida.set_text_features([var])

        media = lambda ts: sum(map(len, ts)) / max(1, len(ts))
        self.lbl_resumo.setText(
            f"{len(saida)} de {len(self.data)} documentos na saída"
            + (f" ({descartados} descartados)" if descartados else "")
            + f". Tamanho médio: {media(originais):.0f} → "
              f"{media(limpos):.0f} caracteres. Prévia dos primeiros "
              f"{min(N_PREVIA, len(self.data))}.")
        self.Outputs.data.send(saida)

    def send_report(self):
        if self.data is not None and self.coluna is not None:
            self.report_items((("Coluna", self.coluna.name),
                               ("Nova coluna", self.nome_saida)))


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWLimpezaTexto).run(Table("book-excerpts"))
