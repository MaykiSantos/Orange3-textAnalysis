import json
import re

from Orange.base import Model
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Input, Output, Msg

from text_analysis.modelo import (
    SentimentoHF, ler_config, carregar_tokenizador)


class OWSentimentoHF(OWWidget):
    name = "Modelo Hugging Face"
    description = ("Cria um modelo de classificação de texto (ex.: sentimento) "
                   "a partir de um modelo do Hugging Face.")
    icon = "icons/sentimento.svg"
    priority = 10
    keywords = ["sentiment", "huggingface", "transformers", "sentimento"]
    want_main_area = False
    resizing_enabled = False

    nome_modelo = Setting("cardiffnlp/twitter-xlm-roberta-base-sentiment")
    nome_tokenizador = Setting("")
    nomes_classes = Setting("")
    nome_alvo = Setting("sentimento")
    coluna_texto = Setting("text")
    limpar_texto = Setting(False)
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

    def __init__(self):
        super().__init__()
        self.data = None

        box = gui.widgetBox(self.controlArea, "Modelo")
        gui.lineEdit(box, self, "nome_modelo", label="Nome no Hugging Face:")
        e = gui.lineEdit(box, self, "nome_tokenizador",
                         label="Tokenizador (opcional; vazio = o do modelo):")
        e.setPlaceholderText("ex.: FacebookAI/xlm-roberta-base")

        box = gui.widgetBox(self.controlArea, "Classes")
        self.lbl_rotulos = gui.widgetLabel(
            box, "Rótulos do modelo: (clique em Aplicar)")
        self.lbl_rotulos.setWordWrap(True)
        e = gui.lineEdit(box, self, "nomes_classes",
                         label="Renomear classes (opcional, mesma ordem):")
        e.setPlaceholderText("ex.: negativo, neutro, positivo")
        self.campo_alvo = gui.lineEdit(
            box, self, "nome_alvo",
            label="Nome da variável alvo (sem corpus conectado):")
        self.lbl_alvo = gui.widgetLabel(
            box, "Dica: conecte o corpus rotulado à entrada deste widget para "
                 "usar o alvo dele automaticamente.")
        self.lbl_alvo.setWordWrap(True)

        box = gui.widgetBox(self.controlArea, "Texto")
        gui.lineEdit(box, self, "coluna_texto", label="Coluna de texto do corpus:")
        gui.checkBox(box, self, "limpar_texto",
                     "Trocar menções por @user e links por http")

        box = gui.widgetBox(self.controlArea,
                            "Parâmetros avançados (JSON, opcional)")
        e = gui.lineEdit(box, self, "json_carregamento",
                         label="Carregamento (from_pretrained / pipeline):")
        e.setPlaceholderText('ex.: {"revision": "main", "device": 0}')
        e = gui.lineEdit(box, self, "json_inferencia",
                         label="Inferência (chamada do pipeline):")
        e.setPlaceholderText('ex.: {"max_length": 128, "batch_size": 8}')

        gui.button(self.controlArea, self, "Aplicar", callback=self.commit)
        self.commit()

    @Inputs.data
    def set_data(self, data):
        self.data = data
        self.commit()

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

    def _criar_modelo(self):
        nome = self.nome_modelo.strip()
        coluna = self.coluna_texto.strip()
        if not nome or not coluna:
            self.Error.sem_nome()
            return None

        kw_carga = self._ler_json(self.json_carregamento, "carregamento")
        kw_inf = self._ler_json(self.json_inferencia, "inferência")
        if kw_carga is None or kw_inf is None:
            return None
        if "top_k" in kw_inf:
            kw_inf.pop("top_k")
            self.Warning.top_k()

        try:
            rotulos, arquiteturas = ler_config(nome, kw_carga)
        except Exception as ex:  # rede, nome errado, modelo privado...
            self.lbl_rotulos.setText("Rótulos do modelo: —")
            self.Error.config(ex)
            return None
        if len(rotulos) < 2:
            self.Error.sem_rotulos()
            return None
        self.lbl_rotulos.setText("Rótulos do modelo: " + ", ".join(rotulos))

        if arquiteturas and not any("SequenceClassification" in a
                                    for a in arquiteturas):
            self.Warning.nao_classificacao(", ".join(arquiteturas))

        # Testa o tokenizador já aqui, para o erro aparecer neste widget
        # (e com a causa real) em vez de só no Predictions.
        try:
            carregar_tokenizador(self.nome_tokenizador.strip() or nome, kw_carga)
        except Exception as ex:
            self.Error.tokenizador(ex)
            return None

        nomes = [c.strip() for c in self.nomes_classes.split(",") if c.strip()]
        if nomes and len(nomes) != len(rotulos):
            self.Error.nomes_classes(len(rotulos))
            return None
        if len(set(nomes)) != len(nomes):
            self.Error.nomes_repetidos()
            return None
        if not nomes and all(re.fullmatch(r"LABEL_\d+", r) for r in rotulos):
            self.Warning.rotulos_genericos(", ".join(rotulos))

        alvo = self._alvo_do_corpus()
        if alvo is not None:
            if not alvo.is_discrete:
                self.Error.alvo_numerico(alvo.name)
                return None
            efetivos = nomes or rotulos
            faltando = [n for n in efetivos if n not in alvo.values]
            if len(faltando) == len(efetivos):
                self.Error.sem_correspondencia(", ".join(efetivos),
                                               ", ".join(alvo.values))
                return None
            if faltando:
                self.Warning.classes_faltando(", ".join(faltando))

        return SentimentoHF(
            nome, rotulos, coluna_texto=coluna, nomes_classes=nomes or None,
            nome_tokenizador=self.nome_tokenizador.strip() or None,
            kwargs_carregamento=kw_carga, kwargs_inferencia=kw_inf,
            limpar_texto=self.limpar_texto,
            nome_alvo=self.nome_alvo.strip() or "sentimento",
            variavel_alvo=alvo)

    def commit(self):
        self.Error.clear()
        self.Warning.clear()
        self.Outputs.model.send(self._criar_modelo())


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWSentimentoHF).run()
