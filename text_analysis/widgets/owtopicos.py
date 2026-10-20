import json

from Orange.base import Model
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.widget import OWWidget, Output, Msg

from text_analysis.modelo import (
    carregar_tokenizador, eh_erro_cuda, liberar_memoria, mensagem_erro,
    numeros_formato_c, ErroTokenizador)
from text_analysis.topicos import TopicosZeroShot, INFERENCIA_PADRAO

TEXTO_TESTE = "Este é um texto de teste para validar a inferência do modelo."

PREDEFINIDO, AVANCADO = 0, 1

MODELOS_PREDEFINIDOS = [
    {
        "id": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        "rotulo": "mDeBERTa-v3-base XNLI (recomendado)",
        "descricao": "Modelo multilíngue (mais de 100 idiomas) baseado em "
                     "DeBERTa-v3; bom equilíbrio entre qualidade e "
                     "velocidade para zero-shot.",
    },
    {
        "id": "MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli",
        "rotulo": "MiniLMv2-L12 multilíngue (rápido, mais leve)",
        "descricao": "Modelo multilíngue leve e rápido; boa opção quando "
                     "velocidade importa mais que precisão máxima.",
    },
    {
        "id": "MoritzLaurer/xlm-v-base-mnli-xnli",
        "rotulo": "XLM-V-base MNLI/XNLI (vocabulário amplo)",
        "descricao": "Modelo multilíngue baseado no XLM-V, com vocabulário "
                     "bem maior que o do XLM-R; útil para idiomas com "
                     "escrita menos comum.",
    },
    {
        "id": "vicgalle/xlm-roberta-large-xnli-anli",
        "rotulo": "XLM-RoBERTa-large XNLI+ANLI (maior, mais preciso)",
        "descricao": "Modelo grande baseado em XLM-RoBERTa-large, treinado "
                     "em XNLI e ANLI; maior qualidade, porém mais lento e "
                     "pesado.",
    },
]


class ErroCarregamentoPesos(Exception):
    pass


class ErroInferenciaTeste(Exception):
    pass


def _tarefa(nome, nome_tokenizador, kw_carga, kw_inf, topicos,
           state: TaskState):
    """Baixa o modelo (tokenizador e pesos) e faz uma inferência de teste
    com os tópicos informados, reportando o progresso em `state`."""
    state.set_status("Baixando o tokenizador...")
    state.set_progress_value(10)
    tokenizador = carregar_tokenizador(nome_tokenizador or nome, kw_carga)
    state.set_progress_value(40)

    state.set_status("Baixando os pesos do modelo...")
    try:
        from transformers import pipeline
        kw = dict(kw_carga)
        if "tokenizer" not in kw:
            kw["tokenizer"] = tokenizador
        kw.pop("use_fast", None)
        with numeros_formato_c():
            try:
                pipe = pipeline("zero-shot-classification", model=nome, **kw)
            except Exception as ex:
                if not eh_erro_cuda(ex) or "device" in kw_carga:
                    raise
                # Um erro de CUDA pode deixar a GPU num estado inválido;
                # tenta recuperar carregando direto na CPU.
                kw["device"] = "cpu"
                pipe = pipeline("zero-shot-classification", model=nome, **kw)
    except Exception as ex:
        raise ErroCarregamentoPesos(mensagem_erro(ex)) from ex
    state.set_progress_value(75)

    state.set_status("Fazendo uma inferência de teste...")
    try:
        kw_teste = dict(INFERENCIA_PADRAO)
        kw_teste.update(kw_inf)
        pipe(TEXTO_TESTE, candidate_labels=topicos, **kw_teste)
    except Exception as ex:
        raise ErroInferenciaTeste(mensagem_erro(ex)) from ex
    finally:
        # O pipeline só serve para validar o carregamento/a inferência aqui;
        # não é guardado para reuso, para não manter o modelo (de centenas
        # de MB a alguns GB) alocado em memória sem necessidade.
        del pipe
        liberar_memoria()
    state.set_progress_value(100)


class OWTopicos(OWWidget, ConcurrentWidgetMixin):
    name = "Zero-shot Classification"
    description = ("Classifica textos por tópicos definidos pelo usuário, "
                   "usando um modelo zero-shot multilíngue (NLI) do "
                   "Hugging Face — não precisa de dados rotulados.")
    icon = "icons/zeroshot.svg"
    priority = 12
    keywords = ["tópicos", "topicos", "topics", "zero-shot", "nli",
               "classificação", "multilíngue"]
    want_main_area = False
    resizing_enabled = False

    modo_modelo = Setting(PREDEFINIDO)
    modelo_idx = Setting(0)
    nome_modelo_avancado = Setting("")
    nome_tokenizador = Setting("")
    topicos = Setting("")
    multi_label = Setting(False)
    nome_alvo = Setting("topico")
    coluna_texto = Setting("text")
    json_carregamento = Setting("")
    json_inferencia = Setting("")
    manter_vivo = Setting(False)

    class Outputs:
        model = Output("Model", Model)

    class Error(OWWidget.Error):
        sem_nome = Msg("Informe o nome do modelo e da coluna de texto.")
        sem_topicos = Msg("Informe ao menos dois tópicos candidatos, "
                          "separados por vírgula.")
        topicos_repetidos = Msg("Os tópicos não podem se repetir.")
        json_invalido = Msg("Parâmetros de {}: JSON inválido ({}). "
                            "Exemplo de formato válido: \"device\": \"cpu\" "
                            "entre chaves.")
        carregamento = Msg("Não foi possível carregar os pesos do modelo: {}")
        inferencia_teste = Msg("Falha ao testar a inferência do modelo: {}")
        tokenizador = Msg("{}")

    class Warning(OWWidget.Warning):
        candidate_labels_ignorado = Msg(
            "O parâmetro 'candidate_labels' é ignorado: use o campo "
            "Tópicos candidatos.")
        multi_label_nao_soma = Msg(
            "Com 'Permitir mais de um tópico por texto' ativado, as "
            "probabilidades de cada tópico são independentes e não somam "
            "100%.")

    class Information(OWWidget.Information):
        pendente = Msg("Há alterações não aplicadas: clique em Aplicar para "
                       "carregar/atualizar o modelo.")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
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
        e = gui.lineEdit(self.caixa_avancado, self, "nome_modelo_avancado",
                         label="Nome no Hugging Face:",
                         callback=self._marcar_pendente)
        e.setPlaceholderText("ex.: facebook/bart-large-mnli")
        e = gui.lineEdit(self.caixa_avancado, self, "nome_tokenizador",
                         label="Tokenizador (opcional; vazio = o do modelo):",
                         callback=self._marcar_pendente)
        e.setPlaceholderText("ex.: FacebookAI/xlm-roberta-large")
        e = gui.lineEdit(self.caixa_avancado, self, "json_carregamento",
                         label="Carregamento (from_pretrained / pipeline):",
                         callback=self._marcar_pendente)
        e.setPlaceholderText('ex.: {"revision": "main", "device": 0}')
        e = gui.lineEdit(self.caixa_avancado, self, "json_inferencia",
                         label="Inferência (chamada do pipeline):",
                         callback=self._marcar_pendente)
        e.setPlaceholderText(
            'ex.: {"hypothesis_template": "Este texto é sobre {}."}')

        box = gui.widgetBox(self.controlArea, "Tópicos candidatos")
        e = gui.lineEdit(box, self, "topicos",
                         label="Tópicos (separados por vírgula):",
                         callback=self._marcar_pendente)
        e.setPlaceholderText("ex.: esporte, política, tecnologia, saúde")
        gui.checkBox(box, self, "multi_label",
                     "Permitir mais de um tópico por texto (não "
                     "mutuamente exclusivos)",
                     callback=self._marcar_pendente)
        gui.lineEdit(box, self, "nome_alvo", label="Nome da variável alvo:",
                     callback=self._marcar_pendente)

        box = gui.widgetBox(self.controlArea, "Texto")
        gui.lineEdit(box, self, "coluna_texto",
                     label="Coluna de texto do corpus:",
                     callback=self._marcar_pendente)

        box = gui.widgetBox(self.controlArea, "Memória")
        gui.checkBox(box, self, "manter_vivo",
                     "Manter o modelo carregado em memória enquanto este "
                     "widget existir",
                     callback=self._mudou_manter_vivo)
        lbl = gui.widgetLabel(
            box, "Desligado (padrão): cada previsão carrega o modelo e o "
                "libera da memória em seguida — usa menos memória, porém é "
                "mais lento a cada uso. Ligado: o modelo fica carregado até "
                "este widget ser removido ou a opção ser desligada — mais "
                "rápido, mas usa mais memória.")
        lbl.setWordWrap(True)

        self.lbl_status = gui.widgetLabel(self.controlArea,
                                          "(clique em Aplicar)")
        self.lbl_status.setWordWrap(True)

        self.botao = gui.button(self.controlArea, self, "Aplicar",
                                callback=self.commit)
        self._modelo_atual = None
        self._atualizar_estado_modelo()
        self._marcar_pendente()

    def onDeleteWidget(self):
        self.shutdown()
        self._liberar_modelo_atual()
        super().onDeleteWidget()

    def _liberar_modelo_atual(self):
        if self._modelo_atual is not None:
            self._modelo_atual.liberar()
            self._modelo_atual = None

    def _mudou_manter_vivo(self):
        # Aplica a troca também ao modelo já enviado (se houver), sem
        # precisar clicar em Aplicar de novo.
        if self._modelo_atual is not None:
            self._modelo_atual.manter_vivo = self.manter_vivo
            if not self.manter_vivo:
                self._modelo_atual.liberar()

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

    def _marcar_pendente(self):
        self.Information.pendente()

    def _modelo_e_tokenizador(self):
        if self.modo_modelo == PREDEFINIDO:
            return MODELOS_PREDEFINIDOS[self.modelo_idx]["id"], None
        return (self.nome_modelo_avancado.strip(),
                self.nome_tokenizador.strip() or None)

    def _topicos_lista(self):
        return [t.strip() for t in self.topicos.split(",") if t.strip()]

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

        topicos = self._topicos_lista()
        if len(topicos) < 2:
            self.Error.sem_topicos()
            return
        if len(set(topicos)) != len(topicos):
            self.Error.topicos_repetidos()
            return

        if self.modo_modelo == AVANCADO:
            kw_carga = self._ler_json(self.json_carregamento, "carregamento")
            kw_inf = self._ler_json(self.json_inferencia, "inferência")
            if kw_carga is None or kw_inf is None:
                return
        else:
            kw_carga, kw_inf = {}, {}
        if "candidate_labels" in kw_inf:
            kw_inf.pop("candidate_labels")
            self.Warning.candidate_labels_ignorado()
        kw_inf["multi_label"] = self.multi_label
        if self.multi_label:
            self.Warning.multi_label_nao_soma()

        self._contexto = dict(nome=nome, coluna=coluna, kw_carga=kw_carga,
                              kw_inf=kw_inf, nome_tokenizador=nome_tokenizador,
                              topicos=topicos)

        chave = (nome, nome_tokenizador, tuple(topicos),
                 json.dumps(kw_carga, sort_keys=True),
                 json.dumps(kw_inf, sort_keys=True))
        if chave == self._chave_cache and self._cache_resultado:
            self._finalizar()
            return

        self.lbl_status.setText("Carregando o modelo...")
        self._chave_cache = chave
        self.botao.setEnabled(False)
        self.start(_tarefa, nome, nome_tokenizador, kw_carga, kw_inf, topicos)

    def on_done(self, _resultado):
        self.botao.setEnabled(True)
        self._cache_resultado = True  # só marca que esta configuração já foi validada
        self._finalizar()

    def on_exception(self, ex):
        self.botao.setEnabled(True)
        self._chave_cache = None
        self.lbl_status.setText("Falha ao carregar o modelo.")
        if isinstance(ex, ErroTokenizador):
            self.Error.tokenizador(ex)
        elif isinstance(ex, ErroInferenciaTeste):
            self.Error.inferencia_teste(ex)
        else:
            self.Error.carregamento(ex)

    def on_partial_result(self, _):
        pass

    def _finalizar(self):
        ctx = self._contexto
        self._liberar_modelo_atual()  # solta o modelo anterior antes de trocar
        modelo = TopicosZeroShot(
            ctx["nome"], ctx["topicos"], coluna_texto=ctx["coluna"],
            nome_tokenizador=ctx["nome_tokenizador"],
            kwargs_carregamento=ctx["kw_carga"],
            kwargs_inferencia=ctx["kw_inf"],
            nome_alvo=self.nome_alvo.strip() or "topico",
            manter_vivo=self.manter_vivo)
        self._modelo_atual = modelo
        self.lbl_status.setText(
            f"Pronto: {len(ctx['topicos'])} tópicos — "
            + ", ".join(ctx["topicos"]))
        self.Outputs.model.send(modelo)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWTopicos).run()
