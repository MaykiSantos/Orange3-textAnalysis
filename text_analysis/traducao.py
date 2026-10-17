"""Tradução de textos com modelos seq2seq do Hugging Face.

Usa AutoTokenizer + AutoModelForSeq2SeqLM + generate() diretamente, porque o
transformers 5.x não tem mais os pipelines "translation" e
"text2text-generation".
"""
import re

from text_analysis.modelo import numeros_formato_c

MODELOS = [
    {"id": "unicamp-dl/translation-pt-en-t5", "tipo": "t5",
     "rotulo": "Português → Inglês (unicamp-dl/translation-pt-en-t5)",
     "prefixo": "translate Portuguese to English: "},
    {"id": "unicamp-dl/translation-en-pt-t5", "tipo": "t5",
     "rotulo": "Inglês → Português (unicamp-dl/translation-en-pt-t5)",
     "prefixo": "translate English to Portuguese: "},
    {"id": "facebook/m2m100_418M", "tipo": "m2m100",
     "rotulo": "Multilíngue, 100 idiomas (facebook/m2m100_418M)"},
]

# Os 100 idiomas do M2M100 (códigos do próprio modelo)
IDIOMAS_M2M100 = {
    "af": "Africâner", "am": "Amárico", "ar": "Árabe", "ast": "Asturiano",
    "az": "Azerbaijano", "ba": "Bashkir", "be": "Bielorrusso",
    "bg": "Búlgaro", "bn": "Bengali", "br": "Bretão", "bs": "Bósnio",
    "ca": "Catalão", "ceb": "Cebuano", "cs": "Tcheco", "cy": "Galês",
    "da": "Dinamarquês", "de": "Alemão", "el": "Grego", "en": "Inglês",
    "es": "Espanhol", "et": "Estoniano", "fa": "Persa", "ff": "Fula",
    "fi": "Finlandês", "fr": "Francês", "fy": "Frísio ocidental",
    "ga": "Irlandês", "gd": "Gaélico escocês", "gl": "Galego",
    "gu": "Guzerate", "ha": "Hauçá", "he": "Hebraico", "hi": "Hindi",
    "hr": "Croata", "ht": "Crioulo haitiano", "hu": "Húngaro",
    "hy": "Armênio", "id": "Indonésio", "ig": "Igbo", "ilo": "Ilocano",
    "is": "Islandês", "it": "Italiano", "ja": "Japonês", "jv": "Javanês",
    "ka": "Georgiano", "kk": "Cazaque", "km": "Khmer", "kn": "Canarês",
    "ko": "Coreano", "lb": "Luxemburguês", "lg": "Luganda",
    "ln": "Lingala", "lo": "Laosiano", "lt": "Lituano", "lv": "Letão",
    "mg": "Malgaxe", "mk": "Macedônio", "ml": "Malaiala", "mn": "Mongol",
    "mr": "Marati", "ms": "Malaio", "my": "Birmanês", "ne": "Nepalês",
    "nl": "Holandês", "no": "Norueguês", "ns": "Soto do norte",
    "oc": "Occitano", "or": "Oriá", "pa": "Panjabi", "pl": "Polonês",
    "ps": "Pashto", "pt": "Português", "ro": "Romeno", "ru": "Russo",
    "sd": "Sindi", "si": "Cingalês", "sk": "Eslovaco", "sl": "Esloveno",
    "so": "Somali", "sq": "Albanês", "sr": "Sérvio", "ss": "Suázi",
    "su": "Sundanês", "sv": "Sueco", "sw": "Suaíli", "ta": "Tâmil",
    "th": "Tailandês", "tl": "Tagalo", "tn": "Tswana", "tr": "Turco",
    "uk": "Ucraniano", "ur": "Urdu", "uz": "Uzbeque", "vi": "Vietnamita",
    "wo": "Wolof", "xh": "Xhosa", "yi": "Iídiche", "yo": "Iorubá",
    "zh": "Chinês", "zu": "Zulu",
}

RE_FRASES = re.compile(r"(?<=[.!?…])\s+")
MAX_CARACTERES = 400   # tamanho dos trechos enviados ao modelo
_CACHE = {}            # modelos já carregados nesta sessão do Orange


class Cancelado(Exception):
    pass


def carregar(id_modelo):
    """Carrega (uma vez por sessão) tokenizador e modelo; usa GPU se houver."""
    if id_modelo not in _CACHE:
        import torch
        from transformers import AutoModelForSeq2SeqLM
        from text_analysis.modelo import carregar_tokenizador
        tok = carregar_tokenizador(id_modelo)
        with numeros_formato_c():
            mod = AutoModelForSeq2SeqLM.from_pretrained(id_modelo)
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            mod.to(dispositivo)
        except Exception:
            dispositivo = "cpu"
            mod.to(dispositivo)
        mod.eval()
        _CACHE[id_modelo] = (tok, mod, dispositivo)
    return _CACHE[id_modelo]


def dividir(texto, max_caracteres=MAX_CARACTERES):
    """Divide textos longos em trechos de frases inteiras.

    Os modelos leem no máximo ~512 tokens; sem dividir, o fim de textos
    longos seria simplesmente cortado da tradução.
    """
    texto = texto.strip()
    if not texto:
        return []
    trechos, atual = [], ""
    for frase in RE_FRASES.split(texto):
        while len(frase) > max_caracteres:      # frase gigante: corta
            corte = frase.rfind(" ", 0, max_caracteres)
            corte = corte if corte > 0 else max_caracteres
            if atual:
                trechos.append(atual)
                atual = ""
            trechos.append(frase[:corte])
            frase = frase[corte:].strip()
        if atual and len(atual) + 1 + len(frase) > max_caracteres:
            trechos.append(atual)
            atual = frase
        else:
            atual = f"{atual} {frase}" if atual else frase
    if atual:
        trechos.append(atual)
    return trechos


def _id_idioma(tok, codigo):
    if hasattr(tok, "get_lang_id"):
        return tok.get_lang_id(codigo)
    return tok.convert_tokens_to_ids(f"__{codigo}__")


def traduzir(textos, cfg, origem="pt", destino="en", tamanho_lote=8,
             num_beams=1, max_tokens=256, progresso=None, cancelado=None):
    """Traduz uma lista de textos. Retorna a lista traduzida, na mesma ordem.

    `cfg` é um dos dicionários de MODELOS (ou um equivalente para um modelo
    avançado/personalizado): precisa de "id" e "tipo" ("t5" ou "m2m100"); se
    "t5", também de "prefixo".
    """
    import torch
    tok, mod, dispositivo = carregar(cfg["id"])

    # Divide em trechos e traduz cada trecho distinto uma vez só
    trechos_por_texto = [dividir(t) for t in textos]
    unicos = list(dict.fromkeys(p for ps in trechos_por_texto for p in ps))
    # Ordenar por tamanho deixa os lotes mais uniformes (mais rápido)
    unicos.sort(key=len)

    gerar = {"num_beams": int(num_beams), "max_new_tokens": int(max_tokens)}
    if cfg["tipo"] == "m2m100":
        tok.src_lang = origem
        gerar["forced_bos_token_id"] = _id_idioma(tok, destino)
        prefixo = ""
    else:
        prefixo = cfg["prefixo"]

    traducao = {}
    n = len(unicos)
    for ini in range(0, n, tamanho_lote):
        if cancelado is not None and cancelado():
            raise Cancelado()
        lote = unicos[ini:ini + tamanho_lote]
        entrada = tok([prefixo + p for p in lote], return_tensors="pt",
                      padding=True, truncation=True, max_length=512)
        entrada = entrada.to(dispositivo)
        with torch.no_grad():
            ids = mod.generate(**entrada, **gerar)
        for original, traduzido in zip(
                lote, tok.batch_decode(ids, skip_special_tokens=True)):
            traducao[original] = traduzido.strip()
        if progresso is not None:
            progresso(100 * min(ini + tamanho_lote, n) / max(n, 1))

    return [" ".join(traducao[p] for p in ps) for ps in trechos_por_texto]
