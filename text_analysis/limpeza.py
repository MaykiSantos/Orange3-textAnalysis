"""Funções de limpeza de texto usadas pelo widget "Limpeza de Texto".

Ficam separadas do widget para poderem ser testadas e reutilizadas
(ex.: em um Python Script).
"""
import html
import re
import unicodedata

try:
    import ftfy
    TEM_FTFY = True
except ImportError:
    TEM_FTFY = False

try:
    import emoji
    TEM_EMOJI = True
except ImportError:
    TEM_EMOJI = False

# Modos para links, e-mails e menções
MANTER, REMOVER, SUBSTITUIR = 0, 1, 2
# Modos para hashtags
HASHTAG_MANTER, HASHTAG_SO_SIMBOLO, HASHTAG_REMOVER = 0, 1, 2
# Modos para emojis
EMOJI_MANTER, EMOJI_REMOVER, EMOJI_TEXTO = 0, 1, 2

RE_TAG = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
RE_MENCAO = re.compile(r"(?<![\w@])@\w+")
RE_HASHTAG = re.compile(r"(?<![\w#])#(\w+)")
RE_NUMERO = re.compile(r"\d+(?:[.,]\d+)*")
RE_ESPACOS = re.compile(r"\s+")
RE_EMOJI = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # emojis, símbolos e pictogramas
    "\U00002600-\U000027BF"   # símbolos diversos e dingbats
    "\U00002300-\U000023FF"   # símbolos técnicos (⌚, ⏰...)
    "\U00002B00-\U00002BFF"   # setas e símbolos (⭐...)
    "\U0000FE0F\U0000200D"    # seletor de variação e "zero width joiner"
    "]+")
RE_EMOJI_NOME = re.compile("\x00([^\x01]*)\x01")


def _repeticoes(texto, maximo):
    # Letras e pontuação repetidas: "muuuuito!!!!!!" -> "muuuito!!!"
    # (dígitos ficam de fora, para não alterar "1000")
    return re.sub(r"([^\s\d])\1{%d,}" % maximo, r"\1" * maximo, texto)


def _sem_acentos(texto):
    decomposto = unicodedata.normalize("NFD", texto)
    sem = "".join(c for c in decomposto if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", sem)


def _sem_pontuacao(texto):
    return "".join(" " if unicodedata.category(c).startswith("P") else c
                   for c in texto)


def _emojis(texto, modo):
    if modo == EMOJI_TEXTO and TEM_EMOJI:
        com_nomes = emoji.demojize(texto, language="pt",
                                   delimiters=("\x00", "\x01"))
        return RE_EMOJI_NOME.sub(
            lambda m: " " + m.group(1).replace("_", " ") + " ", com_nomes)
    if modo in (EMOJI_REMOVER, EMOJI_TEXTO):  # sem a biblioteca: só remove
        if TEM_EMOJI:
            return emoji.replace_emoji(texto, replace=" ")
        return RE_EMOJI.sub(" ", texto)
    return texto


def limpar_texto(texto, *, corrigir_codificacao=False, remover_html=False,
                 links=MANTER, emails=MANTER, mencoes=MANTER,
                 hashtags=HASHTAG_MANTER, emojis=EMOJI_MANTER,
                 numeros=False, repeticoes=False, max_repeticoes=3,
                 minusculas=False, acentos=False, pontuacao=False,
                 espacos=False):
    """Aplica as limpezas escolhidas, numa ordem que evita conflitos
    (ex.: e-mails antes de menções, pontuação depois dos marcadores)."""
    t = "" if texto is None else str(texto)
    t = unicodedata.normalize("NFC", t)

    if corrigir_codificacao and TEM_FTFY:
        t = ftfy.fix_text(t, uncurl_quotes=False)
    if remover_html:
        t = RE_TAG.sub(" ", html.unescape(t))
    if links != MANTER:
        t = RE_URL.sub(" http " if links == SUBSTITUIR else " ", t)
    if emails != MANTER:
        t = RE_EMAIL.sub(" email " if emails == SUBSTITUIR else " ", t)
    if mencoes != MANTER:
        t = RE_MENCAO.sub("@user" if mencoes == SUBSTITUIR else " ", t)
    if hashtags == HASHTAG_SO_SIMBOLO:
        t = RE_HASHTAG.sub(r"\1", t)
    elif hashtags == HASHTAG_REMOVER:
        t = RE_HASHTAG.sub(" ", t)
    if emojis != EMOJI_MANTER:
        t = _emojis(t, emojis)
    if numeros:
        t = RE_NUMERO.sub(" ", t)
    if repeticoes:
        t = _repeticoes(t, max(1, int(max_repeticoes)))
    if minusculas:
        t = t.lower()
    if acentos:
        t = _sem_acentos(t)
    if pontuacao:
        t = _sem_pontuacao(t)
    if espacos:
        t = RE_ESPACOS.sub(" ", t).strip()
    return t
