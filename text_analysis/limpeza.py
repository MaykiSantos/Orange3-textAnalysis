"""Funções de limpeza de texto usadas pelo widget "Limpeza de Texto".

Ficam separadas do widget para poderem ser testadas e reutilizadas
(ex.: em um Python Script).
"""
import csv
import html
import os
import re
import unicodedata
from functools import lru_cache

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

try:
    import nltk
    TEM_NLTK = True
except ImportError:
    TEM_NLTK = False

# Modos para links, e-mails e menções
MANTER, REMOVER, SUBSTITUIR = 0, 1, 2
# Modos para hashtags
HASHTAG_MANTER, HASHTAG_SO_SIMBOLO, HASHTAG_REMOVER = 0, 1, 2
# Modos para emojis
EMOJI_MANTER, EMOJI_REMOVER, EMOJI_TEXTO = 0, 1, 2
# Idiomas para remoção de stopwords: português e inglês vêm de um arquivo
# local (stopwords/*.txt); qualquer outro idioma é baixado sob demanda via
# NLTK (ver _stopwords_nltk).
IDIOMA_PT, IDIOMA_EN, IDIOMA_OUTRO = 0, 1, 2

_DIR_STOPWORDS = os.path.join(os.path.dirname(__file__), "stopwords")
_ARQUIVO_POR_IDIOMA = {IDIOMA_PT: "portugues.txt", IDIOMA_EN: "ingles.txt"}


@lru_cache(maxsize=None)
def _stopwords_arquivo(nome_arquivo):
    caminho = os.path.join(_DIR_STOPWORDS, nome_arquivo)
    with open(caminho, encoding="utf-8") as f:
        return frozenset(
            linha.strip().lower() for linha in f
            if linha.strip() and not linha.startswith("#"))


@lru_cache(maxsize=None)
def _stopwords_nltk(idioma):
    if not TEM_NLTK:
        raise RuntimeError(
            "Para usar stopwords de outros idiomas, instale a biblioteca "
            "'nltk' e reinicie o Orange.")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    try:
        palavras = nltk.corpus.stopwords.words(idioma)
    except (LookupError, OSError) as ex:
        raise RuntimeError(
            f"O NLTK não tem uma lista de stopwords para '{idioma}'. "
            "Use o nome do idioma em inglês (ex.: 'spanish', 'french', "
            "'german').") from ex
    return frozenset(p.lower() for p in palavras)


@lru_cache(maxsize=None)
def _stopwords_csv(caminho):
    if not caminho:
        return frozenset()
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        leitor = csv.reader(f)
        return frozenset(
            celula.strip().lower()
            for linha in leitor for celula in linha if celula.strip())


def resolver_stopwords(idioma=IDIOMA_PT, idioma_nltk="", extras="",
                       arquivo_csv=""):
    """Reúne num só conjunto as stopwords das fontes escolhidas: lista do
    idioma (arquivo local para português/inglês, ou baixada via NLTK para
    outros idiomas), palavras extras digitadas e/ou um CSV com mais
    stopwords.
    """
    arquivo = _ARQUIVO_POR_IDIOMA.get(idioma)
    if arquivo is not None:
        palavras = _stopwords_arquivo(arquivo)
    else:
        palavras = _stopwords_nltk((idioma_nltk or "").strip().lower()
                                   or "english")
    if extras:
        palavras = palavras | {p.strip().lower()
                               for p in extras.split(",") if p.strip()}
    if arquivo_csv:
        palavras = palavras | _stopwords_csv(arquivo_csv)
    return palavras


RE_TAG = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
RE_MENCAO = re.compile(r"(?<![\w@])@\w+")
RE_HASHTAG = re.compile(r"(?<![\w#])#(\w+)")
RE_NUMERO = re.compile(r"\d+(?:[.,]\d+)*")
RE_ESPACOS = re.compile(r"\s+")
RE_PALAVRA = re.compile(r"\w+", re.UNICODE)
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


def _remover_stopwords(texto, palavras):
    if not palavras:
        return texto
    return RE_PALAVRA.sub(
        lambda m: "" if m.group(0).lower() in palavras else m.group(0), texto)


def limpar_texto(texto, *, corrigir_codificacao=False, remover_html=False,
                 links=MANTER, emails=MANTER, mencoes=MANTER,
                 hashtags=HASHTAG_MANTER, emojis=EMOJI_MANTER,
                 numeros=False, stopwords=None, repeticoes=False,
                 max_repeticoes=3, minusculas=False, acentos=False,
                 pontuacao=False, espacos=False):
    """Aplica as limpezas escolhidas, numa ordem que evita conflitos
    (ex.: e-mails antes de menções, pontuação depois dos marcadores).

    `stopwords`: conjunto de palavras a remover (já resolvido com
    resolver_stopwords()), ou None/vazio para não remover nenhuma.
    """
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
    if stopwords:
        t = _remover_stopwords(t, stopwords)
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
