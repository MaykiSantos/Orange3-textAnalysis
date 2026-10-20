# Desenvolvimento

Este documento reúne informações para quem vai contribuir com o código do
add-on. Não é necessário para apenas instalar e usar os widgets — veja o
[README](../README.md) principal para isso.

## Instalação a partir do código-fonte

O add-on deve ser instalado **no mesmo Python que o Orange usa**.

**Pelo terminal**, dentro da pasta do projeto (troque `python` pelo
executável do Python do Orange, se for outro):

```bash
python -m pip install .
```

Para desenvolvimento, instale em modo editável (as alterações no código
valem após reiniciar o Orange, sem reinstalar):

```bash
python -m pip install -e .
```

**Pelo próprio Orange**, colando no widget *Python Script* e clicando em
*Run* (ajuste o caminho da pasta):

```python
import sys, subprocess
pasta = "/caminho/para/orange3-text-analysis"
r = subprocess.run([sys.executable, "-m", "pip", "install", pasta],
                   capture_output=True, text=True)
print(r.stdout[-1500:]); print(r.stderr[-1500:])
```

Depois, **reinicie o Orange**.

## Bibliotecas Python

Instaladas automaticamente junto com o add-on:

| Biblioteca | Para que é usada |
|---|---|
| `transformers` | Carregar e executar os modelos do Hugging Face |
| `torch` | Motor de execução dos modelos (CPU ou GPU) |
| `sentence-transformers` | Geração de embeddings de sentença |
| `sentencepiece` | Tokenizadores de modelos multilíngues (XLM-RoBERTa, T5, M2M100) |
| `protobuf` | Leitura dos arquivos de tokenizador SentencePiece |
| `ftfy` | Correção de codificação quebrada (ex.: `pÃ©ssimo` → `péssimo`) |
| `emoji` | Remoção de emojis e conversão em texto em português |

## Estrutura do projeto

```
orange3-text-analysis/
├── README.md
├── pyproject.toml
├── doc/                    # documentação de cada widget + imagens
├── tests/
└── text_analysis/
    ├── modelo.py           # adaptador Hugging Face → modelo do Orange
    ├── limpeza.py          # funções de limpeza de texto
    ├── traducao.py         # funções de tradução
    ├── embeddings.py       # geração de embeddings de sentença
    ├── topicos.py          # adaptador zero-shot (NLI) → modelo do Orange
    ├── topicos_bertopic.py # extração de tópicos com BERTopic
    ├── stopwords/          # listas de stopwords (português, inglês)
    └── widgets/
        ├── owlimpezatexto.py
        ├── owtraducao.py
        ├── owsentimentohf.py
        ├── owtopicos.py          # Zero-shot Classification
        ├── owembeddingstexto.py
        ├── owtopicosbertopic.py  # Extração de Tópicos (BERT)
        └── icons/
```
