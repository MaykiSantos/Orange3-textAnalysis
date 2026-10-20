#!/usr/bin/env python
"""Só existe para empacotar a documentação HTML em `data_files` (usada pelo
botão "Help" dos widgets no Orange): `data_files` é um recurso legado do
setuptools que não tem equivalente em `pyproject.toml`. Todo o resto da
metadata do pacote (nome, versão, dependências, entry points etc.) vive em
`pyproject.toml` — não duplique aqui.
"""
import sys
from os import path, walk

from setuptools import setup

DATA_FILES = []


def include_documentation(local_dir, install_dir):
    global DATA_FILES
    if 'bdist_wheel' in sys.argv and not path.exists(local_dir):
        print("Directory '{}' does not exist. "
              "Please build documentation before running bdist_wheel."
              .format(path.abspath(local_dir)))
        sys.exit(0)

    doc_files = []
    for dirpath, dirs, files in walk(local_dir):
        doc_files.append((dirpath.replace(local_dir, install_dir),
                          [path.join(dirpath, f) for f in files]))
    DATA_FILES.extend(doc_files)


if __name__ == '__main__':
    include_documentation('doc/_build/html', 'help/orange3-text-analysis')
    setup(data_files=DATA_FILES)
