#!/usr/bin/env python

from os import path, walk

import sys
from setuptools import setup, find_packages

NAME = "Orange3 text analises add-on"

VERSION = "0.0.1"

AUTHOR = 'Mayki dos Santos Oliveira, Brasil BR'
AUTHOR_EMAIL = 'maykioliveira97@gmail.com'

URL = 'http://orange.biolab.si/download'
DESCRIPTION = "The add-on contains a set of tools to assist with data analysis using models from huggingface.com."
LONG_DESCRIPTION = open(path.join(path.dirname(__file__), 'README.md'),
                        'r', encoding='utf-8').read()

LICENSE = "GNU"

KEYWORDS = [
    'orange3 add-on',
    'text mining',
    'text analysis',
    'textable',
    'orange3'
]

PACKAGES = find_packages()

PACKAGE_DATA = {
    'text_analysis': ['widgets/icons/*']
}

DATA_FILES = []

INSTALL_REQUIRES = [
    'Orange3',
    "transformers",
    "torch",
    "sentencepiece",
    "protobuf",
    "ftfy",
    "emoji"
]

ENTRY_POINTS = {
    'orange3.addon': (
        'ta = text_analysis',
    ),
    'orange.widgets': (
        'ta = text_analysis.widgets',
    ),
}

# TEST_SUITE = "text_analysis.tests.suite"


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
    setup(
        name=NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        license=LICENSE,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        data_files=DATA_FILES,
        install_requires=INSTALL_REQUIRES,
        entry_points=ENTRY_POINTS,
        keywords=KEYWORDS,
        zip_safe=False,
    )