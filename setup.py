import json
from glob import glob
from os.path import basename, dirname
from os.path import splitext

from setuptools import setup
from setuptools import find_packages

import financialml


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name="financialml",
    version=financialml.__version__,
    packages=['financialml'],  # import可能な名前空間を指定
    package_dir={'financialml': 'financialml'},  # 名前空間とディレクトリsrcの対応
    install_requires=_requires_from_file('requirements.txt'),
)
