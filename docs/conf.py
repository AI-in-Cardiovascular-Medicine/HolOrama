import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../src'))


def get_version():
    ns = {}
    exec(Path('../src/version.py').read_text(), ns)
    return ns.get('__version__', '0.0.0')


project = 'AIVUS-CAA'
copyright = '2025, AI-in-Cardiovascular-Medicine'
author = 'Anselm W. Stark'
release = get_version()
version = '.'.join(release.split('.')[:2])

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

_here = os.path.abspath(os.path.dirname(__file__))
templates_path = [os.path.join(_here, '_templates')]
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

myst_heading_anchors = 3

html_theme = 'sphinx_rtd_theme'
html_title = 'AIVUS-CAA Documentation'
html_theme_options = {
    'navigation_depth': 3,
    'collapse_navigation': False,
}

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
}
autodoc_typehints = 'description'
numfig = True
