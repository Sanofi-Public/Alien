# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from shutil import copyfile

import tomli

sys.path.insert(0, os.path.abspath(".."))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.confluencebuilder",
]

confluence_publish = True
confluence_server_url = os.environ.get("CONFLUENCE_SERVER")
confluence_space_key = os.environ.get("CONFLUENCE_SPACE_KEY")
confluence_server_user = os.environ.get("CONFLUENCE_USER")
confluence_server_pass = os.environ.get("CONFLUENCE_USER_KEY")
confluence_parent_page = "Unstructured data search"

#print(os.getcwd())

#def setup(app):
#    try:
#        if not any(['sanofi.css' in fn for fn in os.listdir('./build/html/_static')]):
#            copyfile('sanofi.css', 'build/html/_static/sanofi.css')
#    except:
#        copyfile('sanofi.css', 'build/html/_static/sanofi.css')
#    app.add_css_file('sanofi.css')

def _get_project_meta():
    with open("../pyproject.toml", mode="rb") as pyproject:
        return tomli.load(pyproject)["tool"]["poetry"]


project = "ALIEN"
copyright = "2024"
author = "Michael Bailey"

pkg_meta = _get_project_meta()

# The short X.Y version
version = str(pkg_meta["version"])
# The full version, including alpha/beta/rc tags
release = version


templates_path = ["_templates"]
exclude_patterns = ["_build", "_templates", "Thumbs.db", ".DS_Store"]

autodoc_inherit_docstrings = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "inherited-members": True,
}

default_role = "obj"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_logo = "../media/curio_small.png"

# html_static_path = ['_static']
