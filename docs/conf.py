# noqa: INP001

import importlib_metadata
import re
# -- Project information -----------------------------------------------------

project = "transformer-smaller-training-vocab"
copyright = "2023, Benedikt Fuchs"
author = "Benedikt Fuchs"

# The full version, including alpha/beta/rc tags
version = importlib_metadata.version(project)
release = importlib_metadata.version(project)
top_level = project.replace("-", "_")

# get the url on a hacky way, TODO: think of a better way
linkcode_url = importlib_metadata.metadata(project)["Project-URL"].split(" ")[-1]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",  # to render Google format docstrings
    "sphinx.ext.githubpages",
    "myst_parser",
    "sphinx_github_style",
    "sphinx_autodoc_typehints",
]

# Sphinxcontrib configuration
scv_priority = "tags"
scv_show_banner = True
scv_banner_greatest_tag = True
scv_sort = ("semver",)
scv_whitelist_branches = (re.compile("^(master|main)$"),)
# scv_whitelist_tags = ('None',)
scv_grm_exclude = ("README.md", ".gitignore", ".nojekyll", "CNAME")
scv_greatest_tag = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# Napoleon settings
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
