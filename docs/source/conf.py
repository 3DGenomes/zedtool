# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import subprocess
print(f"Running Sphinx configuration from {os.path.abspath(__file__)} with cwd {os.getcwd()}")
package_name = "zedtool"
# Get package dir from os.path.abspath(__file__)
doc_source_path = os.path.abspath(os.path.dirname(__file__))
package_root = os.path.abspath(os.path.join(doc_source_path, "..", ".."))
package_src = os.path.join(package_root, "src")
package_path = os.path.join(package_src, package_name)
# Add your package root to sys.path so Sphinx can import it if needed
sys.path.insert(0, package_src)

project = package_name
author = 'John Markham'
copyright = f'2025, {author}'

# Read version from VERSION file
version_file = os.path.join(package_path,"VERSION")
print(f"Reading version from {version_file}")
with open(version_file, "r") as f:
    release = f.read().strip()

# Short X.Y version for headers
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # if you want to generate API docs automatically
    "sphinx.ext.napoleon",  # Optional: supports Google/NumPy docstrings
    "myst_parser" # for markdown support
]

myst_enable_extensions = [
    "deflist",
    "html_admonition",
    "html_image",
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'sphinx_rtd_theme'
# html_theme = 'alabaster'
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navbar_end": ["search-field.html"],
}
html_static_path = ['_static']

# -- Auto-generate API docs using sphinx-apidoc ------------------------------
apidoc_output = os.path.join(package_root, "docs", "source")
submodules_rst = os.path.join(apidoc_output, f"{package_name}.rst")
print(f"Checking for existing {submodules_rst}")

if not os.path.exists(submodules_rst):
    print(f"Generating API docs in {apidoc_output} from {package_src}")
    # Only generate if not already present to avoid overwriting local changes
    subprocess.run([
        "sphinx-apidoc",
        "-o", apidoc_output,
        package_path,
        "--force",
        "--module-first",
        "--no-toc",
    ])