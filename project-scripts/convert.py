import os
import jupytext
import jupytext.config

envfiles = os.getenv("QUARTO_PROJECT_OUTPUT_FILES")

files = [f for f in envfiles.split("\n") if f.endswith(".ipynb") and not f.endswith("-sol.ipynb")]

config = jupytext.config.load_jupytext_configuration_file("jupytext.toml")

for file in files:
    # solfile = file.replace(".ipynb", "-sol.ipynb")
    # if os.path.exists(solfile):
    #     os.remove(solfile)
    # os.rename(file, solfile)
    # solnb = jupytext.read(solfile)
    # nb = solnb.copy()
    # cells = [ c for c in nb.cells if not "solution" in c.metadata.get("tags", []) ]
    # nb.cells = cells
    # nbformat.write(nb, file)
    nb = jupytext.read(file)
    pyfile = file.replace(".ipynb", ".py")
    jupytext.write(nb, pyfile, fmt="py:percent", config=config)