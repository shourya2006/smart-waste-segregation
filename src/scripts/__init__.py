import pathlib
import sys

# Add the top-level 'scripts' directory to sys.path and __path__ for submodule imports.
_scripts_path = pathlib.Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_scripts_path) not in sys.path:
    sys.path.insert(0, str(_scripts_path))
# Extend the package __path__ so that submodules are found in the external scripts directory.
__path__.append(str(_scripts_path))
