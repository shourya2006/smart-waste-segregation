# This package forwards imports to the src layout
import pathlib
import sys
# Add the src/waste_sorting directory to the package search path
_package_path = pathlib.Path(__file__).resolve().parent.parent / "src" / "waste_sorting"
if str(_package_path) not in sys.path:
    sys.path.insert(0, str(_package_path))
# Extend __path__ to include the src location for namespace package behavior
__path__ = [str(_package_path)]
