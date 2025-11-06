import sys

sys.path.append(".")
from hydrolib.post import __version__


def test_version():
    assert __version__ == "0.5.0"


def test_namespace():
    # Make sure we can access hydrolib-core package
    from hydrolib.core import __name__ as hc_name

    assert hc_name == "hydrolib.core"
