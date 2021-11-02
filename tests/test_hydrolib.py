from hydrolib.post import __version__
from hydrolib.utils import square

def test_version():
    assert __version__ == "0.1.0"


def test_namespace():
    # Make sure we can access hydrolib-core package
    from hydrolib.core import __version__

    assert __version__ == "0.1.5"

def test_square():
    assert square(2) == 4
