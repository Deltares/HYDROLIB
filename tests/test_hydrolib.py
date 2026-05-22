import re
from pathlib import Path


def _pyproject_version():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    match = re.search(
        r'^\[tool\.poetry\][\s\S]*?^version = "([^"]+)"',
        pyproject.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match is not None
    return match.group(1)


def test_version():
    from hydrolib.dhydamo import __version__ as dhydamo_version
    from hydrolib.post import __version__ as post_version

    pyproject_version = _pyproject_version()
    assert post_version == pyproject_version
    assert dhydamo_version == pyproject_version


def test_namespace():
    # Make sure we can access hydrolib-core package
    from hydrolib.core import __name__ as hc_name

    assert hc_name == "hydrolib.core"
