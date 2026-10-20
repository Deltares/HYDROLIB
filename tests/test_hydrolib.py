from pathlib import Path
import tomllib

def _pyproject_version():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    version = (
        data.get("project", {}).get("version")
        or data.get("tool", {}).get("poetry", {}).get("version")
    )
    assert version is not None, "no version found in pyproject.toml"
    return version

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
