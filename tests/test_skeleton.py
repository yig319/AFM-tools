from pathlib import Path


def test_readme_mentions_package_name():
    readme = Path(__file__).resolve().parents[1] / "README.rst"
    content = readme.read_text(encoding="utf-8")
    assert "AFM-tools" in content
