"""One version for everything: the Python package, the Claude Code plugin and the git tag v<version>."""
import json
import tomllib

from gb10_assets import __version__
from gb10_assets.server.config import REPO_ROOT


def test_package_and_plugin_have_the_same_version():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["version"]
    plugin = json.loads((REPO_ROOT / "plugin" / ".claude-plugin" / "plugin.json").read_text())["version"]
    assert project == plugin == __version__
