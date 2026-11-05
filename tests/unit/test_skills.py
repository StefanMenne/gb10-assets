"""The plugin's skills: their frontmatter must be valid YAML (GitHub and Claude Code parse it strictly)."""
import pytest
import yaml

from gb10_assets.server.config import REPO_ROOT

SKILLS = sorted((REPO_ROOT / "plugin" / "skills").glob("*/SKILL.md"))


@pytest.mark.parametrize("path", SKILLS, ids=[p.parent.name for p in SKILLS])
def test_frontmatter(path):
    _, front, _ = path.read_text().split("---", 2)
    meta = yaml.safe_load(front)
    assert meta["name"] == path.parent.name and meta["description"].strip()
