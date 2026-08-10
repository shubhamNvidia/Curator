# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The instructions each host reads are one file, discoverable, and shippable.

Three hosts read the audio agent's instructions from three different places, and every
one of those failures is silent: a skill missing its ``name`` is skipped without a
message, a shim that git turned into a text file makes the skill simply not exist, and a
`.py` helper under a non-package directory is absent from the wheel while working
perfectly from a checkout. None of that shows up in a run; it shows up as an agent that
quietly does not know the procedure.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from nemo_curator.audio_agent import available_skills, skills_dir

_REPO = Path(__file__).resolve().parents[2]
_SKILLS = Path(skills_dir())

# Shim -> the packaged skill it must resolve to. Codex and Cursor read ``.agents/skills``;
# Claude Code reads only ``.claude/skills``, hence the one entry in a directory upstream
# also uses.
_SHIMS = {
    ".agents/skills/audio-curation": "audio-curation",
    ".agents/skills/audio-stage-authoring": "audio-stage-authoring",
    ".claude/skills/audio-curation": "audio-curation",
}

# Codex truncates a description past this, and a truncated one loses its trigger clause.
_MAX_DESCRIPTION = 500

# What MANIFEST.in's ``recursive-include nemo_curator`` actually names. A suffix outside
# this set exists in a checkout and is missing from the wheel.
_PACKAGED_SUFFIXES = {".csv", ".json", ".yaml", ".yml", ".txt", ".md"}


def _frontmatter(path: Path) -> dict[str, str]:
    """The YAML frontmatter block as a flat mapping, or ``{}`` when there is none."""
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        return {}
    import yaml

    loaded = yaml.safe_load(match.group(1))
    return loaded if isinstance(loaded, dict) else {}


@pytest.fixture(scope="module")
def cli_verbs() -> set[str]:
    from nemo_curator.audio_agent.cli import build_parser

    parser = build_parser()
    return {
        name
        for action in parser._subparsers._group_actions
        for name in getattr(action, "choices", {})
    }


def test_the_packaged_skills_are_the_ones_we_expect() -> None:
    assert available_skills() == ["audio-curation", "audio-stage-authoring"]


@pytest.mark.parametrize("skill", available_skills())
def test_every_skill_declares_a_name_matching_its_folder(skill: str) -> None:
    """Cursor and Codex both require ``name``, and Codex requires it to equal the folder.

    The first version of this skill shipped with only a ``description``, which Claude Code
    tolerates and the other two do not.
    """
    meta = _frontmatter(_SKILLS / skill / "SKILL.md")
    assert meta.get("name") == skill, f"{skill}: frontmatter name must equal the folder name"
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", skill), f"{skill}: folder must be kebab-case"


@pytest.mark.parametrize("skill", available_skills())
def test_every_skill_description_survives_the_strictest_host(skill: str) -> None:
    """One line, under 500 chars, and it must say when to use the skill.

    The description is the whole routing signal: it is all a host loads before deciding
    whether to open the body, so a description that only says what the skill *is* leaves
    the host no basis to pick it.
    """
    meta = _frontmatter(_SKILLS / skill / "SKILL.md")
    description = meta.get("description", "")
    assert description, f"{skill}: no description"
    assert "\n" not in description, f"{skill}: description must be a single line"
    assert len(description) <= _MAX_DESCRIPTION, (
        f"{skill}: description is {len(description)} chars, over the {_MAX_DESCRIPTION} limit"
    )
    assert "use " in description.lower(), (
        f"{skill}: description must name its trigger conditions, not only what it is"
    )


@pytest.mark.parametrize(("shim", "skill"), sorted(_SHIMS.items()))
def test_every_host_shim_resolves_into_the_package(shim: str, skill: str) -> None:
    """Existing is not enough: it has to resolve to the packaged file.

    On a Windows checkout without ``core.symlinks``, git writes each link as a plain text
    file holding its target path. Every host then finds no skill, with no error anywhere.
    """
    path = _REPO / shim
    assert path.is_dir(), f"{shim}: missing, or a text file from a checkout without symlink support"
    for entry in sorted(p.name for p in (_SKILLS / skill).iterdir()):
        linked = path / entry
        assert linked.exists(), f"{shim}/{entry}: missing or dangling"
        assert linked.resolve() == (_SKILLS / skill / entry).resolve(), (
            f"{shim}/{entry}: resolves to {linked.resolve()}, not the packaged {skill}"
        )


@pytest.mark.parametrize("shim", sorted(_SHIMS))
def test_a_walk_that_does_not_follow_symlinks_still_finds_the_shim(shim: str) -> None:
    """The shim's own directory has to be real, not a link to the packaged one.

    A walker that does not follow symlinks -- a common default, and what a ripgrep-backed
    file search does without ``--follow`` -- lists nothing whatsoever inside a symlinked
    directory. Linking the directory therefore hides the skill from any host that discovers
    by walking, and hides it silently. Linking the entries into a real directory does not:
    to such a walk they are ordinary entries.
    """
    path = _REPO / shim
    assert not path.is_symlink(), f"{shim}: the skill directory itself must not be a symlink"
    found = [f for _d, _sub, files in os.walk(path) for f in files if f == "SKILL.md"]
    assert found == ["SKILL.md"], f"{shim}: a non-following walk finds no SKILL.md"


def test_no_skill_ships_a_scripts_directory() -> None:
    """``scripts/`` is part of the skill standard but not of this package.

    MANIFEST.in takes documents, not code, from a non-package directory, so a script here
    would work from a checkout and be absent for every pip user. The deterministic verbs
    are the script layer.
    """
    assert not [p for p in _SKILLS.rglob("scripts") if p.is_dir()]


def test_every_file_under_skills_is_one_the_wheel_carries() -> None:
    for path in _SKILLS.rglob("*"):
        if path.is_file():
            assert path.suffix in _PACKAGED_SUFFIXES, (
                f"{path.relative_to(_SKILLS)}: MANIFEST.in does not include {path.suffix} files, "
                "so this exists in a checkout and is missing from the wheel"
            )


def test_every_verb_the_instructions_mention_exists(cli_verbs: set[str]) -> None:
    """Catches a renamed verb leaving stale instructions behind.

    An agent following a document that names a verb the CLI dropped gets a usage error at
    the step it was told to run, which reads as a broken tool rather than a stale document.
    """
    invocation = re.compile(
        r"(?:-m nemo_curator\.audio_agent|nemo-curator-audio)\s+(?:\.\.\.\s+)?([a-z][a-z-]+)"
    )
    docs = [
        *sorted(_SKILLS.rglob("*.md")),
        _REPO / "nemo_curator/audio_agent/AGENTS.md",
        _REPO / "nemo_curator/stages/audio/AGENTS.md",
    ]
    stale: dict[str, set[str]] = {}
    for doc in docs:
        mentioned = set(invocation.findall(doc.read_text(encoding="utf-8")))
        missing = {v for v in mentioned if v not in cli_verbs}
        if missing:
            stale[str(doc.relative_to(_REPO))] = missing
    assert not stale, f"instructions name verbs the CLI does not have: {stale}"


@pytest.mark.parametrize(
    "directory", ["nemo_curator/audio_agent", "nemo_curator/stages/audio"]
)
def test_every_claude_md_is_an_import_of_its_sibling_agents_md(directory: str) -> None:
    """The pair used to be two hand-maintained copies differing only in the word "Claude".

    Claude Code resolves ``@AGENTS.md``, so one line keeps both hosts on the same text and
    makes divergence impossible rather than merely unlikely.
    """
    base = _REPO / directory
    assert (base / "AGENTS.md").is_file(), f"{directory}: no AGENTS.md to import"
    lines = [ln.strip() for ln in (base / "CLAUDE.md").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines == ["@AGENTS.md"], (
        f"{directory}/CLAUDE.md must be exactly '@AGENTS.md'; found {lines!r}"
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git to read the ignore rules")
@pytest.mark.parametrize(
    "relative",
    [
        "nemo_curator/audio_agent/AGENTS.md",
        "nemo_curator/audio_agent/CLAUDE.md",
        "nemo_curator/stages/audio/AGENTS.md",
        "nemo_curator/stages/audio/CLAUDE.md",
        ".agents/skills/audio-curation/SKILL.md",
        ".claude/skills/audio-curation/SKILL.md",
    ],
)
def test_the_instructions_are_not_swallowed_by_gitignore(relative: str) -> None:
    """``.gitignore`` ignores ``AGENTS.md`` and ``CLAUDE.md`` repo-wide, by design.

    The audio agent's four are negated back in, because they are part of the tool surface
    rather than someone's scratch notes. Without the negation the file works locally and in
    a locally built wheel, and is missing from every clone -- visible to nobody who did not
    write it.
    """
    listed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", relative],  # noqa: S607
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode != 0:
        pytest.skip("not a git checkout")
    assert listed.stdout.strip(), f"{relative} is git-ignored, so no clone would have it"


def test_the_audio_rules_stay_out_of_the_shared_cursor_directory() -> None:
    """Merge-safety guard: ``.cursor/rules/`` is upstream's, and audio content left there
    interleaves the fork with files that serve text, image and video work.

    The workflow lives in the nested ``AGENTS.md`` files, which auto-attach on the same
    trigger the glob-scoped rule used and are read by all three hosts rather than one.
    """
    rules = {p.name for p in (_REPO / ".cursor/rules").glob("*.mdc")}
    assert not {"audio-agent.mdc", "use-audio-agent.mdc", "no-repo-edits-for-use-cases.mdc"} & rules
    audio_specific = sorted(
        p.name
        for p in (_REPO / ".cursor/rules").glob("*.mdc")
        if "audio_agent" in p.read_text(encoding="utf-8")
    )
    assert not audio_specific, (
        f"audio-agent guidance belongs in nemo_curator/audio_agent/AGENTS.md, not in {audio_specific}"
    )


def test_our_only_footprint_in_the_shared_claude_directory_is_one_shim() -> None:
    """Everything else lives in fork-owned paths, so a merge sees a new filename at worst.

    ``getting-started/`` and ``nemo-curator-docs/`` arrived from upstream and serve every
    modality; editing either guarantees a conflict on a file that carries no audio value.
    """
    entries = sorted(p.name for p in (_REPO / ".claude/skills").iterdir())
    ours = [name for name in entries if name.startswith("audio")]
    assert ours == ["audio-curation"], f"unexpected fork-owned entries in .claude/skills: {ours}"
    assert (_REPO / ".claude/skills/audio-curation/SKILL.md").is_symlink(), (
        "the audio skill must be a shim into the package, not a second copy to maintain"
    )


def test_install_skill_reaches_a_directory_each_host_scans(tmp_path: Path) -> None:
    """A pip user has the instructions in site-packages, where no host looks."""
    from nemo_curator.audio_agent import install_skill

    result = install_skill(dest=str(tmp_path), mode="copy")
    assert result["status"] == "ok", result
    for relative in (".agents/skills", ".claude/skills"):
        assert (tmp_path / relative / "audio-curation" / "SKILL.md").is_file()
    assert (tmp_path / ".agents/skills/audio-curation/references").is_dir()

    # Idempotent: the second pass writes nothing, so a user re-running it after an upgrade
    # cannot be told files changed when they did not.
    again = install_skill(dest=str(tmp_path), mode="copy")
    assert {e["action"] for e in again["installed"]} == {"unchanged"}


def test_install_skill_refuses_to_overwrite_someone_elses_skill(tmp_path: Path) -> None:
    from nemo_curator.audio_agent import install_skill

    target = tmp_path / ".agents/skills/audio-curation"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text("a skill someone wrote by hand", encoding="utf-8")

    refused = install_skill(dest=str(tmp_path), host="codex", skills=["audio-curation"])
    assert refused["status"] == "error"
    assert [e["action"] for e in refused["installed"]] == ["conflict"]
    assert (target / "SKILL.md").read_text(encoding="utf-8") == "a skill someone wrote by hand"

    forced = install_skill(dest=str(tmp_path), host="codex", skills=["audio-curation"], force=True)
    assert forced["status"] == "ok"
    assert (target / "SKILL.md").read_text(encoding="utf-8").startswith("---")


def test_a_symlink_install_stays_discoverable_and_idempotent(tmp_path: Path) -> None:
    """``--symlink`` links the entries, not the directory, for the same discovery reason."""
    from nemo_curator.audio_agent import install_skill

    result = install_skill(dest=str(tmp_path), host="codex", mode="symlink")
    assert result["status"] == "ok", result
    installed = tmp_path / ".agents/skills/audio-curation"
    assert not installed.is_symlink()
    assert installed.joinpath("SKILL.md").is_symlink()
    assert [f for _d, _s, files in os.walk(installed) for f in files if f == "SKILL.md"] == ["SKILL.md"]

    again = install_skill(dest=str(tmp_path), host="codex", mode="symlink")
    assert {e["action"] for e in again["installed"]} == {"unchanged"}

    # Switching mode relays the same content rather than reporting a conflict over it.
    switched = install_skill(dest=str(tmp_path), host="codex", mode="copy")
    assert {e["action"] for e in switched["installed"]} == {"replaced"}
    assert not installed.joinpath("SKILL.md").is_symlink()


def test_install_skill_repairs_the_shims_a_windows_checkout_leaves(tmp_path: Path) -> None:
    """The Windows failure mode, which needs a repair rather than a force flag.

    Without ``core.symlinks`` each committed link becomes a text file holding its target
    path. Refusing that as a conflict would tell a Windows user to force-overwrite something
    that was never real content in the first place.
    """
    from nemo_curator.audio_agent import install_skill

    shim = tmp_path / ".claude/skills/audio-curation"
    shim.mkdir(parents=True)
    prefix = "../../../nemo_curator/audio_agent/skills/audio-curation"
    (shim / "SKILL.md").write_text(f"{prefix}/SKILL.md", encoding="utf-8")
    (shim / "references").write_text(f"{prefix}/references", encoding="utf-8")

    result = install_skill(dest=str(tmp_path), host="claude", skills=["audio-curation"])
    assert result["status"] == "ok"
    assert [e["action"] for e in result["installed"]] == ["repaired"]
    assert (shim / "SKILL.md").read_text(encoding="utf-8").startswith("---")
    assert (shim / "references").is_dir()


def test_install_skill_repairs_an_older_whole_directory_shim(tmp_path: Path) -> None:
    """A link to the whole directory reads fine but hides the skill from a plain walk."""
    from nemo_curator.audio_agent import install_skill

    shim = tmp_path / ".claude/skills/audio-curation"
    shim.parent.mkdir(parents=True)
    shim.symlink_to(_SKILLS / "audio-curation", target_is_directory=True)

    result = install_skill(dest=str(tmp_path), host="claude", skills=["audio-curation"], mode="symlink")
    assert result["status"] == "ok"
    assert [e["action"] for e in result["installed"]] == ["replaced"]
    assert not shim.is_symlink()
    assert (shim / "SKILL.md").is_symlink()


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git to stage a source tree")
def test_a_built_wheel_carries_the_skills(tmp_path: Path) -> None:
    """The claim "it ships automatically" is only true until someone adds a file type.

    Built from a copy so the build directory never lands in the working tree.
    """
    pytest.importorskip("setuptools")
    source = tmp_path / "src"
    source.mkdir()
    for name in ("pyproject.toml", "MANIFEST.in", "LICENSE", "README.md"):
        if (_REPO / name).exists():
            shutil.copy(_REPO / name, source / name)
    (source / "nemo_curator").symlink_to(_REPO / "nemo_curator", target_is_directory=True)

    out = tmp_path / "dist"
    out.mkdir()
    built = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            sys.executable,
            "-c",
            f"from setuptools import build_meta; print(build_meta.build_wheel({str(out)!r}))",
        ],
        cwd=source,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "SETUPTOOLS_SCM_PRETEND_VERSION": "0.0.0"},
    )
    if built.returncode != 0:
        pytest.skip(f"wheel build unavailable in this environment: {built.stderr[-400:]}")

    wheels = list(out.glob("*.whl"))
    assert wheels, "no wheel produced"
    names = zipfile.ZipFile(wheels[0]).namelist()
    packaged = {n for n in names if "/skills/" in n}
    expected = {
        f"nemo_curator/audio_agent/skills/{p.relative_to(_SKILLS).as_posix()}"
        for p in _SKILLS.rglob("*")
        if p.is_file()
    }
    assert expected <= packaged, f"missing from the wheel: {sorted(expected - packaged)}"
    assert any("references/" in n for n in packaged), "progressive-disclosure references did not ship"
    assert "nemo_curator/audio_agent/AGENTS.md" in names
