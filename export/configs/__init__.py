"""The public rule tables, loaded once for a command: the TOML files beside this module, and what each decides.

- `harnesses.toml`: which harness sent a request, by its phrases and user agent;
- `tool_names.toml` and `tool_categories.toml`: a tool call's canonical name,
  family, shell purpose and category;
- `outcome_rules.toml`: whether a call's result says ok, error, denied or timeout;
- `public_tool_words.toml`: the tool words that ship, since enough accounts used them;
- `compaction_notes.toml`: the phrases a harness writes when it compacts a context;
- `exclusions.toml`: the harnesses and sessions the release leaves out, and why.
"""

import dataclasses
import pathlib
import tomllib

from export.privacy import public_words
from export.log_records import harness
from export.release import exclusions
from export.tool_calls import categories, outcomes

FOLDER = pathlib.Path(__file__).resolve().parent
HARNESS_TABLE = FOLDER / "harnesses.toml"
TOOL_NAMES = FOLDER / "tool_names.toml"
TOOL_CATEGORIES = FOLDER / "tool_categories.toml"
OUTCOME_RULES = FOLDER / "outcome_rules.toml"
PUBLIC_TOOL_WORDS = FOLDER / "public_tool_words.toml"
COMPACTION_NOTES = FOLDER / "compaction_notes.toml"
EXCLUSIONS = FOLDER / "exclusions.toml"


@dataclasses.dataclass(frozen=True)
class RuleTables:
    """Every public rule table, loaded."""

    harnesses: tuple[harness.Harness, ...]
    tool_tables: categories.ToolTables
    outcome_rules: outcomes.OutcomeRules
    public_words: public_words.PublicWords
    compaction_notes: tuple[str, ...]
    exclusions: exclusions.Exclusions

    @property
    def harness_names(self) -> frozenset[str]:
        """The name of every harness the table lists."""
        return frozenset(entry.name for entry in self.harnesses)


def load_compaction_notes(path: pathlib.Path) -> tuple[str, ...]:
    """The phrases of `compaction_notes.toml`."""
    with open(path, "rb") as handle:
        return tuple(note["phrase"] for note in tomllib.load(handle)["note"])


def load() -> RuleTables:
    """The committed rule tables."""
    return RuleTables(
        harnesses=harness.load_harness_table(HARNESS_TABLE),
        tool_tables=categories.load_tables(TOOL_NAMES, TOOL_CATEGORIES),
        outcome_rules=outcomes.load_rules(OUTCOME_RULES),
        public_words=public_words.load_public_words(PUBLIC_TOOL_WORDS),
        compaction_notes=load_compaction_notes(COMPACTION_NOTES),
        exclusions=exclusions.load_exclusions(EXCLUSIONS),
    )
