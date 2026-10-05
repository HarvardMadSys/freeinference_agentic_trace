PYTHON = .venv/bin/python

# pip, pytest and every step write their temporary files on the data home's
# disk, which has the room for them.
export TMPDIR ?= $(CURDIR)/data/export/tmp
export TIKTOKEN_CACHE_DIR ?= $(CURDIR)/data/export/tmp/tiktoken

.PHONY: setup test weeks figures

# Create the virtual environment and install the package with its test tools.
setup:
	python3.12 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e '.[huggingface,test]'

# The maintainer's targets, test, week-<sunday> and weeks, need the tests and
# the weekly logs, which are not published. A reader needs setup and figures.
test:
	$(PYTHON) -m pytest

# Build one week from its weekly log and release it:  make week-2026-08-30
week-%:
	$(PYTHON) -m export build --week $*
	$(PYTHON) -m export release --week $*

# Build and release several weeks, one after another:  make weeks WEEKS="2026-08-23 2026-08-30"
weeks: $(addprefix week-,$(WEEKS))

# Simulate every released week not yet simulated from its current tables, then
# draw the paper's tables and figures from data/v1 into figures/v1, one folder
# per section. A large machine simulates faster with:  make figures PROCESSES=32
figures:
	$(PYTHON) -m simulation $(if $(PROCESSES),--processes $(PROCESSES))
	$(PYTHON) -m paper
