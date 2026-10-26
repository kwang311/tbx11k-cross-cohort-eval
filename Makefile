SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
PY := $(if $(PY_BIN),$(PY_BIN),python3)

.PHONY: help check audit stats figures train all

help:
	@echo "make check    # compile every Python source"
	@echo "make audit    # re-derive labels and metrics from the stored predictions"
	@echo "make stats    # bootstrap intervals and per-cell summaries"
	@echo "make figures  # redraw Figures 1 to 4"
	@echo "make train    # print the train commands (CUDA host required)"
	@echo "make all      # check + audit + stats + figures"

check:
	$(PY) -m py_compile $$(find code -name '*.py') audit.py

audit:
	$(PY) code/audit_tbx11k.py

stats:
	$(PY) code/stats_paper2.py
	$(PY) code/stats_cross_cells.py --prefix resnet50_full

figures:
	$(PY) code/figures/make_p2_fig1.py
	$(PY) code/figures/make_p2_fig2.py
	$(PY) code/figures/make_p2_fig3_active_vs_latent.py
	$(PY) code/figures/make_p2_fig4_cross_cohort.py

train:
	@echo "Training needs a CUDA host; see the command lines in README.md and the run_*.bat launchers."

all: check audit stats figures
