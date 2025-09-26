PYFILES := $(shell git ls-files *.py)

GROUP = 27
SCRIPT := scripts/autograder.py
PYTHON := python3

format-check:
	@if [ -n "$(PYFILES)" ]; then 			    \
		ruff format --quiet --check $(PYFILES); \
	fi

format-all:
	@if git diff --quiet; then :; else 								\
		echo "There are unstaged changes that may be overwritten."; \
		read -p "Would you like to continue anyway? [Y/n] " ans; 	\
		case $$ans in 												\
			[yY]*) ;; 												\
			*) echo "Aborted."; exit 1 ;; 							\
		esac; 														\
	fi
	@if [ -n "$(PYFILES)" ]; then \
		ruff format $(PYFILES);	  \
	fi

lint-check:
	@mypy --strict $(PYFILES) > /dev/null || (mypy --strict $(PYFILES) && exit 1)
	@pyright $(PYFILES) > /dev/null || (pyright $(PYFILES) && exit 1)

test:
	python3 -m pytest --cov=src

check-token:
	@if [ -z "$(GH_TOKEN)" ]; then \
		echo "Error: GH_TOKEN environment variable is not set"; \
		echo "Please set it with: export GH_TOKEN='your_token_here'"; \
		exit 1; \
	fi

check-script:
	@if [ ! -f "$(SCRIPT)" ]; then \
		echo "Error: $(SCRIPT) not found"; \
		echo "Please create the Python script first"; \
		exit 1; \
	fi

run: check-token check-script
	@$(PYTHON) $(SCRIPT)

schedule: check-token check-script
	@$(PYTHON) $(SCRIPT) --schedule

monitor: check-token check-script
	@$(PYTHON) $(SCRIPT) --monitor

best: check-token check-script
	@$(PYTHON) $(SCRIPT) --best

logs: check-token check-script
	@$(PYTHON) $(SCRIPT) --logs

auto: check-token check-script
	@$(PYTHON) $(SCRIPT) --auto

help:
	@echo "Autograder Makefile Targets:"
	@echo ""
	@echo "  make run       - Interactive menu (default)"
	@echo "  make schedule  - Schedule a new autograder run"
	@echo "  make monitor   - Check status of all runs"
	@echo "  make best      - Get results from the best run"
	@echo "  make logs      - Download logs from the best run"
	@echo "  make auto      - Full workflow (schedule + monitor + fetch + logs)"
	@echo "  make test      - Run the test suite with debugging output"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  Group: $(GROUP)"
	@echo ""
	@echo "Note: GH_TOKEN environment variable must be set"
	@echo "      export GH_TOKEN='your_token_here'"

.PHONY: check-token check-script run schedule monitor best logs auto help test
.DEFAULT_GOAL := run
