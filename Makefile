PYFILES := $(shell git ls-files *.py)

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
	python3 -m pytest

# Autograder targets
register:
	@curl --location 'http://dl-berlin.ecn.purdue.edu/api/register' \
	--header 'Content-Type: application/json' \
	--data "{\"group\": 27,\"github\": \"https://github.com/chaudhary-nikhil/ECE30861_Project.git\",\"names\": [\"Ryan Baker\",\"Nikhil Chaudhary\",\"Aadhavan Srinivasan\",\"Luisa Cruz Miotto\"],\"gh_token\": \"$(GH_TOKEN)\"}"

schedule:
	@curl --location 'http://dl-berlin.ecn.purdue.edu/api/schedule' \
	--header 'Content-Type: application/json' \
	--data '{"group": 27,"gh_token": "$(GH_TOKEN)"}'

monitor:
	@curl --location --request GET 'http://dl-berlin.ecn.purdue.edu/api/run/all' \
	--header 'Content-Type: application/json' \
	--data '{"group": 27,"gh_token": "$(GH_TOKEN)"}'

check-run:
	@curl --location --request GET 'http://dl-berlin.ecn.purdue.edu/api/last_run' \
	--header 'Content-Type: application/json' \
	--data '{"group": 27,"gh_token": "$(GH_TOKEN)"}'

check-best-run:
	@curl --location --request GET 'http://dl-berlin.ecn.purdue.edu/api/best_run' \
	--header 'Content-Type: application/json' \
	--data '{"group": 27,"gh_token": "$(GH_TOKEN)"}'

.PHONY: test register schedule monitor check-run check-best-run
