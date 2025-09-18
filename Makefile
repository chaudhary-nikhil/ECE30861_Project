PYFILES := $(git ls-files *.py)

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
	@mypy --strict "$(PYFILES)" # Update the filepaths if there is code 
	@			           		# that should be exempt from linting
