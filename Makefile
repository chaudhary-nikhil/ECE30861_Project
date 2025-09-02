CFILES  := $(shell find . -type f \( -name "*.c" -o -name "*.h" -o -name "*.cpp" -o -name "*.hpp" \))
PYFILES := $(shell find . -type f -name "*.py")
JSFILES := $(shell find . -type f \( -name "*.js" -o -name "*.ts" \))

format-check:
	@if [ -n "$(CFILES)" ]; then 				   \
		clang-format --dry-run --Werror $(CFILES); \
	fi
	@if [ -n "$(PYFILES)" ]; then 			    \
		ruff format --quiet --check $(PYFILES); \
	fi
	@if [ -n "$(JSFILES)" ]; then 		 \
		npx prettier --check $(JSFILES); \
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
	@if [ -n "$(CFILES)" ]; then   \
		clang-format -i $(CFILES); \
	fi	
	@if [ -n "$(PYFILES)" ]; then \
		ruff format $(PYFILES);	  \
	fi
	@if [ -n "$(JSFILES)" ]; then 		 \
		npx prettier --write $(JSFILES); \
	fi
