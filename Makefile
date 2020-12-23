MAKE			:= make
PIP				:= pip
REQUIREMENTS	:= requirements.txt
CFG_DIR			:= cfg
VIZBOARD_DIR	:= vizboard


.PHONY: all dep cfg vizboard


all: dep cfg vizboard commit push


dep: $(REQUIREMENTS)
	$(PIP) install -r $^


cfg:
	$(MAKE) -C $(CFG_DIR)


vizboard:
	$(MAKE) -C $(VIZBOARD_DIR)


commit: .git
	# NOT RECOMMENDED 
	git add -A
	git commit -m 'Updaet project'


push: commit
	git push
