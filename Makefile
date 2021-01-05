MAKE			:= make
PIP				:= pip
REQUIREMENTS	:= requirements.txt
CFG_DIR			:= cfg
MLUTILS_DIR		:= mlutils


.PHONY: all dep cfg mlutils install


all: dep install


install: install_cfg install_mlutils


dep: $(REQUIREMENTS)
	$(PIP) install -r $^


cfg:
	$(MAKE) -C $(CFG_DIR)


mlutils:
	$(MAKE) -C $(MLUTILS_DIR)


install_cfg:
	$(MAKE) -C $(CFG_DIR) install


install_mlutils:
	$(MAKE) -C $(MLUTILS_DIR) install


commit: .git
	# NOT RECOMMENDED 
	git add -A
	-git commit -m 'Updaet project'


push: commit
	git push
