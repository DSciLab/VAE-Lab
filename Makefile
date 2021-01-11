MAKE			:= make
PIP				:= pip
REQUIREMENTS	:= requirements.txt
CFG_DIR			:= cfg
MLUTILS_DIR		:= mlutils
NATURE_DATASETS	:= nature_datasets


.PHONY: all dep cfg mlutils install


all: dep install


install: dep install_cfg install_mlutils install_n_datasets


dep: $(REQUIREMENTS)
	$(PIP) install -r $^


install_cfg: $(CFG_DIR)
	$(MAKE) -C $^ install


install_mlutils: $(MLUTILS_DIR)
	$(MAKE) -C $^ install


install_n_datasets: $(NATURE_DATASETS)
	$(MAKE) -C $^ install