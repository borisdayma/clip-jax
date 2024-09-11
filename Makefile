check_dirs := ./

style:
	ruff check --select I --fix $(check_dirs)
	ruff format $(check_dirs)