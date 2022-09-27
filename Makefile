check_dirs := ./

style:
	black $(check_dirs)
	isort $(check_dirs)