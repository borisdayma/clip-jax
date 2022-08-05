check_dirs := ./

quality:
	black $(check_dirs)
	isort $(check_dirs)
	flake8 $(check_dirs)

style:
	black $(check_dirs)
	isort $(check_dirs)