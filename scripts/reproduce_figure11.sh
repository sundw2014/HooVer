pkill -f "python"

parallel -u python -u HooVer_collect_data_SL.py ::: 1 2 3 4 5 6 7 8 9 10
parallel -u python -u PlasmaLab_collect_data_SL.py ::: 1 2 3 4 5 6 7 8 9 10

python plot_figure_1.py 1
