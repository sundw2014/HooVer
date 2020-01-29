pkill -f "python"

parallel -u python -u HooVer_collect_data_SL_rhomax.py ::: 1 2 3 4 5 6 7 8 9 10

python plot_table_2.py
