pkill -f "simulator"

parallel python HooVer_collect_data_NU.py ::: 1 2 3 4 5 6 7 8 9 10
parallel python PlasmaLab_collect_data_NU.py ::: 1 2 3 4 5 6 7 8 9 10

python plot_table_3.py
