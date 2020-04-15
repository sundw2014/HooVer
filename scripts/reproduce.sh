pkill -f "simulator"

parallel -u python -u HooVer_collect_data_ML.py ::: 1 2 3 4 5 6 7 8 9 10
parallel -u python -u PlasmaLab_collect_data_ML.py ::: 1 2 3 4 5 6 7 8 9 10

parallel -u python -u HooVer_collect_data_DP.py ::: 1 2 3 4 5 6 7 8 9 10
parallel -u python -u PlasmaLab_collect_data_DP.py ::: 1 2 3 4 5 6 7 8 9 10

parallel -u python -u HooVer_collect_data_ME.py ::: 1 2 3 4 5 6 7 8 9 10
parallel -u python -u PlasmaLab_collect_data_ME.py ::: 1 2 3 4 5 6 7 8 9 10

#parallel -u python -u HooVer_collect_data_NU.py ::: 1 2 3 4 5 6 7 8 9 10
#parallel -u python -u PlasmaLab_collect_data_NU.py ::: 1 2 3 4 5 6 7 8 9 10

parallel -u python -u HooVer_collect_data_NU_nqueries_regret.py ::: 1 2 3 4 5 6 7 8 9 10
parallel -u python -u PlasmaLab_collect_data_NU_nqueries_regret.py ::: 1 2 3 4 5 6 7 8 9 10

parallel -u python -u HooVer_collect_data_SL.py ::: 1 2 3 4 5 6 7 8 9 10
parallel -u python -u PlasmaLab_collect_data_SL.py ::: 1 2 3 4 5 6 7 8 9 10

#parallel -u python -u HooVer_collect_data_SL_bs.py ::: 1 2 3 4 5 6 7 8 9 10
parallel -u python -u HooVer_collect_data_SL_rhomax.py ::: 1 2 3 4 5 6 7 8 9 10
parallel -u python -u HooVer_collect_data_SL_nqueries_regret_bs.py ::: 1 2 3 4 5 6 7 8 9 10

#python plot_figure_1.py
