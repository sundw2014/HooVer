mkdir -p ../data
mkdir -p ../results
pkill -f "simulator"

parallel -u python3 -u HooVer_collect_data_SL_rhomax.py ::: 1 2 3 4 5 6 7 8 9 10

python3 plot_table_2.py > ../results/Table2.txt
