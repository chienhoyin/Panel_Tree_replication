cd data_preparation/code

python3 data_prep.py 

printf '\n data prepared \n' 

cd ../../grow_tree/code

Rscript grow_tree.R 

printf '\n tree grown \n' 

cd ../../table_preparation/code

python3 table_1.py 

python3 table_2.py 

