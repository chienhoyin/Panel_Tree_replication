# How to install Tree Factor
- Download and install R package "TreeFactor" at https://github.com/Quantactix/TreeFactor

# How to run the replication
- Fork and clone this repo to your machine
- Move trainp.csv and testp.csv to raw_data
- Set the current directory to the local GitHub repo
- Run master_script.sh (or you can run each individual script separately)

# How to run Python/R scripts separately 
- Run ```python3 data_preparation/code/data_prep.py```
- Run ```Rscript grow_tree/code/Vanilla_Boosted.R```
- Run ```python3 table_preparation/code/table_1.py```
- Run ```python3 table_preparation/code/table_2.py```
