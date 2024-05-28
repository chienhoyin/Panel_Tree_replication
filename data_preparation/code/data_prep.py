# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:47:10 2024

@author: Vic Chien
"""

import pandas as pd
import numpy as np
import wrds
from  scipy.stats.mstats import winsorize
import os


def standardize_X(sample_df):
    
    #make X at each month uniform [-1,1]
    chars_col=[f"f{_}" for _ in range(1,51)]
    for col in chars_col:    
        sample_df[col]=sample_df.groupby(["year","month"])[col].transform(lambda x:x.rank(method="average")/x.count())
    sample_df[chars_col]= sample_df[chars_col] * 2 - 1

    return sample_df


def add_weight(sample_df,crsp_df):
    
    #add stock weights in root portfolio from crsp data to sample
    crsp_df.rename({"prc":"PRC","permno":"PERMNO","shrout":"SHROUT"},axis=1,inplace=True)
    crsp_df["PRC"]=crsp_df["PRC"].abs()
    crsp_df["mkt_cap"]=crsp_df["PRC"] * crsp_df["SHROUT"]
    crsp_df["mkt_cap"]=crsp_df["mkt_cap"].replace(0, np.nan)
    crsp_df.sort_values(by=["PERMNO","year","month"],inplace=True)
    crsp_df["mkt_cap"]=crsp_df.groupby("PERMNO")["mkt_cap"].shift()
    
    merged_df=pd.merge(sample_df,crsp_df[["year","month","PERMNO","mkt_cap"]], on=["year","month","PERMNO"],how="left",validate="1:1")
    merged_df.sort_values(by=["PERMNO","year","month"],inplace=True)
    merged_df["mkt_cap"]=merged_df.groupby("PERMNO")["mkt_cap"].ffill()
    merged_df["mkt_cap"]=merged_df.groupby(["year","month"])["mkt_cap"].transform(lambda x: x.fillna(x.mean()))
    merged_df["weight"]=merged_df.groupby(["year","month"])["mkt_cap"].transform(lambda x:x/x.sum())
    merged_df.drop(columns=["mkt_cap"],inplace=True)
    
    return merged_df


def win(sample_df):
    
    #cross sectionally winsorize return
    sample_df["RET"]=sample_df.groupby(["year","month"])["RET"].transform(lambda x:winsorize(x,limits=0.01))

    return sample_df
    

if __name__ == "__main__":
    
    #requst wrds for crsp data
    conn = wrds.Connection()
    crsp_df = conn.raw_sql("""select permno, date, prc, ret, shrout 
                            from crsp.msf 
                            where date>='01/01/1980'""")
    crsp_df["year"]=pd.to_datetime(crsp_df["date"]).dt.year
    crsp_df["month"]=pd.to_datetime(crsp_df["date"]).dt.month

    # truncate data before 1984 (too few stocks), standardize X, add weight, winsorize training sample
    train_df=pd.read_csv(os.path.join("..","..","raw_data","trainp.csv"))
    train_df["year"]=pd.to_datetime(train_df["date"]).dt.year
    train_df["month"]=pd.to_datetime(train_df["date"]).dt.month
    train_df=train_df[train_df["year"]>=1984]       
    train_df=standardize_X(train_df)
    train_df=add_weight(train_df,crsp_df)
    train_df=win(train_df)
    
    # truncate data before 1984 (too few stocks), standardize X, add weight, winsorize testing sample
    test_df=pd.read_csv(os.path.join("..","..","raw_data","testp.csv"))    
    test_df["year"]=pd.to_datetime(test_df["date"]).dt.year
    test_df["month"]=pd.to_datetime(test_df["date"]).dt.month
    test_df=test_df[test_df["year"]>=2004]
    test_df=standardize_X(test_df)
    test_df=add_weight(test_df,crsp_df)
    test_df=win(test_df)
    
    train_df.to_csv(os.path.join("..","output","weighted_trainp.csv"))
    test_df.to_csv(os.path.join("..","output","weighted_testp.csv"))
