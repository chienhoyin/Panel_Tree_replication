# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:47:10 2024

@author: Vic Chien
"""

import pandas as pd
import numpy as np
from  scipy.stats.mstats import winsorize


def standardize_X(sample_df):
    
    chars_col=[f"f{_}" for _ in range(1,51)]
    
    for col in chars_col:    
        sample_df[col]=sample_df.groupby(["year","month"])[col].transform(lambda x:x.rank(method="average")/x.count())
        
    sample_df[chars_col]= sample_df[chars_col] * 2 - 1

    return sample_df


def add_weight(sample_df,crsp_df):
        
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
    sample_df
    sample_df["RET"]=sample_df.groupby(["year","month"])["RET"].transform(lambda x:winsorize(x,limits=0.01))

    return sample_df
    

if __name__ == "__main__":


    train_df=pd.read_csv(r"C:\Users\Vic Chien\Documents\Cornell\Will Cong\trainp\trainp.csv")
    train_crsp_df=pd.read_csv(r"C:\Users\Vic Chien\Documents\Cornell\Will Cong\train_crsp_monthly.csv")
    
    train_df["year"]=pd.to_datetime(train_df["date"]).dt.year
    train_df["month"]=pd.to_datetime(train_df["date"]).dt.month
    train_crsp_df["year"]=pd.to_datetime(train_crsp_df["date"]).dt.year
    train_crsp_df["month"]=pd.to_datetime(train_crsp_df["date"]).dt.month
    train_df=train_df[train_df["year"]>=1984]   
    
    train_df=standardize_X(train_df)
    train_df=add_weight(train_df,train_crsp_df)
    train_df=win(train_df)
    train_df=train_df[train_df["year"]>=1984]
    
    test_df=pd.read_csv(r"C:\Users\Vic Chien\Documents\Cornell\Will Cong\testp\testp.csv")
    test_crsp_df=pd.read_csv(r"C:\Users\Vic Chien\Documents\Cornell\Will Cong\test_crsp_monthly.csv")
    
    test_df["year"]=pd.to_datetime(test_df["date"]).dt.year
    test_df["month"]=pd.to_datetime(test_df["date"]).dt.month
    test_crsp_df["year"]=pd.to_datetime(test_crsp_df["date"]).dt.year
    test_crsp_df["month"]=pd.to_datetime(test_crsp_df["date"]).dt.month
    test_df=test_df[test_df["year"]>=2004]         
    
    test_df=standardize_X(test_df)
    test_df=add_weight(test_df,test_crsp_df)
    test_df=win(test_df)
    
