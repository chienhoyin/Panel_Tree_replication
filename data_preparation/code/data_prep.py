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
    
def add_loss_weight(sample_df):
    
    weights=sample_df.groupby(["date"])["RET"].count()
    weights=1/weights
    weights.rename("loss_weight",inplace=True)
    sample_df=pd.merge(left=sample_df,right=weights,left_on="date",right_index=True,how="left",validate="m:1")
    
    return sample_df

def subtract_rf(sample_df,ff_5_path):
        
    ff_factors_monthly = pd.read_csv(ff_5_path,index_col=0)
    ff_factors_monthly=ff_factors_monthly/100
    ff_factors_monthly["date"] = pd.to_datetime(ff_factors_monthly.index, format="%Y%m")
    ff_factors_monthly["year"]=pd.to_datetime(ff_factors_monthly["date"]).dt.year
    ff_factors_monthly["month"]=pd.to_datetime(ff_factors_monthly["date"]).dt.month
    rf_df=ff_factors_monthly[["RF","year","month"]]
    
    #merge and subtract rf
    sample_df=pd.merge(left=sample_df,right=rf_df,on=["year","month"],how="left",validate="m:1")
    sample_df["RET"]=sample_df["RET"]-sample_df["RF"]
    sample_df.drop(columns=["RF"],inplace=True)
    
    return sample_df

if __name__ == "__main__":

    # Set current directory to the location of this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

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
    train_df=add_loss_weight(train_df)
    train_df=subtract_rf(train_df,os.path.join("..","..","raw_data","F-F_Research_Data_5_Factors_2x3_adj.csv"))
    
    # truncate data before 2004 (too few stocks), standardize X, add weight, winsorize testing sample
    
    test_df=pd.read_csv(os.path.join("..","..","raw_data","testp.csv"))    
    test_df["year"]=pd.to_datetime(test_df["date"]).dt.year
    test_df["month"]=pd.to_datetime(test_df["date"]).dt.month
    test_df=test_df[test_df["year"]>=2004]
    test_df=standardize_X(test_df)
    test_df=add_weight(test_df,crsp_df)
    test_df=win(test_df)
    test_df=add_loss_weight(test_df)
    test_df=subtract_rf(test_df,os.path.join("..","..","raw_data","F-F_Research_Data_5_Factors_2x3_adj.csv"))   
 
    train_df.to_csv(os.path.join("..","output","weighted_trainp_loss_weight_rf.csv"))
    test_df.to_csv(os.path.join("..","output","weighted_testp_loss_weight_rf.csv"))

    #train_df_toy=train_df.drop(columns=[f"f{_}" for _ in range(4,51)])
    #test_df_toy=test_df.drop(columns=[f"f{_}" for _ in range(4,51)])
    
    #train_df_toy.to_csv(os.path.join("..","output","weighted_trainp_loss_weight_toy.csv"))
    #test_df_toy.to_csv(os.path.join("..","output","weighted_testp_loss_weight_toy.csv"))    