# -*- coding: utf-8 -*-
"""
@author: Vic Chien
"""

import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

def OLS(y,X, error_type="HAC", lags=6):
    
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit(cov_type=error_type,cov_kwds={'maxlags':lags})
    
    return model

def Sharpe(X):
    
    n = X.shape[0]
    X = X.values.reshape(n, -1) #reshape to 2d array in case X is 1d
    y = np.ones(n)
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    residuals = y - X @ beta
    rss = np.sum(residuals**2)
    sr =  (n / rss - 1)**0.5
    
    return sr

def get_ff5(path):
    
    #import FF5 factors 
    ff_factors_monthly = pd.read_csv(path,index_col=0)
    ff_factors_monthly=ff_factors_monthly/100
    ff_factors_monthly["date"] = pd.to_datetime(ff_factors_monthly.index, format="%Y%m")
    ff_factors_monthly["year"]=pd.to_datetime(ff_factors_monthly["date"]).dt.year
    ff_factors_monthly["month"]=pd.to_datetime(ff_factors_monthly["date"]).dt.month
    ff_factors_monthly.drop(columns=["date"],inplace=True)
    
    return ff_factors_monthly

def get_date_range(sample_df_path):
    
    train_sample_df=pd.read_csv(sample_df_path)
    date_range=sorted(list(set(train_sample_df["date"])))
    
    return date_range

def table_2(f_factor_df,b_factors_df,ff_df,date_range):
    
    b_factors_df["ft0"]=f_factor_df["V1"]
    b_factors_df["date"]=date_range
    b_factors_df["year"]=pd.to_datetime(b_factors_df["date"]).dt.year
    b_factors_df["month"]=pd.to_datetime(b_factors_df["date"]).dt.month
    b_factors_df.drop(columns=["date"],inplace=True)
    b_factors_df=pd.merge(b_factors_df, ff_factors_monthly, on=["year","month"], how="left", validate="1:1")

    factors=[f"ft{_}" for _ in range(21)]
    table_cols=["single_sharpe","cum_sharpe","capm_alpha","capm_alpha_t","FF5_alpha","FF5_alpha_t","EF_alpha","EF_alpha_t","EF_R2"]
    table_2_df=pd.DataFrame(index=factors,columns=table_cols)

    for col in  factors:
        b_factors_df[col]=b_factors_df[col]-b_factors_df["RF"]

    table_2_df["single_sharpe"]=[Sharpe(b_factors_df[_]) for _ in factors]
    table_2_df["cum_sharpe"]=[Sharpe(b_factors_df[factors[:_+1]]) for _ in range(len(factors))]
    table_2_df["capm_alpha"]=[OLS(b_factors_df[_],b_factors_df["Mkt-RF"]).params["const"] for _ in factors]
    table_2_df["capm_alpha_t"]=[OLS(b_factors_df[_],b_factors_df["Mkt-RF"]).tvalues["const"] for _ in factors]
    table_2_df["FF5_alpha"]=[OLS(b_factors_df[_],b_factors_df[["Mkt-RF","SMB","HML","RMW","CMA"]]).params["const"] for _ in factors]
    table_2_df["FF5_alpha_t"]=[OLS(b_factors_df[_],b_factors_df[["Mkt-RF","SMB","HML","RMW","CMA"]]).tvalues["const"] for _ in factors]
    table_2_df["EF_alpha"]=[OLS(b_factors_df[_],b_factors_df[[f for f in factors if f != _]]).params["const"] for _ in factors]
    table_2_df["EF_alpha_t"]=[OLS(b_factors_df[_],b_factors_df[[f for f in factors if f != _]]).tvalues["const"] for _ in factors]
    table_2_df["EF_alpha_R2"]=[OLS(b_factors_df[_],b_factors_df[[f for f in factors if f != _]]).rsquared for _ in factors]
    
    return table_2_df

if __name__ == "__main__":

    # Set current directory to the location of this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
    ff_factors_monthly=get_ff5(os.path.join("..","..","raw_data","F-F_Research_Data_5_Factors_2x3_adj.csv"))

    f_factor_train_df=pd.read_csv(os.path.join("..","..","grow_tree","output","Vanilla_First_factor.csv"))
    b_factors_train_df=pd.read_csv(os.path.join("..","..","grow_tree","output","Vanilla_Boosted_factors.csv"))
    date_range_train=get_date_range(os.path.join("..","..","data_preparation","output","weighted_trainp.csv"))
    table_2_train_df=table_2(f_factor_train_df,b_factors_train_df,ff_factors_monthly,date_range_train)
    table_2_train_df.to_csv(os.path.join("..","output","train_table_2_7_21_2024.csv"))    

    f_factor_test_df=pd.read_csv(os.path.join("..","..","grow_tree","output","Vanilla_First_factor_test.csv"))
    b_factors_test_df=pd.read_csv(os.path.join("..","..","grow_tree","output","Vanilla_Boosted_factors_test.csv"))
    date_range_test=get_date_range(os.path.join("..","..","data_preparation","output","weighted_testp.csv"))
    table_2_test_df=table_2(f_factor_test_df,b_factors_test_df,ff_factors_monthly,date_range_test)
    table_2_test_df.to_csv(os.path.join("..","output","test_table_2_7_21_2024.csv"))    
