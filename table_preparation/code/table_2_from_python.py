# -*- coding: utf-8 -*-
"""
@author: Vic Chien
"""

import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

def shrink_weight_factor(X,lambda_mu,lambda_cov,abs_norm,short_if_avg_neg):
    
    k = X.shape[1]
    cov = X.cov(ddof=0)
    w = np.linalg.inv(np.array(cov) + lambda_cov * np.eye(k)) @ (np.array(X.mean()) + lambda_mu * np.ones(k))
    
    if abs_norm:
        w = w / np.sum(np.abs(w))
    else:
        w = w / np.sum(w)
        
    f = X @ w
  
    if (short_if_avg_neg) and (np.sum(f)<0):
        
        f = -f       
    
    return w, f

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

def beta(X):
    
    n = X.shape[0]
    X = X.values.reshape(n, -1) #reshape to 2d array in case X is 1d
    y = np.ones(n)
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    
    return beta

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
    table_2_df["EF_R2"]=[OLS(b_factors_df[_],b_factors_df[[f for f in factors if f != _]]).rsquared for _ in factors]
    
    return table_2_df


def table_2_py(factors_df,ff_df):
    
    factors_df["year"]=pd.to_datetime(factors_df.index).year
    factors_df["month"]=pd.to_datetime(factors_df.index).month
    
    factors_df=pd.merge(factors_df, ff_factors_monthly, on=["year","month"], how="left", validate="1:1")

    factors=[str(_) for _ in range(20)]
    table_cols=["single_sharpe","cum_sharpe","capm_alpha","capm_alpha_t","FF5_alpha","FF5_alpha_t","EF_alpha","EF_alpha_t","EF_R2"]
    table_2_df=pd.DataFrame(index=factors,columns=table_cols)

    #for col in  factors:
    #    factors_df[col]=factors_df[col]-factors_df["RF"]

    table_2_df["single_sharpe"]=[Sharpe(factors_df[_])*(12**0.5) for _ in factors]
    table_2_df["cum_sharpe"]=[Sharpe(factors_df[factors[:_+1]])*(12**0.5) for _ in range(len(factors))]
    table_2_df["capm_alpha"]=[OLS(factors_df[_],factors_df["Mkt-RF"]).params["const"] for _ in factors]
    table_2_df["capm_alpha_t"]=[OLS(factors_df[_],factors_df["Mkt-RF"]).tvalues["const"] for _ in factors]
    table_2_df["FF5_alpha"]=[OLS(factors_df[_],factors_df[["Mkt-RF","SMB","HML","RMW","CMA"]]).params["const"] for _ in factors]
    table_2_df["FF5_alpha_t"]=[OLS(factors_df[_],factors_df[["Mkt-RF","SMB","HML","RMW","CMA"]]).tvalues["const"] for _ in factors]
    table_2_df["EF_alpha"]=[OLS(factors_df[_],factors_df[[f for f in factors if f != _]]).params["const"] for _ in factors]
    table_2_df["EF_alpha_t"]=[OLS(factors_df[_],factors_df[[f for f in factors if f != _]]).tvalues["const"] for _ in factors]
    table_2_df["EF_R2"]=[OLS(factors_df[_],factors_df[[f for f in factors if f != _]]).rsquared for _ in factors]

    
def table_2_py_OOS(train_factors_df,factors_df,ff_df):
    
    factors_df["year"]=pd.to_datetime(factors_df.index).year
    factors_df["month"]=pd.to_datetime(factors_df.index).month
    
    factors_df=pd.merge(factors_df, ff_factors_monthly, on=["year","month"], how="left", validate="1:1")

    factors=range(20)
    table_cols=["single_sharpe","cum_sharpe","capm_alpha","capm_alpha_t","FF5_alpha","FF5_alpha_t","EF_alpha","EF_alpha_t","EF_R2"]
    table_2_df=pd.DataFrame(index=factors,columns=table_cols)

    #for col in  factors:
    #    factors_df[col]=factors_df[col]-factors_df["RF"]

    table_2_df["single_sharpe"]=[Sharpe(factors_df[_])*(12**0.5) for _ in factors]
    #table_2_df["cum_sharpe"]=[Sharpe(factors_df[factors[:_+1]])*(12**0.5) for _ in range(len(factors))]
    betas=[beta(train_factors_df.iloc[:,:_+1]) for _ in range(len(factors))]
    table_2_df["cum_sharpe"]=[Sharpe(pd.DataFrame(factors_df[factors[:_+1]]) @ betas[_] )*(12**0.5) for _ in range(len(factors))]
    table_2_df["capm_alpha"]=[OLS(factors_df[_],factors_df["Mkt-RF"]).params["const"] for _ in factors]
    table_2_df["capm_alpha_t"]=[OLS(factors_df[_],factors_df["Mkt-RF"]).tvalues["const"] for _ in factors]
    table_2_df["FF5_alpha"]=[OLS(factors_df[_],factors_df[["Mkt-RF","SMB","HML","RMW","CMA"]]).params["const"] for _ in factors]
    table_2_df["FF5_alpha_t"]=[OLS(factors_df[_],factors_df[["Mkt-RF","SMB","HML","RMW","CMA"]]).tvalues["const"] for _ in factors]
    table_2_df["EF_alpha"]=[OLS(factors_df[_],factors_df[[f for f in factors if f != _]]).params["const"] for _ in factors]
    table_2_df["EF_alpha_t"]=[OLS(factors_df[_],factors_df[[f for f in factors if f != _]]).tvalues["const"] for _ in factors]
    table_2_df["EF_R2"]=[OLS(factors_df[_],factors_df[[f for f in factors if f != _]]).rsquared for _ in factors]
    
    return table_2_df    

if __name__ == "__main__":

    # Set current directory to the location of this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
    ff_factors_monthly=get_ff5(os.path.join("..","..","raw_data","F-F_Research_Data_5_Factors_2x3_adj.csv"))
    
    #factor_train_df=pd.read_csv(os.path.join("..","..","grow_tree","output","Vanilla_Boosted_train_factor_19_rf.csv"),index_col=0).sort_index()
    #table_2_train_df=table_2_py(factor_train_df,ff_factors_monthly)
    #table_2_train_df.to_csv(os.path.join("..","output","train_table_2_8_4_2024.csv"))    
    
    factor_train_df=pd.read_csv(os.path.join("..","..","grow_tree","output","Vanilla_Boosted_train_factor_19_rf.csv"),index_col=0).sort_index()
    factor_train_df.columns=factor_train_df.columns.astype(int)
    
    factor_OOS_df=pd.read_csv(os.path.join("..","..","grow_tree","output","Vanilla_Boosted_OOS_factor_rf_8_7_2024.csv"),index_col=0).sort_index()
    factor_OOS_df.columns=factor_OOS_df.columns.astype(int)
    
    table_2_OOS_df=table_2_py_OOS(factor_train_df,factor_OOS_df,ff_factors_monthly)
    #table_2_OOS_df=table_2_py_OOS(factor_train_df,factor_OOS_df,ff_factors_monthly)
    table_2_OOS_df.to_csv(os.path.join("..","output","OOS_table_2_8_7_2024.csv"))  
    
    print("stop")
    '''
        f_factor_test_df=pd.read_csv(os.path.join("..","..","grow_tree","output","Vanilla_First_factor_test.csv"))
        b_factors_test_df=pd.read_csv(os.path.join("..","..","grow_tree","output","Vanilla_Boosted_factors_test.csv"))
        date_range_test=get_date_range(os.path.join("..","..","data_preparation","output","weighted_testp.csv"))
        table_2_test_df=table_2(f_factor_test_df,b_factors_test_df,ff_factors_monthly,date_range_test)
        table_2_test_df.to_csv(os.path.join("..","output","test_table_2_7_21_2024.csv"))
        
        f_factor_oos_df=pd.read_csv(os.path.join("..","..","grow_tree","output","Vanilla_First_factor_OOS.csv"))
        b_factors_oos_df=pd.read_csv(os.path.join("..","..","grow_tree","output","Vanilla_Boosted_factors_OOS.csv"))
        date_range_oos=get_date_range(os.path.join("..","..","data_preparation","output","weighted_testp.csv"))
        table_2_oos_df=table_2(f_factor_oos_df,b_factors_oos_df,ff_factors_monthly,date_range_oos)
        table_2_oos_df.to_csv(os.path.join("..","output","OOS_table_2_7_21_2024.csv"))    
    '''
    