# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:54:20 2024

@author: Vic Chien
"""

import pandas as pd
import statsmodels.api as sm
import os

def OLS(y,X, error_type="HAC", lags=6):
    
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit(cov_type=error_type,cov_kwds={'maxlags':lags})
    
    return model

def get_ff5(path):
    
    #import FF5 factors 
    ff_factors_monthly = pd.read_csv(path,index_col=0)
    ff_factors_monthly=ff_factors_monthly/100
    ff_factors_monthly["date"] = pd.to_datetime(ff_factors_monthly.index, format="%Y%m")
    ff_factors_monthly["year"]=pd.to_datetime(ff_factors_monthly["date"]).dt.year
    ff_factors_monthly["month"]=pd.to_datetime(ff_factors_monthly["date"]).dt.month
    ff_factors_monthly.drop(columns=["date"],inplace=True)
    
    return ff_factors_monthly
    
def table_1(sample_df,leaf_index_df,ff_df):
    
    #add leaf node to sample
    sample_df["leaf_node"]=leaf_index_df['V1']
    sample_df["leaf_weight"]=sample_df.groupby(["leaf_node","date"])["weight"].transform(lambda x:x/x.sum())
    sample_df["weighted_ret"]=sample_df["leaf_weight"]*sample_df["RET"]
    
    #reconstruct leaf portfolios
    port_ret_long=sample_df.groupby(["leaf_node","date"])["weighted_ret"].sum().reset_index()
    port_ret_wide=port_ret_long.pivot(index='date', columns='leaf_node', values='weighted_ret')
    ret_cols=port_ret_wide.columns
    port_ret_wide["date"]=pd.to_datetime(port_ret_wide.index)
    port_ret_wide["year"]=port_ret_wide["date"].dt.year
    port_ret_wide["month"]=port_ret_wide["date"].dt.month
    port_ret_wide.drop(columns=["date"],inplace=True)
    
    #count number of stocks in leaf portfolios
    port_n_df=sample_df.groupby(["leaf_node","date"])["PERMNO"].count().reset_index()
    port_n_median=port_n_df.groupby("leaf_node")["PERMNO"].median()
    
    #merge leaf portfolio returns and factors
    m_ret_df=pd.merge(port_ret_wide, ff_df, on=["year","month"], how="left", validate="1:1")

    #calculate excess return
    for col in  ret_cols:
        m_ret_df[col]=m_ret_df[col]-m_ret_df["RF"]
    
    #make table 1
    result_df=pd.DataFrame(index=ret_cols,columns=["median_N","avg_ret","std","capm_alpha","capm_alpha_p","capm_beta","capm_beta_p","capm_R2","FF5_alpha","FF5_alpha_p"])
    result_df["median_N"]=port_n_median
    result_df["avg_ret"]=port_ret_wide.mean().drop(["year","month"])
    result_df["std"]=port_ret_wide.std().drop(["year","month"])
    result_df["capm_alpha"]=[OLS(m_ret_df[_],m_ret_df[["Mkt-RF"]]).params["const"] for _ in ret_cols]
    result_df["capm_alpha_p"]=[OLS(m_ret_df[_],m_ret_df[["Mkt-RF"]]).pvalues["const"] for _ in ret_cols]
    result_df["capm_beta"]=[OLS(m_ret_df[_],m_ret_df[["Mkt-RF"]]).params["Mkt-RF"] for _ in ret_cols]
    result_df["capm_beta_p"]=[OLS(m_ret_df[_],m_ret_df[["Mkt-RF"]]).pvalues["Mkt-RF"] for _ in ret_cols]
    result_df["capm_R2"]=[OLS(m_ret_df[_],m_ret_df[["Mkt-RF"]]).rsquared for _ in ret_cols]
    result_df["FF5_alpha"]=[OLS(m_ret_df[_],m_ret_df[["Mkt-RF","SMB","HML","RMW","CMA"]]).params["const"] for _ in ret_cols]
    result_df["FF5_alpha_p"]=[OLS(m_ret_df[_],m_ret_df[["Mkt-RF","SMB","HML","RMW","CMA"]]).pvalues["const"] for _ in ret_cols]    
    result_df[["avg_ret","capm_alpha","FF5_alpha"]]=result_df[["avg_ret","capm_alpha","FF5_alpha"]] * 100
    
    return result_df
  
if __name__ == "__main__":
    
    # Set current directory to the location of this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    ff_factors_monthly=get_ff5(os.path.join("..","..","raw_data","F-F_Research_Data_5_Factors_2x3_adj.csv"))
    
    train_sample_df=pd.read_csv(os.path.join("..","..","data_preparation","output","weighted_trainp.csv"))
    train_leaf_index_df=pd.read_csv(os.path.join("..","..","grow_tree","output","train_tree_leaf_index.csv"))
    train_table=table_1(train_sample_df,train_leaf_index_df,ff_factors_monthly)
    train_table.to_csv(os.path.join("..","output","train_table_1_7_21_2024.csv"))
    
    test_sample_df=pd.read_csv(os.path.join("..","..","data_preparation","output","weighted_testp.csv"))
    test_leaf_index_df=pd.read_csv(os.path.join("..","..","grow_tree","output","test_tree_leaf_index.csv"))
    test_table=table_1(test_sample_df,test_leaf_index_df,ff_factors_monthly)
    test_table.to_csv(os.path.join("..","output","test_table_1_7_21_2024.csv"))    


    