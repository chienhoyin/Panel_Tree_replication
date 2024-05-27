# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:54:20 2024

@author: Vic Chien
"""

import pandas as pd
import statsmodels.api as sm
import os

def OLS(y,x):
    X = sm.add_constant(x)
    # Run the regression
    model = sm.OLS(y, X).fit()
    # Display the summary of the regression
    print(model.summary())
    return model

def table(sample_df,leaf_index_df):
      
    sample_df["leaf_node"]=leaf_index_df['V1']
    sample_df["leaf_weight"]=sample_df.groupby(["leaf_node","date"])["weight"].transform(lambda x:x/x.sum())
    sample_df["weighted_ret"]=sample_df["leaf_weight"]*sample_df["RET"]
    
    
    port_ret_long=sample_df.groupby(["leaf_node","date"])["weighted_ret"].sum().reset_index()
    port_ret_wide=port_ret_long.pivot(index='date', columns='leaf_node', values='weighted_ret')
    ret_cols=port_ret_wide.columns
    port_ret_wide["date"]=pd.to_datetime(port_ret_wide.index)
    port_ret_wide["year"]=port_ret_wide["date"].dt.year
    port_ret_wide["month"]=port_ret_wide["date"].dt.month
    port_ret_wide.drop(columns=["date"],inplace=True)
    
    
    port_n_df=sample_df.groupby(["leaf_node","date"])["PERMNO"].count().reset_index()
    port_n_median=port_n_df.groupby("leaf_node")["PERMNO"].median()
    
    
    ff_factors_monthly = pd.read_csv(os.path.join("..","..","raw_data","F-F_Research_Data_Factors_CSV\F-F_Research_Data_5_Factors_2x3_adj.csv"),index_col=0)
    ff_factors_monthly=ff_factors_monthly/100
    ff_factors_monthly["date"] = pd.to_datetime(ff_factors_monthly.index, format="%Y%m")
    ff_factors_monthly["year"]=pd.to_datetime(ff_factors_monthly["date"]).dt.year
    ff_factors_monthly["month"]=pd.to_datetime(ff_factors_monthly["date"]).dt.month
    ff_factors_monthly.drop(columns=["date"],inplace=True)
    
    
    m_ret_df=pd.merge(port_ret_wide, ff_factors_monthly, on=["year","month"], how="left", validate="1:1")
    
    for col in  ret_cols:
        m_ret_df[col]=m_ret_df[col]-m_ret_df["RF"]
    
    result_df=pd.DataFrame(index=ret_cols,columns=["median_N","avg_ret","std","capm_alpha","capm_beta","capm_R2","FF5_alpha"])
    
    result_df["median_N"]=port_n_median
    result_df["avg_ret"]=port_ret_wide.mean().drop(["year","month"])
    result_df["std"]=port_ret_wide.std().drop(["year","month"])
    result_df["capm_alpha"]=[OLS(m_ret_df[_],m_ret_df[["Mkt-RF"]]).params["const"] for _ in ret_cols]
    result_df["capm_beta"]=[OLS(m_ret_df[_],m_ret_df[["Mkt-RF"]]).params["Mkt-RF"] for _ in ret_cols]
    result_df["capm_R2"]=[OLS(m_ret_df[_],m_ret_df[["Mkt-RF"]]).rsquared for _ in ret_cols]
    result_df["FF5_alpha"]=[OLS(m_ret_df[_],m_ret_df[["Mkt-RF","SMB","HML","RMW","CMA"]]).params["const"] for _ in ret_cols]
    return result_df
  
if __name__ == "__main__":
    
    test_sample_df=pd.read_csv(os.path.join("..","..","raw_data","testp.csv"))
    test_leaf_index_df=pd.read_csv(os.path.join("..","..","grow_tree","output","test_first_tree_leaf_index.csv"))
    test_table=table(test_sample_df,test_leaf_index_df)

    train_sample_df=pd.read_csv(os.path.join("..","..","raw_data","trainp.csv"))
    train_leaf_index_df=pd.read_csv(os.path.join("..","..","grow_tree","output","train_first_tree_leaf_index.csv"))
    train_table=table(train_sample_df,train_leaf_index_df)
    
    test_table.to_csv(os.path.join("..","output","test_table_1.csv"))
    train_table.to_csv(os.path.join("..","output","train_table_1.csv"))
