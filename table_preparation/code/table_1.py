# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:54:20 2024

@author: Vic Chien
"""

import pandas as pd
import statsmodels.api as sm

'''
# use Yahoo Finance to download historical data for QQQ
# over the last 15 years, from 2006-01-01 to 2021-01-01
qqq_daily = yf.download("QQQ", start="2006-01-01", end="2023-12-31")
qqq_daily["Adj Close"].plot(title="QQQ Daily Adjusted Close", figsize=(5, 3))

# calculate monthly returns of QQQ
qqq_monthly = qqq_daily["Adj Close"].resample("M").ffill().to_frame()
qqq_monthly.index = qqq_monthly.index.to_period("M")
qqq_monthly["Return"] = qqq_monthly["Adj Close"].pct_change() * 100
qqq_monthly.dropna(inplace=True)
qqq_monthly



'''

def factor_model(y,x):
    X = sm.add_constant(x)
    # Run the regression
    model = sm.OLS(y, X).fit()
    # Display the summary of the regression
    print(model.summary())
    return model


test=pd.read_csv(r"C:\Users\Vic Chien\Documents\Cornell\Will Cong\weighted_testp.csv")
dates=sorted(list(set(test["date"])))

port_leaf_index_df=pd.read_csv(r"C:\Users\Vic Chien\Documents\Cornell\Will Cong\tree result\test_first_tree_leaf_index.csv")
test["leaf_node"]=port_leaf_index_df['V1']
test["leaf_weight"]=test.groupby(["leaf_node","date"])["weight"].transform(lambda x:x/x.sum())
test["weighted_ret"]=test["leaf_weight"]*test["RET"]


port_ret_long=test.groupby(["leaf_node","date"])["weighted_ret"].sum().reset_index()
port_ret_wide=port_ret_long.pivot(index='date', columns='leaf_node', values='weighted_ret')
ret_cols=port_ret_wide.columns
port_ret_wide["date"]=pd.to_datetime(port_ret_wide.index)
port_ret_wide["year"]=port_ret_wide["date"].dt.year
port_ret_wide["month"]=port_ret_wide["date"].dt.month
port_ret_wide.drop(columns=["date"],inplace=True)


port_n_df=test.groupby(["leaf_node","date"])["PERMNO"].count().reset_index()
port_n_median=port_n_df.groupby("leaf_node")["PERMNO"].median()


ff_factors_monthly = pd.read_csv(r"C:\Users\Vic Chien\Documents\Cornell\Will Cong\F-F_Research_Data_Factors_CSV\F-F_Research_Data_Factors.csv",index_col=0)
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
result_df["capm_alpha"]=[factor_model(m_ret_df[_],m_ret_df[["Mkt-RF"]]).params["const"] for _ in ret_cols]
result_df["capm_beta"]=[factor_model(m_ret_df[_],m_ret_df[["Mkt-RF"]]).params["Mkt-RF"] for _ in ret_cols]
result_df["capm_R2"]=[factor_model(m_ret_df[_],m_ret_df[["Mkt-RF"]]).rsquared for _ in ret_cols]
result_df["FF5_alpha"]=[factor_model(m_ret_df[_],m_ret_df[["Mkt-RF","SMB","HML"]]).rsquared for _ in ret_cols]

result_df.to_csv("table_1_rep.csv")
