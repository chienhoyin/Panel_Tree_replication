import numpy as np
import pandas as pd
import os
import re
import statsmodels.api as sm

def shrink_weight_factor(X,lambda_mu,lambda_cov,abs_norm):
    
    k = X.shape[1]
    cov = X.cov()
    w = np.linalg.inv(np.array(cov) + lambda_cov * np.eye(k)) @ (np.array(X.mean()) + np.ones(k))
    
    if abs_norm:
        w = w / np.sum(np.abs(w))
    else:
        w = w / np.sum(w)
    f = X @ w
    
    return f

def Sharpe(X):
    
    n = X.shape[0]
    X = X.values.reshape(n, -1) #reshape to 2d array in case X is 1d
    y = np.ones(n)
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    residuals = y - X @ beta
    rss = np.sum(residuals**2)
    sr =  (n / rss - 1)**0.5
    
    return sr

    '''
    mean = np.array([0, 1])
    cov = np.array([[2, 0.5], [0.5, 3]])
    n_obs = 100
    df = pd.DataFrame(np.random.multivariate_normal(mean, cov, n_obs), columns=['var1', 'var2'])
    df'''
    
def filter_panel_by_node(node_row,sample_df,max_node):
    temp_df=sample_df
    for parent_node_no in range(2*max_node-1):
        if node_row[f"node_{parent_node_no}_parent"] == "left":
            temp_df=temp_df[temp_df[f"split_{parent_node_no}"]]
        elif node_row[f"node_{parent_node_no}_parent"] == "right":
            temp_df=temp_df[~temp_df[f"split_{parent_node_no}"]]
    return temp_df

def split_panel(node_sample_df,feat,thres):
        
        left_df=node_sample_df[node_sample_df[feat]>=thres]
        right_df=node_sample_df[node_sample_df[feat]<thres]
        
        return left_df,right_df
    
def gen_port_ret(node_sample_df):
        
        ret=node_sample_df.groupby("date").apply(lambda x: (x["ret"]*x["weight"]/(x["weight"].sum())).sum())
        
        return ret

def gen_ran_data(n_stock,n_period,n_feat):
    
    X = np.vstack([np.vstack([np.random.permutation(np.linspace(-1,1,n_stock)) for _ in range(n_feat)]).T for n in range(n_period)])
    cols=[f"f{_}" for _ in range(n_feat)]
    sample_df=pd.DataFrame(X,columns=cols)
    dates = pd.date_range('2000-01-01', periods=n_period, freq='MS')
    sample_df["date"] = [date for date in dates for _ in range(n_stock)]
    sample_df["PERMNO"] = list(range(n_stock))*n_period
    sample_df["ret"] = np.random.randn(n_stock*n_period)
    sample_df["weight"] = np.random.rand(n_stock*n_period)
    sample_df["weight"] = sample_df["weight"]/sample_df.groupby("date")["weight"].transform("sum")
    
    return sample_df

def gen_data_by_tree(tree_df,n_stock,n_period,n_feat,thres_list,max_node,min_node_size):
    # Xs are indpedent for all stocks
    # Within the same group, return is 100% correlated (same mean as well, otherwise it messes with expected return in SR)
    # same weight
    
    X = np.vstack([np.vstack([np.random.permutation(np.linspace(-1,1,n_stock)) for _ in range(n_feat)]).T for n in range(n_period)])
    cols=[f"f{_}" for _ in range(n_feat)]
    sample_df=pd.DataFrame(X,columns=cols)
    dates = pd.date_range('2000-01-01', periods=n_period, freq='MS')
    sample_df["date"] = [date for date in dates for _ in range(n_stock)]
    sample_df["PERMNO"] = list(range(n_stock))*n_period
    
    for ind,row in tree_df[tree_df["leaf"]].iterrows():
        temp = sample_df
        no_nodes_col = len(sample_df.filter(regex=re.compile(r'^node_')).columns)
        for col_no in range(no_nodes_col):
            if row[f"node_{col_no}_parent"] == "left":
                temp_df = temp_df[temp_df[f"f{col_no}"]>=row["thres"]]
    
    return sample_df

def check_node_size(node_sample_df,min_node_size,date_range):
    if min([node_sample_df[node_sample_df["date"]==date].shape[0] for date in date_range]) < min_node_size:
        return False
    else:
        return True

def gen_tree(sample_df,n_period,n_stock,n_feat,thres_list,max_node,min_node_size,prior_tree_factors_df,lambda_mu,lambda_cov,abs_norm):
    
    node_cols=[f"node_{i}_parent" for i in range(2*max_node-1)]
    tree_df=pd.DataFrame([[999,999,True]+["Irr"]*(2*max_node-1)], columns=["char","thres","leaf"]+node_cols)
    port_df=pd.DataFrame(index=sample_df["date"].unique())
    feat_list=[f"f{_}" for _ in range(n_feat)]
    log_df=pd.DataFrame()
    date_range=sample_df["date"].unique()
    
    for split in range(max_node-1):
        for ind,row in tree_df[tree_df["leaf"]].iterrows():
            node_sample_df=filter_panel_by_node(row,sample_df,max_node)
            for feat in feat_list:
                for thres in thres_list:
                        left_df,right_df=split_panel(node_sample_df,feat,thres)
                        if (not check_node_size(left_df,min_node_size,date_range)) or (not check_node_size(right_df,min_node_size,date_range)):
                            log_df=pd.concat([log_df,pd.DataFrame({"char":feat,"thres":thres,"split":split,"node":ind,"sr":np.nan},index=[0])],ignore_index=True)
                            continue
                        left_port_ret=gen_port_ret(left_df)
                        right_port_ret=gen_port_ret(right_df)
                        temp_port_df=pd.concat([left_port_ret,right_port_ret,port_df],axis=1)
                        tree_factor=shrink_weight_factor(temp_port_df,lambda_mu,lambda_cov,abs_norm)
                        sr=Sharpe(pd.concat([tree_factor,prior_tree_factors_df],axis=1))
                        log_df=pd.concat([log_df,pd.DataFrame({"char":feat,"thres":thres,"split":split,"node":ind,"sr":sr},index=[0])],ignore_index=True)
                        
        best_split=log_df[(log_df["split"]==split)]
        best_split=best_split[(best_split["sr"]==best_split["sr"].max())].iloc[0]
        
        new_left_node={"char":best_split["char"],"thres":best_split["thres"],"leaf":True}
        new_left_node.update(tree_df.loc[best_split["node"]][node_cols].to_dict())
        new_left_node.update({f"node_{best_split['node']}_parent":"left"})
        tree_df=pd.concat([tree_df,pd.DataFrame(new_left_node,index=[0])],ignore_index=True)
        
        new_right_node={"char":best_split["char"],"thres":best_split["thres"],"leaf":True}
        new_right_node.update(tree_df.loc[best_split["node"]][node_cols].to_dict())
        new_right_node.update({f"node_{best_split['node']}_parent":"right"})
        tree_df=pd.concat([tree_df,pd.DataFrame(new_right_node,index=[0])],ignore_index=True)
        tree_df.loc[best_split["node"],"leaf"]=False
        
        sample_df["split_"+str(best_split["node"])]=sample_df[best_split["char"]]>=best_split["thres"]
        
        new_left_df=filter_panel_by_node(new_left_node,sample_df,max_node)
        left_port_ret=gen_port_ret(new_left_df)
        port_df[tree_df.shape[0]-2]=left_port_ret
        
        new_right_df=filter_panel_by_node(new_right_node,sample_df,max_node)
        right_port_ret=gen_port_ret(new_right_df)        
        port_df[tree_df.shape[0]-1]=right_port_ret
        
        if best_split["node"]>0:
            port_df.drop(best_split["node"],axis=1,inplace=True)
        
    return tree_df,log_df,port_df
        

if __name__ == "__main__":
    n_period = 10
    n_stock = 100
    n_feat = 3
    thres_list = [-0.6,-0.2,0.2,0.6]
    max_node = 4
    min_node_size= 2
    lambda_mu=0.0001
    lambda_cov=0.0001
    abs_norm=True    
    boost_no=3
    
    sample_df=gen_ran_data(n_stock,n_period,n_feat)
    prior_tree_factors_df=pd.DataFrame(index=sample_df["date"].unique())
    
    for boost_no in range(boost_no):
        tree_df,log_df,port_df=gen_tree(sample_df.copy(),n_period,n_stock,n_feat,thres_list,max_node,min_node_size,prior_tree_factors_df,lambda_mu,lambda_cov,abs_norm)
        prior_tree_factors_df=pd.concat([prior_tree_factors_df,pd.DataFrame(shrink_weight_factor(port_df,0,0,False),columns=[boost_no])],axis=1)
        
    print("stop")