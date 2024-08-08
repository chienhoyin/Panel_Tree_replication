import pandas as pd
import numpy as np
import os

def shrink_weight_factor(X,lambda_mu,lambda_cov,abs_norm,short_if_avg_neg=True):
    
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

def filter_panel_by_node(node_row,tree_df,sample_df,max_node):
    
    temp_df=sample_df.copy()
    for parent_node_no in range(2*max_node-1):
        if node_row[f"node_{parent_node_no}_parent"] == "left":
            temp_df=temp_df[temp_df[tree_df.loc[parent_node_no,"char"]]<=tree_df.loc[parent_node_no,"thres"]]
        elif node_row[f"node_{parent_node_no}_parent"] == "right":
            temp_df=temp_df[temp_df[tree_df.loc[parent_node_no,"char"]]>tree_df.loc[parent_node_no,"thres"]]
            
    return temp_df

def gen_port_ret(node_sample_df):
        
    ret=node_sample_df.groupby("date").apply(lambda x: (x["RET"]*x["weight"]).sum()/(x["weight"].sum()))
        
    return ret
    
def construct_factor(tree_df,train_sample_df,max_node,abs_norm,short_if_neg):
    
    train_date_range=train_sample_df["date"].unique()
    node_train_ret_df=pd.DataFrame(index=train_date_range)
    leaf_df=tree_df[tree_df["leaf"]]
    
    for ind, row in leaf_df.iterrows():
        node_train_df=filter_panel_by_node(row,tree_df,train_sample_df,max_node)
        node_train_ret_df[f"node_{ind}_ret"]=gen_port_ret(node_train_df)
    
    w, is_factor = shrink_weight_factor(node_train_ret_df,lambda_mu,lambda_cov,abs_norm,short_if_neg)
        
    return is_factor

n_feat=50
max_node=10
max_boost_no=20
lambda_mu=0.0001
lambda_cov=0.0001

#predict train sample for checking

sample_df=pd.read_csv("/mnt/work/hc2235/Panel_Tree_replication/data_preparation/output/weighted_trainp_loss_weight_rf.csv",index_col=0)
sample_df.rename(columns={f"f{_}":f"f{_-1}" for _ in range(1,n_feat+1)},inplace=True)
tree_df_list=[pd.read_csv(f"/mnt/work/hc2235/Panel_Tree_replication/grow_tree/output/Vanilla_Boosted_train_tree_{_}_rf.csv",index_col=0) for _ in range(max_boost_no)]
factors_df=pd.concat([construct_factor(tree_df_list[_], sample_df,max_node,True, True, "in-sample").rename(_) for _ in range(max_boost_no)],axis=1)         
factors_df.to_csv("/mnt/work/hc2235/Panel_Tree_replication/grow_tree/output/Vanilla_Boosted_train_factor_rf.csv")

#predict test sample

test_sample_df=pd.read_csv("/mnt/work/hc2235/Panel_Tree_replication/data_preparation/output/weighted_testp_loss_weight_rf.csv",index_col=0)
test_sample_df.rename(columns={f"f{_}":f"f{_-1}" for _ in range(1,n_feat+1)},inplace=True)
tree_df_list=[pd.read_csv(f"/mnt/work/hc2235/Panel_Tree_replication/grow_tree/output/Vanilla_Boosted_train_tree_{_}_rf.csv",index_col=0) for _ in range(max_boost_no)]      
factors_df=pd.concat([construct_factor(tree_df_list[_], test_sample_df,max_node,True, True).rename(_) for _ in range(max_boost_no)],axis=1)         
factors_df.to_csv("/mnt/work/hc2235/Panel_Tree_replication/grow_tree/output/Vanilla_Boosted_OOS_factor_rf_8_7_2024.csv")
