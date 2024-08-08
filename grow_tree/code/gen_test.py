import numpy as np
import pandas as pd
import os
import re
import statsmodels.api as sm
import multiprocessing as mp
import itertools
from numba import jit
import time

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


def beta_Sharpe(X):
    
    n = X.shape[0]
    X = X.values.reshape(n, -1) #reshape to 2d array in case X is 1d
    y = np.ones(n)
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    residuals = y - X @ beta
    rss = np.sum(residuals**2)
    sr =  (n / rss - 1)**0.5
    
    return beta, sr


def filter_panel_by_node(node_row,sample_df,max_node):
    
    temp_df=sample_df.copy()
    for parent_node_no in range(2*max_node-1):
        if node_row[f"node_{parent_node_no}_parent"] == "left":
            temp_df=temp_df[temp_df[f"split_{parent_node_no}"]]
        elif node_row[f"node_{parent_node_no}_parent"] == "right":
            temp_df=temp_df[~temp_df[f"split_{parent_node_no}"]]
            
    return temp_df.index

def split_panel(node_sample_df,feat,thres):
        
        left_df=node_sample_df[node_sample_df[feat]<=thres]
        right_df=node_sample_df[node_sample_df[feat]>thres]
        
        return left_df,right_df
    
def gen_port_ret(node_sample_df):
        
        ret=node_sample_df.groupby("date").apply(lambda x: (x["RET"]*x["weight"]).sum()/(x["weight"].sum()))
        
        return ret

def check_node_size(node_sample_df,min_node_size,date_range):
    if min([node_sample_df[node_sample_df["date"]==date].shape[0] for date in date_range]) < min_node_size:
        return False
    else:
        return True

def gen_log(node_sample_df_index,port_df,ind,feat,thres,split,date_range,max_node,min_node_size,prior_tree_factors_df,lambda_mu,lambda_cov,abs_norm):
    
    node_sample_df=sample_df.loc[node_sample_df_index]
    left_df,right_df=split_panel(node_sample_df,feat,thres)
    if (not check_node_size(left_df,min_node_size,date_range)) or (not check_node_size(right_df,min_node_size,date_range)):
        return pd.DataFrame({"char":feat,"thres":thres,"split":split,"node":ind,"sr":np.nan},index=[0])
         
    left_port_ret=gen_port_ret(left_df)
    right_port_ret=gen_port_ret(right_df)
    temp_port_df=pd.concat([left_port_ret,right_port_ret,port_df],axis=1)
    
    w, tree_factor=shrink_weight_factor(temp_port_df,lambda_mu,lambda_cov,abs_norm,short_if_avg_neg=True)
    beta , sr = beta_Sharpe(pd.concat([tree_factor,prior_tree_factors_df],axis=1))
    cov = pd.concat([left_port_ret,right_port_ret],axis=1).cov()
    
    new_row={"char":feat,"thres":thres,"split":split,"node":ind,"sr":sr,"left_var":cov.iloc[0,0],"right_var":cov.iloc[1,1],"cov":cov.iloc[0,1]}
    
    factor_beta_filled = list(beta) + [0] * (boost_no - len(beta))
    new_row.update({f"factor_beta_{_}":beta[_] for _ in range(boost_no)})
    w_filled = list(w) + [0] * (max_node - len(w))
    new_row.update({f"w_{_}":w_filled[_] for _ in range(max_node)})
    new_row=pd.DataFrame(new_row,index=[0])
    
    return new_row
    
def gen_tree_mp(no_core,sample_df,n_period,n_stock,n_feat,thres_list,max_node,min_node_size,prior_tree_factors_df,lambda_mu,lambda_cov,abs_norm):
   
    node_cols=[f"node_{i}_parent" for i in range(2*max_node-1)]
    tree_df=pd.DataFrame([[999,999,True]+["Irr"]*(2*max_node-1)], columns=["char","thres","leaf"]+node_cols)
    port_df=pd.DataFrame(index=sample_df["date"].unique())
    feat_list=[f"f{_}" for _ in range(n_feat)]
    log_df=pd.DataFrame()
    date_range=sample_df["date"].unique()
    
    for split in range(max_node-1):
        
        leaf_df=tree_df[tree_df["leaf"]]
        start_time = time.time()
        input_list=[(filter_panel_by_node(leaf_df.loc[ind],sample_df,max_node),
                     port_df,
                     ind,
                     feat,
                     thres,
                     split,
                     date_range,
                     max_node,
                     min_node_size,
                     prior_tree_factors_df,
                     lambda_mu,
                     lambda_cov,
                     abs_norm) for ind,feat,thres in itertools.product(leaf_df.index,feat_list,thres_list)]

        pool=mp.Pool(no_core)
        results=pool.starmap(gen_log,input_list)
        pool.close()
        pool.join()

        log_df=pd.concat([log_df]+results,ignore_index=True)
        
        best_split=log_df[(log_df["split"]==split)]
        best_split=best_split[(best_split["sr"]==best_split["sr"].max())].iloc[0]
        
        new_left_node={"char":"NA","thres":"NA","leaf":True}
        new_left_node.update(tree_df.loc[best_split["node"]][node_cols].to_dict())
        new_left_node.update({f"node_{best_split['node']}_parent":"left"})
        tree_df=pd.concat([tree_df,pd.DataFrame(new_left_node,index=[0])],ignore_index=True)
        
        new_right_node={"char":"NA","thres":"NA","leaf":True}
        new_right_node.update(tree_df.loc[best_split["node"]][node_cols].to_dict())
        new_right_node.update({f"node_{best_split['node']}_parent":"right"})
        tree_df=pd.concat([tree_df,pd.DataFrame(new_right_node,index=[0])],ignore_index=True)
        tree_df.loc[best_split["node"],"leaf"]=False
        tree_df.loc[best_split["node"],"char"]=best_split["char"]
        tree_df.loc[best_split["node"],"thres"]=best_split["thres"]

        sample_df["split_"+str(best_split["node"])]=sample_df[best_split["char"]]<=best_split["thres"]
        
        new_left_ind=filter_panel_by_node(new_left_node,sample_df,max_node)
        left_port_ret=gen_port_ret(sample_df.loc[new_left_ind])
        port_df[tree_df.shape[0]-2]=left_port_ret
        
        new_right_ind=filter_panel_by_node(new_right_node,sample_df,max_node)
        right_port_ret=gen_port_ret(sample_df.loc[new_right_ind])        
        port_df[tree_df.shape[0]-1]=right_port_ret
        
        if best_split["node"]>0:
            port_df.drop(best_split["node"],axis=1,inplace=True)
        print("--- %s seconds ---" % (time.time() - start_time))
        print(split)
        
    return tree_df,log_df,port_df

def gen_tree(sample_df,n_period,n_stock,n_feat,thres_list,max_node,min_node_size,prior_tree_factors_df,lambda_mu,lambda_cov,abs_norm):
    
    #sample_df.sort_values(by=["date","PERMNO"],inplace=True)
    node_cols=[f"node_{i}_parent" for i in range(2*max_node-1)]
    
    tree_df=pd.DataFrame([[999,999,True]+["Irr"]*(2*max_node-1)], columns=["char","thres","leaf"]+node_cols)
    port_df=pd.DataFrame(index=sample_df["date"].unique())
    feat_list=[f"f{_}" for _ in range(n_feat)]
    log_df=pd.DataFrame()
    date_range=sample_df["date"].unique()
    
    for split in range(max_node-1):
        for ind,row in tree_df[tree_df["leaf"]].iterrows():
            node_sample_df=filter_panel_by_node(row,sample_df,max_node)
            #node_sample_ind=filter_panel_by_node(row,sample_df,max_node)
            #node_sample_df=sample_df.loc[node_sample_ind]
            for feat in feat_list:
                for thres in thres_list:
                        left_df,right_df=split_panel(node_sample_df,feat,thres)
                        if (not check_node_size(left_df,min_node_size,date_range)) or (not check_node_size(right_df,min_node_size,date_range)):
                            log_df=pd.concat([log_df,pd.DataFrame({"char":feat,"thres":thres,"split":split,"node":ind,"sr":np.nan},index=[0])],ignore_index=True)
                            continue
                        left_port_ret=gen_port_ret(left_df)
                        right_port_ret=gen_port_ret(right_df)
                        temp_port_df=pd.concat([left_port_ret,right_port_ret,port_df],axis=1)
                        
                        w, tree_factor=shrink_weight_factor(temp_port_df,lambda_mu,lambda_cov,abs_norm)
                        beta , sr = beta_Sharpe(pd.concat([tree_factor,prior_tree_factors_df],axis=1))
                        cov = pd.concat([left_port_ret,right_port_ret],axis=1).cov()
                        
                        new_row={"char":feat,"thres":thres,"split":split,"node":ind,"sr":sr,"left_var":cov.iloc[0,0],"right_var":cov.iloc[1,1],"cov":cov.iloc[0,1]}
                        
                        factor_beta_filled = list(beta) + [0] * (boost_no - len(beta))
                        new_row.update({f"factor_beta_{_}":beta[_] for _ in range(boost_no)})
                        w_filled = list(w) + [0] * (max_node - len(w))
                        new_row.update({f"w_{_}":w_filled[_] for _ in range(max_node)})
                        
                        log_df=pd.concat([log_df,pd.DataFrame(new_row,index=[0])],ignore_index=True)
                        
                        print(thres)
                print(feat)
            print(ind)
        best_split=log_df[(log_df["split"]==split)]
        best_split=best_split[(best_split["sr"]==best_split["sr"].max())].iloc[0]
        
        new_left_node={"char":"NA","thres":"NA","leaf":True}
        new_left_node.update(tree_df.loc[best_split["node"]][node_cols].to_dict())
        new_left_node.update({f"node_{best_split['node']}_parent":"left"})
        tree_df=pd.concat([tree_df,pd.DataFrame(new_left_node,index=[0])],ignore_index=True)
        
        new_right_node={"char":"NA","thres":"NA","leaf":True}
        new_right_node.update(tree_df.loc[best_split["node"]][node_cols].to_dict())
        new_right_node.update({f"node_{best_split['node']}_parent":"right"})
        tree_df=pd.concat([tree_df,pd.DataFrame(new_right_node,index=[0])],ignore_index=True)
        tree_df.loc[best_split["node"],"leaf"]=False
        tree_df.loc[best_split["node"],"char"]=best_split["char"]
        tree_df.loc[best_split["node"],"thres"]=best_split["thres"]

        sample_df["split_"+str(best_split["node"])]=sample_df[best_split["char"]]<=best_split["thres"]
        
        new_left_df=filter_panel_by_node(new_left_node,sample_df,max_node)
        #new_left_df=sample_df.loc[filter_panel_by_node(new_left_node,sample_df,max_node)]
        left_port_ret=gen_port_ret(new_left_df)
        port_df[tree_df.shape[0]-2]=left_port_ret
        
        new_right_df=filter_panel_by_node(new_right_node,sample_df,max_node)
        #new_right_df=sample_df.loc[filter_panel_by_node(new_right_node,sample_df,max_node)]
        right_port_ret=gen_port_ret(new_right_df)        
        port_df[tree_df.shape[0]-1]=right_port_ret
        
        if best_split["node"]>0:
            port_df.drop(best_split["node"],axis=1,inplace=True)
        
        print(split)
        
    return tree_df,log_df,port_df



if __name__ == "__main__":
    
    np.random.seed(1234)

    sample_df=pd.read_csv("/mnt/work/hc2235/Panel_Tree_replication/data_preparation/output/weighted_trainp_loss_weight_rf.csv")

    #set parameters
    
    n_period = len(sample_df["date"].unique())
    n_stock = len(sample_df["PERMNO"].unique())   
    n_feat = 50
    sample_df.rename(columns={f"f{_}":f"f{_-1}" for _ in range(1,n_feat+1)},inplace=True)
    thres_list = [-0.6,-0.2,0.2,0.6]
    max_node = 10
    min_node_size= 20
    lambda_mu=0.0001
    lambda_cov=0.0001
    abs_norm=True
    short_if_avg_neg=True
    max_boost_no=20
    
    #initialize
    prior_tree_factors_df=pd.DataFrame(index=sample_df["date"].unique())
    tree_df_list=[]
    log_df_list=[]
    port_df_list=[]
    
    for boost_no in range(max_boost_no):

        tree_df,log_df,port_df=gen_tree_mp(50, sample_df.copy(),n_period,n_stock,n_feat,thres_list,max_node,min_node_size,prior_tree_factors_df,lambda_mu,lambda_cov,abs_norm)
        w,tree_factor=shrink_weight_factor(port_df,lambda_mu,lambda_cov,abs_norm,short_if_avg_neg)
        prior_tree_factors_df=pd.concat([prior_tree_factors_df,pd.DataFrame(tree_factor,columns=[boost_no])],axis=1)
        tree_df_list.append(tree_df)
        log_df_list.append(log_df)
        port_df_list.append(port_df)
        print(f"Tree {boost_no} grown")
        tree_df.to_csv(f"/mnt/work/hc2235/Panel_Tree_replication/grow_tree/output/Vanilla_Boosted_train_tree_{boost_no}_rf.csv")
        log_df.to_csv(f"/mnt/work/hc2235/Panel_Tree_replication/grow_tree/output/Vanilla_Boosted_train_log_{boost_no}_rf.csv")
        port_df.to_csv(f"/mnt/work/hc2235/Panel_Tree_replication/grow_tree/output/Vanilla_Boosted_train_port_{boost_no}_rf.csv")
        prior_tree_factors_df.to_csv(f"/mnt/work/hc2235/Panel_Tree_replication/grow_tree/output/Vanilla_Boosted_train_factor_{boost_no}_rf.csv")
     
    