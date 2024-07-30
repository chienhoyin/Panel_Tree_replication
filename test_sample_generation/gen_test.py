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

def gen_ran_tree(n_feat,max_node):
    
    node_cols=[f"node_{i}_parent" for i in range(2*max_node-1)]
    tree_df=pd.DataFrame([[999,999,True]+["Irr"]*(2*max_node-1)], columns=["char","thres","leaf"]+node_cols)
    for split in range(max_node-1):
        
        best_split={"char":feat,"thres":thres,"split":split,"node":ind}
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
        left_port_ret=gen_port_ret(new_left_df)
        port_df[tree_df.shape[0]-2]=left_port_ret
        
        new_right_df=filter_panel_by_node(new_right_node,sample_df,max_node)
        right_port_ret=gen_port_ret(new_right_df)        
        port_df[tree_df.shape[0]-1]=right_port_ret
        
        if best_split["node"]>0:
            port_df.drop(best_split["node"],axis=1,inplace=True)
        

def gen_ran_data(n_stock,n_period,n_feat):
    
    X = np.vstack([np.vstack([np.random.permutation(np.linspace(-1,1,n_stock)) for _ in range(n_feat)]).T for n in range(n_period)])
    cols=[f"f{_}" for _ in range(n_feat)]
    sample_df=pd.DataFrame(X,columns=cols)
    dates = pd.date_range('2000-01-01', periods=n_period, freq='MS')
    sample_df["date"] = [date for date in dates for _ in range(n_stock)]
    sample_df["PERMNO"] = list(range(n_stock))*n_period
    sample_df["RET"] = np.random.randn(n_stock*n_period) + 0.5
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
    sample_df["RET"]= np.nan
    for ind,row in tree_df[tree_df["leaf"]].iterrows():
        temp_df = sample_df.iloc[:,:-1].copy()
        no_nodes_col = len(tree_df.filter(regex=re.compile(r'^node_')).columns)
        
        for col_no in range(no_nodes_col):
            if row[f"node_{col_no}_parent"] == "left":
                temp_df = temp_df[temp_df[tree_df.loc[col_no,"char"]]<=tree_df.loc[col_no,"thres"]]
            elif row[f"node_{col_no}_parent"] == "right":
                temp_df = temp_df[temp_df[tree_df.loc[col_no,"char"]]>tree_df.loc[col_no,"thres"]]
        
        assert min([temp_df[temp_df["date"]==date].shape[0] for date in dates]) >= min_node_size, "min node size requirement violated"
        
        ret_df=pd.DataFrame(np.random.randn(len(dates))+0.5,columns=["RET"],index=dates)
        temp_df=pd.merge(left=temp_df,right=ret_df,how="left",left_on="date",right_index=True,validate="m:1")
        sample_df.loc[temp_df.index,"RET"]=temp_df["RET"]
    
    sample_df["weight"] = 1/n_stock
    
    return sample_df




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
        left_port_ret=gen_port_ret(new_left_df)
        port_df[tree_df.shape[0]-2]=left_port_ret
        
        new_right_df=filter_panel_by_node(new_right_node,sample_df,max_node)
        right_port_ret=gen_port_ret(new_right_df)        
        port_df[tree_df.shape[0]-1]=right_port_ret
        
        if best_split["node"]>0:
            port_df.drop(best_split["node"],axis=1,inplace=True)
        
        print(split)
        
    return tree_df,log_df,port_df
        

if __name__ == "__main__":
    
    '''functions:   -gen_ran_data() generates data with independent elements of X, and normally distributed Y, independent as well
                    -gen_data_by_tree() generates data according to a given tree, where all the stocks within the same leaf node panel at each time have  
                    the exact same return. This gurantees that if the tree growing algorithm works correctly, and that if n_period and n_stock are large enough,
                    the input tree can be recovered
                    -gen_tree() grows a tree that maximizes SR given data
    '''
    
    # test 1: Generate an arbitrary tree. Use this tree to generate a sample. Grow a tree using that sample and see if the input tree can be recovered
    # Result: Failed. The way that the sample is generated from the given tree makes it so mutliple trees can be optimal
    '''
    np.random.seed(1234)
    n_period = 1000
    n_stock = 50
    n_feat = 3
    thres_list = [-0.6,-0.2,0.2,0.6]
    max_node = 4
    min_node_size= 2
    lambda_mu=0.0001
    lambda_cov=0.0001
    abs_norm=True    
    boost_no=1
    '''
    #Test 2: Analytically solving the SR of the splits and optimal splits, and checking with the algorithm

    #sample_df=gen_ran_data(n_stock,n_period,n_feat)
    #sample_df.to_csv("/mnt/work/hc2235/Panel_Tree_replication/test_sample_generation/test_sample.csv",index=False)
    
    #result: successfully reconciled SR
    '''
    tree_df=pd.read_csv("/mnt/work/hc2235/Panel_Tree_replication/test_sample_generation/test_sample_tree.csv")
    
    new_sample_df=gen_data_by_tree(tree_df,n_stock,n_period,n_feat,thres_list,max_node,min_node_size)
    
    prior_tree_factors_df=pd.DataFrame(index=new_sample_df["date"].unique())
    new_tree_df,log_df,port_df=gen_tree(new_sample_df.copy(),n_period,n_stock,n_feat,thres_list,max_node,min_node_size,prior_tree_factors_df,lambda_mu,lambda_cov,abs_norm)
    '''
    # Test 3: Using the actual training sample (toy version, only 3 feats) to grow a small boosted tree (only 3 nodes), check with C++ algo result  
    
    np.random.seed(1234)

    sample_df=pd.read_csv("/mnt/work/hc2235/Panel_Tree_replication/test_sample_generation/weighted_trainp_loss_weight_toy.csv")
    n_period = len(sample_df["date"].unique())
    n_stock = len(sample_df["PERMNO"].unique())   
    n_feat = 3
    sample_df.rename(columns={f"f{_}":f"f{_-1}" for _ in range(1,n_feat+1)},inplace=True)
    thres_list = [-0.6,-0.2,0.2,0.6]
    max_node = 3
    min_node_size= 20
    lambda_mu=0.0001
    lambda_cov=0.0001
    abs_norm=True    
    boost_no=5
    
    
    prior_tree_factors_df=pd.DataFrame(index=sample_df["date"].unique())
    tree_df_list=[]
    log_df_list=[]
    port_df_list=[]
    
    for boost_no in range(boost_no):
        tree_df,log_df,port_df=gen_tree(sample_df.copy(),n_period,n_stock,n_feat,thres_list,max_node,min_node_size,prior_tree_factors_df,lambda_mu,lambda_cov,abs_norm)
        w,tree_factor=shrink_weight_factor(port_df,0,0,False)
        prior_tree_factors_df=pd.concat([prior_tree_factors_df,pd.DataFrame(tree_factor,columns=[boost_no])],axis=1)
        tree_df_list.append(tree_df)
        log_df_list.append(log_df)
        port_df_list.append(port_df)
    
    print("stop")
        
    
    