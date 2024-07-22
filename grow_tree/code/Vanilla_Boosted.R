library(TreeFactor)
library(dplyr)

tf_residual <- function(fit, Y, Z, H, months, no_H) {
    # Tree Factor Models
    regressor <- Z
for (j in 1:dim(Z)[2])
    {
        regressor[, j] <- Z[, j] * fit$ft[months + 1]
    }
    if (!no_H) {
        regressor <- cbind(regressor, H)
    }
    # print(fit$R2*100)
    x <- as.matrix(regressor)
    y <- Y
    b_tf <- solve(t(x) %*% x) %*% t(x) %*% y
    haty <- (x %*% b_tf)[, 1]
    print(b_tf)
    return(Y - haty)
}
#set parameters for sample 1980-2000
#set work dir for debugging
#setwd("/mnt/work/hc2235/Panel_Tree_replication/grow_tree/code")
print("flag 1")
m_train_df = read.csv(file.path("..","..","data_preparation","output","weighted_trainp.csv"))
m_train_df['cons'] = 1 
all_chars = paste0("f", 1:50)
first_split_var = c(0:49)                                                       #variables that first split must happen on
second_split_var = c(0:49)                                                      #variables that second split must happen on
X_train=m_train_df[,all_chars]                                                  #stock characters, long format
R_train=m_train_df[,c("RET")]                                                   #return, long format
months_train = as.numeric(as.factor(m_train_df[,c("date")])) - 1                #month index, start from 0
stocks_train = as.numeric(as.factor(m_train_df[,c("PERMNO")])) - 1              #stock index, start from 0
portfolio_weight_train = m_train_df[,c("weight")]                               #stock weight
num_months = length(unique(months_train))                                 
num_stocks = length(unique(stocks_train))                                 
min_leaf_size = 20
max_depth = 10
num_iter = 9
num_cutpoints = 4
equal_weight = FALSE
lambda=0.0001                                                                   #avoid matrix inversion error
loss_weight_train = m_train_df[,c("cons")]                                      #weight to each obs when calculating loss
Z_train = as.matrix(m_train_df[,c("cons")])                                    # instrument for conditional factor model, 1 for vanilla MSRR
Y_train1 = m_train_df[,c("cons")]                                               #return if minimizing individual stock pricing error, 1 for vanilla MSRR
H_train1 = m_train_df[,c("cons")]                                               #predefined market factors (e.g. MKT) for conditional factor model
H_train1 = H_train1 * Z_train                                                   #no difference for no_H=TRUE
eta=1                                                                           #1 means MVE weights for basis portfolios, 0 means equal weights 
no_H = TRUE                                                                     #make the pricing loss regression Rt ~ Zt * Ft + Ht
abs_normalize = FALSE                                                           #apply abs to the weights for basis portfolios, then normalize
weighted_loss = FALSE                                                           #whether to apply loss weight
stop_no_gain = FALSE                                                            #stop splitting when no reduction in loss
print("flag 2")

#fit the first tree and output leaf node for each observation (for table 1)

fit1 = TreeFactor_APTree(R_train, Y_train1, X_train, Z_train, H_train1, portfolio_weight_train,
                         loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, num_stocks, 
                         num_months, min_leaf_size, max_depth, num_iter, num_cutpoints, eta, equal_weight, 
                         no_H, abs_normalize, weighted_loss, stop_no_gain, lambda, lambda)
pred_train = predict(fit1, X_train, R_train, months_train, portfolio_weight_train)

write.csv(pred_train$leaf_index,file.path("..","output","train_tree_leaf_index.csv"),row.names=FALSE)
write.csv(fit1$ft, file.path("..", "output", "Vanilla_First_factor.csv"), row.names = FALSE)

# fit boosted trees and output tree MVE portfolio (tree factor) returns for each tree (for table 2)
Y_train_boost = tf_residual(fit1, Y_train1, Z_train, H_train1, months_train, no_H)  # initialize output variables as residuals from the first tree
tree_factors_df <- matrix(fit1$ft, nrow = length(fit1$ft), ncol = 20, dimnames = list(NULL, paste0("ft", 1:20)))

for (i in 1:20) {

    fit = TreeFactor_APTree(
        R_train, Y_train_boost, X_train, Z_train, H_train1, portfolio_weight_train,
        loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, num_stocks,
        num_months, min_leaf_size, max_depth, num_iter, num_cutpoints, eta, equal_weight,
        no_H, abs_normalize, weighted_loss, stop_no_gain, lambda, lambda
    )
    Y_train_boost = tf_residual(fit, Y_train_boost, Z_train, H_train1, months_train, no_H)
    print("tree grown")
    tree_factors_df[, paste0("ft", i)] <- fit$ft
    write.csv(tree_factors_df, file.path("..", "output", "Vanilla_Boosted_factors.csv"), row.names = FALSE)

}

#-----------------------------------------------------------------------
#Repeat for sample 2000-2020


m_test_df = read.csv(file.path("..","..","data_preparation","output","weighted_testp.csv"))
m_test_df['cons'] = 1
all_chars = paste0("f", 1:50)
first_split_var = c(0:49)
second_split_var = c(0:49)
X_test=m_test_df[,all_chars]
R_test=m_test_df[,c("RET")]
months_test = as.numeric(as.factor(m_test_df[,c("date")]))
months_test = months_test - 1 # start from 0
stocks_test = as.numeric(as.factor(m_test_df[,c("PERMNO")])) - 1
portfolio_weight_test = m_test_df[,c("weight")] 
num_months = length(unique(months_test))
num_stocks = length(unique(stocks_test))
min_leaf_size = 20
max_depth = 10
num_iter = 9
num_cutpoints = 4
equal_weight = FALSE
lambda=0.0001
loss_weight_test = m_test_df[,c("cons")]
Z_test = as.matrix(m_test_df[, c("cons")])
Y_test1 = m_test_df[,c("cons")]
H_test1 = m_test_df[,c("cons")]
H_test1 = H_test1 * Z_test
eta=1
no_H = TRUE
abs_normalize = FALSE
weighted_loss = FALSE
stop_no_gain = FALSE


fit2 = TreeFactor_APTree(R_test, Y_test1, X_test, Z_test, H_test1, portfolio_weight_test,
                         loss_weight_test, stocks_test, months_test, first_split_var, second_split_var, num_stocks, 
                         num_months, min_leaf_size, max_depth, num_iter, num_cutpoints, eta, equal_weight, 
                         no_H, abs_normalize, weighted_loss, stop_no_gain, lambda, lambda)
pred = predict(fit2, X_test, R_test, months_test, portfolio_weight_test)
write.csv(pred$leaf_index,file.path("..","output","test_tree_leaf_index_7_21_2024.csv"),row.names=FALSE)

write.csv(fit2$ft, file.path("..", "output", "Vanilla_First_factor_test.csv"), row.names = FALSE)

# fit boosted trees and output tree MVE portfolio (tree factor) returns for each tree (for table 2)
Y_test_boost <- tf_residual(fit2, Y_test1, Z_test, H_test1, months_test, no_H) # initialize output variables as residuals from the first tree
tree_factors_df_test <- matrix(fit2$ft, nrow = length(fit2$ft), ncol = 20, dimnames = list(NULL, paste0("ft", 1:20)))

for (i in 1:20) {
    fit <- TreeFactor_APTree(
        R_test, Y_test_boost, X_test, Z_test, H_test1, portfolio_weight_test,
        loss_weight_test, stocks_test, months_test, first_split_var, second_split_var, num_stocks,
        num_months, min_leaf_size, max_depth, num_iter, num_cutpoints, eta, equal_weight,
        no_H, abs_normalize, weighted_loss, stop_no_gain, lambda, lambda
    )
    Y_test_boost <- tf_residual(fit, Y_test_boost, Z_test, H_test1, months_test, no_H)
    print("tree grown")
    tree_factors_df_test[, paste0("ft", i)] <- fit$ft
    write.csv(tree_factors_df_test, file.path("..", "output", "Vanilla_Boosted_factors_test.csv"), row.names = FALSE)

}

