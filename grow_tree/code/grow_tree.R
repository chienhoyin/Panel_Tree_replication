library(TreeFactor)
library(dplyr)

#import weighted and winsorized train data

m_train_df = read.csv(file.path('..","..","data_preparation","output","weighted_trainp.csv'))

#create constant column for tree input

m_train_df['cons'] = 1

#specify tree input that affects portfolio return

all_chars = paste0("f", 1:50)

first_split_var = c(0:49)

second_split_var = c(0:49)

X_train=m_train_df[,all_chars]

R_train=m_train_df[,c("RET")]

months_train = as.numeric(as.factor(m_train_df[,c("date")]))

months_train = months_train - 1 # start from 0

stocks_train = as.numeric(as.factor(m_train_df[,c("PERMNO")])) - 1

portfolio_weight_train = m_train_df[,c("weight")] 

num_months = length(unique(months_train))

num_stocks = length(unique(stocks_train))

min_leaf_size = 20

max_depth = 10

num_iter = 9

num_cutpoints = 4

equal_weight = FALSE

lambda=0.0001


#specify tree input that does not affect portfolio return following demo1/main.R

loss_weight_train = m_train_df[,c("cons")]

instruments = all_chars

first_split_var_boosting = first_split_var

second_split_var_boosting = first_split_var

Z_train = m_train_df[, instruments]

Z_train = cbind(1, Z_train)

Y_train1 = m_train_df[,c("RET")]

H_train1 = m_train_df[,c("cons")]

H_train1 = H_train1 * Z_train

eta=1

no_H = TRUE

abs_normalize = TRUE

weighted_loss = FALSE

stop_no_gain = FALSE


#fit the tree

fit1 = TreeFactor_APTree(R_train, Y_train1, X_train, Z_train, H_train1, portfolio_weight_train,
                         loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, num_stocks, 
                         num_months, min_leaf_size, max_depth, num_iter, num_cutpoints, eta, equal_weight, 
                         no_H, abs_normalize, weighted_loss, stop_no_gain, lambda, lambda)

#get output portfolio

pred_train = predict(fit1, X_train, R_train, months_train, portfolio_weight_train)

write.csv(pred_train$leaf_index,file.path("..","output","train_first_tree_leaf_index.csv"),row.names=FALSE)


#load weighted and winsorized test data
m_test_df = read.csv(file.path("..","..","data_preparation","output","weighted_testp.csv"))

X_test=m_test_df[,all_chars]
R_test=m_test_df[,c("RET")]

portfolio_weight_test = m_test_df[,c("weight")] 

months_test = as.numeric(as.factor(m_test_df[,c("date")]))
months_test = months_test - 1 # start from 0

pred = predict(fit1, X_test, R_test, months_test, portfolio_weight_test)

write.csv(pred$leaf_index,file.path("..","output","test_first_tree_leaf_index.csv"),row.names=FALSE)

