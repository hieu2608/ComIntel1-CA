### Setup data for winequality
pkgs <- c('doParallel', 'foreach', 'grnn', 'RSNNS', 'nnet')
lapply(pkgs, require, character.only = T)
registerDoParallel(cores=8)

library(grnn)
set.seed(123)

data = read.csv(file=file.path("data", "winequality-white.csv"), header=TRUE)

### Pre-processing data
X = normalizeData(data[,-ncol(data)], type="norm") 
Y = data[,ncol(data)]
wine_data <- data.frame(X, Y)

### Setup data for testing
wine_size=nrow(wine_data)
wine_length=ncol(wine_data)
wine_index <- 1:wine_size

wine_positions <- sample(wine_index, trunc(wine_size * 0.75))

wine_training <- wine_data[wine_positions,]
wine_training_input <- wine_training[,-wine_length]
wine_training_target <- wine_training[,wine_length]
wine_testing <- wine_data[-wine_positions,]
wine_testing_input <- wine_testing[,-wine_length]
wine_testing_target <- wine_testing[,wine_length]

# ### Training and testing using nnet
# nnet_training_input <- wine_training_input
# nnet_training_target <- data.frame(wine_training_target)
# nnet_training <- data.frame(nnet_training_input, nnet_training_target)
# 
# nnet_model <- nnet(nnet_training, size=1000, 
#                    rang=0.1, decay=5e-4, maxit=200)
# nnet_pred<- predict(nnet_model, wine_testing_input)
# # table(true=wine_testing_target, predicted=nnet_pred)
# nnet_accuracy = mean(wine_testing_target == nnet_pred)
# nnet_test_predict <- data.frame(actual=wine_testing_target,
#                                 predict=nnet_pred)

### Training for MLP
mlp_training_input <- wine_training_input
mlp_training_target <- wine_training_target
mlp_test_input <- wine_testing_input
mlp_test_target <- wine_testing_target

mlp_sizes = c(10, 25, 50, 75, 100)
mlp_analysis <- foreach(s = mlp_sizes, .combine=rbind) %dopar% {
  start <- Sys.time()
  model <- mlp(mlp_training_input, mlp_training_target, size=s, learnFunc = "Rprop", maxit=100, linOut=TRUE)
  test_predict <- data.frame(actual=mlp_test_target, predict=predict(model, mlp_test_input))
  accuracy <- mean(test_predict$actual == round(test_predict$predict))
  result = data.frame(s, accuracy, time=Sys.time()-start)
}
mlp_size = mlp_analysis[mlp_analysis$accuracy == max(mlp_analysis$accuracy),1]
mlp_model = mlp(mlp_training_input, mlp_training_target, size=mlp_size, learnFunc = "Rprop", maxit=100, linOut=TRUE)
mlp_test_predict <- data.frame(actual=mlp_test_target, predict=predict(mlp_model, mlp_test_input))
mlp_accuracy <- mean(mlp_test_predict$actual == round(mlp_test_predict$predict))
cat(mlp_accuracy)

### Training and testing for RBF
rbf_training_input <- wine_training_input
rbf_training_target <- decodeClassLabels(wine_training_target)
rbf_test_input <- wine_testing_input
rbf_test_target <- encodeClassLabels(decodeClassLabels(wine_testing_target))

rbf_model = rbf(rbf_training_input, rbf_training_target, size=40, maxit=1000, 
                initFuncParams=c(0, 1, 0, 0.01, 0.01), 
                learnFuncParams=c(1e-8, 0, 1e-8, 0.1, 0.8), 
                linOut = TRUE)
rbf_test_predict <- data.frame(actual=rbf_test_target, predict=encodeClassLabels(predict(rbf_model, rbf_test_input)))
rbf_accuracy <- mean(rbf_test_predict$actual == rbf_test_predict$predict)
table(rbf_test_predict$actual, round(rbf_test_predict$predict))

### Define a function for prediction given a GRNN
grnn_training_input <- wine_training_input
grnn_training_target <- wine_training_target
grnn_training <- data.frame(grnn_training_input, grnn_training_target)
grnn_test_input <- wine_testing_input
grnn_test_target <- wine_testing_target

pred_grnn <- function(grnn, input, target){
  group <- split(input, 1:nrow(input))
  pred <- foreach(i = group, .combine = rbind) %dopar% {
    data.frame(pred = guess(grnn, as.matrix(i)), i, row.names = NULL)
  }
  result = data.frame(actual=target, predict=pred$pred)
}

sigma <- c(0.3, 0.4, 0.5, 0.6, 0.7)
grnn_analysis <- foreach(s = sigma, .combine=rbind) %dopar% {
  start <- Sys.time()
  model <- smooth(learn(grnn_training, variable.column=ncol(grnn_training)), sigma = s)
  test_predict <- pred_grnn(model, input=grnn_test_input, target=grnn_test_target)
  test_predict[is.na(test_predict)] <- 0
  accuracy = mean(test_predict$actual == round(test_predict$predict))
  result = data.frame(s, accuracy, time=Sys.time()-start)
}
grnn_sigma <- grnn_analysis[grnn_analysis$accuracy == max(grnn_analysis$accuracy),1]
grnn_model <- smooth(learn(grnn_training, variable.column=ncol(grnn_training)), sigma = grnn_sigma)
grnn_test_predict <- pred_grnn(grnn_model, input=grnn_test_input, target=grnn_test_target)
grnn_test_predict[is.na(grnn_test_predict)] <- 0
grnn_accuracy <- mean(grnn_test_predict$actual == round(grnn_test_predict$predict))
