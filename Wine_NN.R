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

### Training for MLP
mlp_training_input <- wine_training_input
mlp_training_target <- wine_training_target
mlp_test_input <- wine_testing_input
mlp_test_target <- wine_testing_target

params = c(1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1)
mlp_analysis <- foreach(p = params, .combine=rbind) %dopar% {
  start <- Sys.time()
  training_input <- mlp_training_input
  test_input <- mlp_test_input
  model <- mlp(training_input, mlp_training_target, size=c(10,10), learnFunc = "Std_Backpropagation",
               learnFuncParams=c(p), maxit=1000, linOut=TRUE)
  test_predict <- data.frame(actual=mlp_test_target, predict=predict(model, test_input))
  accuracy <- mean(test_predict$actual == round(test_predict$predict))
  mse <- mean((test_predict$actual - test_predict$predict)^2)
  result = data.frame(p, accuracy, mse, time=Sys.time()-start)
}

mlp_param <- mlp_analysis[mlp_analysis$mse == min(mlp_analysis$mse),1]
mlp_model <- mlp(mlp_training_input, mlp_training_target, size=c(10,10), learnFunc = "Std_Backpropagation",
                 learnFuncParams=c(mlp_param), maxit=1000, linOut=TRUE)
mlp_test_predict <- data.frame(actual=mlp_test_target, predict=predict(mlp_model, mlp_test_input))
mlp_accuracy <- mean(mlp_test_predict$actual == round(mlp_test_predict$predict))
mlp_mse <- mean((mlp_test_predict$actual - mlp_test_predict$predict)^2)
cat(mlp_mse)

### Training and testing for RBF
rbf_training_input <- wine_training_input
rbf_training_target <- wine_training_target
rbf_test_input <- wine_testing_input
rbf_test_target <- wine_testing_target

rbf_model = rbf(rbf_training_input, rbf_training_target, size=40, maxit=1000, 
                initFuncParams=c(0, 1, 0, 0.01, 0.01), 
                learnFuncParams=c(1e-8, 0, 1e-8, 0.1, 0.8), 
                linOut = TRUE)
rbf_test_predict <- data.frame(actual=rbf_test_target, predict=predict(rbf_model, rbf_test_input))
rbf_accuracy <- mean(rbf_test_predict$actual == round(rbf_test_predict$predict))
rbf_mse <- mean((rbf_test_predict$actual-rbf_test_predict$predict)^2)
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
  test_predict[is.na(test_predict)] <- 6
  accuracy = mean(test_predict$actual == round(test_predict$predict))
  mse <- mean((test_predict$actual-test_predict$predict)^2)
  result = data.frame(s, accuracy, mse, time=Sys.time()-start)
}
# grnn_sigma <- grnn_analysis[grnn_analysis$accuracy == max(grnn_analysis$accuracy),1]
grnn_sigma <- grnn_analysis[grnn_analysis$mse == min(grnn_analysis$mse),1]
grnn_model <- smooth(learn(grnn_training, variable.column=ncol(grnn_training)), sigma = grnn_sigma)
grnn_test_predict <- pred_grnn(grnn_model, input=grnn_test_input, target=grnn_test_target)
grnn_test_predict[is.na(grnn_test_predict)] <- 6
grnn_accuracy <- mean(grnn_test_predict$actual == round(grnn_test_predict$predict))
grnn_mse <- mean((grnn_test_predict$actual-grnn_test_predict$predict)^2)

### Using Hybrid 
hybrid_data = data.frame(mlp_predict=mlp_test_predict$predict, rbf_predict=rbf_test_predict$predict, grnn_predict=grnn_test_predict$predict)
hybrid_matrix = data.matrix(hybrid_data)
names <- c("mlp","rbf","grnn")
hybrid_mse = matrix(nrow = 3, ncol = 3, dimnames = list(names, names))
for (i in 1:3) {
  for (j in 1:3) {
    hybrid_mse[i,j] = mean((wine_testing_target - (hybrid_matrix[,i]+hybrid_matrix[,j])/2)^2)
  }
}
