### Setup data for winequality
pkgs <- c('doParallel', 'foreach', 'RSNNS')
lapply(pkgs, require, character.only = T)
registerDoParallel(cores=8)

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

# mlp_sizes = c(50)
# mlp_analysis <- foreach(s = mlp_sizes, .combine=rbind) %dopar% {
#   start <- Sys.time()
#   model <- mlp(mlp_training_input, mlp_training_target, size=s, learnFunc = "Rprop", maxit=100, linOut=TRUE)
#   test_predict <- data.frame(actual=mlp_test_target, predict=predict(model, mlp_test_input))
#   accuracy <- mean(test_predict$actual == round(test_predict$predict))
#   result = data.frame(s, accuracy, time=Sys.time()-start)
# }
# mlp_size = mlp_analysis[mlp_analysis$accuracy == max(mlp_analysis$accuracy),1]
mlp_size = 10
mlp_test_target <- mlp_training_target
mlp_test_input <- mlp_training_input
mlp_model = mlp(mlp_training_input, mlp_training_target, size=mlp_size, learnFunc = "Rprop", maxit=100, linOut=TRUE)
mlp_test_predict <- data.frame(actual=mlp_test_target, predict=predict(mlp_model, mlp_test_input))
mlp_test_predict[is.na(mlp_test_predict)] <- 0

mlp_accuracy <- mean(mlp_test_predict$actual == round(mlp_test_predict$predict))
cat(mlp_accuracy)

params = c(0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3)
mlp_analysis <- foreach(p = params, .combine=rbind) %dopar% {
  start <- Sys.time()
  training_input <- mlp_training_input
  test_input <- mlp_test_input
  model <- mlp(training_input, mlp_training_target, size=10, learnFunc = "Std_Backpropagation",
              learnFuncParams=c(p), maxit=100, linOut=TRUE)
  test_predict <- data.frame(actual=mlp_test_target, predict=predict(model, test_input))
  test_predict[is.na(test_predict)] <- 0
  accuracy <- mean(test_predict$actual == round(test_predict$predict))
  result = data.frame(p, accuracy, time=Sys.time()-start)
}

mlp_size = 10
mlp_param = 0.7
mlp_model <- mlp(mlp_training_input, mlp_training_target, size=10, learnFunc = "Std_Backpropagation",
             learnFuncParams=c(mlp_param), maxit=100, linOut=TRUE)
mlp_test_predict <- data.frame(actual=mlp_test_target, predict=predict(mlp_model, mlp_test_input))
mlp_test_predict[is.na(mlp_test_predict)] <- 0
mlp_accuracy <- mean(mlp_test_predict$actual == round(mlp_test_predict$predict))
cat(mlp_accuracy)

