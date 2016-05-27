### Setup data for yeastquality
pkgs <- c('doParallel', 'foreach', 'grnn', 'RSNNS', 'nnet')
lapply(pkgs, require, character.only = T)
registerDoParallel(cores=8)

library(grnn)
set.seed(123)

data = read.table(file=file.path("data", "yeast.data"), header=TRUE)

### Pre-processing data
X = normalizeData(data[,-c(1,ncol(data))], type="norm")
Y = data[,ncol(data)]
yeast_data <- data.frame(X, Y)

### Setup data for testing
yeast_size=nrow(yeast_data)
yeast_length=ncol(yeast_data)
yeast_index <- 1:yeast_size

yeast_positions <- sample(yeast_index, trunc(yeast_size * 0.75))

yeast_training <- yeast_data[yeast_positions,]
yeast_training_input <- yeast_training[,-yeast_length]
yeast_training_target <- yeast_training[,yeast_length]
yeast_testing <- yeast_data[-yeast_positions,]
yeast_testing_input <- yeast_testing[,-yeast_length]
yeast_testing_target <- yeast_testing[,yeast_length]

### Predict using nnet function
nnet_model <- nnet(Y ~ ., data=yeast_training, size=10, maxit=1000)
nnet_pred<- predict(nnet_model, yeast_testing, type="class")
table(true=yeast_testing_target, predicted=nnet_pred)
nnet_accuracy = mean(yeast_testing_target == nnet_pred)
nnet_test_predict <- data.frame(actual=encodeClassLabels(decodeClassLabels(yeast_testing_target)),
                                predict=encodeClassLabels(decodeClassLabels(nnet_pred)))

### Training for MLP
mlp_training_input <- yeast_training_input
mlp_training_target <- decodeClassLabels(yeast_training_target)
mlp_test_input <- yeast_testing_input
mlp_test_target <- encodeClassLabels(decodeClassLabels(yeast_testing_target))

mlp_model = mlp(mlp_training_input, mlp_training_target, size=5, learnFunc = "Rprop", maxit=100, linOut=TRUE)
mlp_test_predict <- data.frame(actual=mlp_test_target, predict=encodeClassLabels(predict(mlp_model, mlp_test_input)))
mlp_accuracy <- mean(mlp_test_predict$actual == mlp_test_predict$predict)

### Training and testing for RBF
rbf_training_input <- yeast_training_input
rbf_training_target <- decodeClassLabels(yeast_training_target)
rbf_test_input <- yeast_testing_input
rbf_test_target <- encodeClassLabels(decodeClassLabels(yeast_testing_target))

rbf_model = rbf(rbf_training_input, rbf_training_target, size=40, maxit=1000, 
                initFuncParams=c(0, 1, 0, 0.01, 0.01), 
                learnFuncParams=c(1e-8, 0, 1e-8, 0.1, 0.8), 
                linOut = TRUE)
rbf_test_predict <- data.frame(actual=rbf_test_target, predict=encodeClassLabels(predict(rbf_model, rbf_test_input)))
rbf_accuracy <- mean(rbf_test_predict$actual == rbf_test_predict$predict)
table(rbf_test_predict$actual, round(rbf_test_predict$predict))

### Define a function for prediction given a GRNN
grnn_training_input <- yeast_training_input
grnn_training_target <- encodeClassLabels(decodeClassLabels(yeast_training_target))
grnn_training <- data.frame(grnn_training_input, grnn_training_target)
grnn_test_input <- yeast_testing_input
grnn_test_target <- encodeClassLabels(decodeClassLabels(yeast_testing_target))

pred_grnn <- function(grnn, input, target){
  group <- split(input, 1:nrow(input))
  pred <- foreach(i = group, .combine = rbind) %dopar% {
    data.frame(pred = guess(grnn, as.matrix(i)), i, row.names = NULL)
  }
  result = data.frame(actual=target, predict=pred$pred)
}

sigma <- c(0.03, 0.04, 0.05, 0.06, 0.07, 0.1, 0.15, 0.2, 0.5)
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

### RBF using dynamic decay adjustment (DDA)
rbfDDA_model <- rbfDDA(rbf_training_input, rbf_training_target, maxit = 1, initFunc = "Randomize_Weights",
                       initFuncParams = c(-25, 25), learnFunc = "RBF-DDA",
                       learnFuncParams = c(0.8, 0.4, 10), updateFunc = "Topological_Order",
                       updateFuncParams = c(0), shufflePatterns = TRUE, linOut = FALSE)
rbfDDA_test_predict <- data.frame(actual=rbf_test_target, predict=encodeClassLabels(predict(rbfDDA_model, rbf_test_input)))
rbfDDA_accuracy <- mean(rbfDDA_test_predict$actual == rbfDDA_test_predict$predict)
table(rbfDDA_test_predict$actual, round(rbfDDA_test_predict$predict))
cat(rbfDDA_accuracy)
