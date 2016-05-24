### Setup data for winequality
pkgs <- c('doParallel', 'foreach', 'grnn')
lapply(pkgs, require, character.only = T)
registerDoParallel(cores=8)

library(grnn)
set.seed(123)

#wine_data = read.csv("F:\\Mtech ISS\\Unit 2-6 - Computation Intelligence I\\CA\\data\\winequality-white.csv", header=TRUE)
data = read.csv(file=file.path("data", "winequality-white.csv"), header=TRUE)

### Pre-processing data
X = data[-12]
scaleX = scale(X)
Y = data[12]
wine_data <- data.frame(scaleX, Y)

### Setup data for testing
wine_size=nrow(wine_data)
wine_length=ncol(wine_data)
wine_index <- 1:wine_size

wine_positions <- sample(wine_index, trunc(wine_size * 0.75))

wine_training <- wine_data[wine_positions,]
wine_testing <- wine_data[-wine_positions,]

### Define a function for prediction given a GRNN
pred_grnn <- function(test, grnn){
  group <- split(test, 1:nrow(test))
  pred <- foreach(i = group, .combine = rbind) %dopar% {
    data.frame(pred = guess(grnn, as.matrix(i)), i, row.names = NULL)
  }
}

# Search for optimal Sigma and output result as a list
time_start <- Sys.time()
cv <- foreach(s = c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), .combine = rbind) %dopar% {
  grnn <- smooth(learn(wine_training, variable.column = ncol(wine_training)), sigma = s)
  pred <- pred_grnn(wine_testing[, -ncol(wine_testing)], grnn)
  test_result = data.frame(actual=wine_testing[, ncol(wine_testing)],predict=pred$pred)
  correct = nrow(test_result[round(test_result$predict) == test_result$actual,])
  size = nrow(test_result)
  accuracy = correct / size * 100
  sse <- sum((wine_testing[, ncol(wine_testing)] - pred$pred)^2) 
  data.frame(s, correct, size, accuracy, sse)
}
time_end <- Sys.time()
time_taken <- time_end - time_start

cat("\n### SSE FROM VALIDATIONS ###\n")
print(cv)
jpeg('grnn_cv.jpeg', width = 800, height = 400, quality = 100)
with(cv, plot(s, sse, type = 'b'))
#  
# cat("\n### BEST SIGMA WITH THE LOWEST SSE ###\n")
# print(best.s <- cv[cv$sse == min(cv$sse), 1])
 
# # SCORE THE WHOLE DATASET WITH GRNN
# final_grnn <- smooth(learn(wine_training, variable.column = ncol(wine_training)), sigma = best.s)
# pred_all <- pred_grnn(win[, -ncol(wine_testing)], final_grnn)
# jpeg('grnn_fit.jpeg', width = 800, height = 400, quality = 100)
# plot(pred_all$pred, boston$medv) 
# dev.off()