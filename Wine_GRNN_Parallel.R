### Setup data for winequality
pkgs <- c('doParallel', 'foreach', 'grnn')
lapply(pkgs, require, character.only = T)
registerDoParallel(cores = 8)

set.seed(123)

#wine_data = read.csv("F:\\Mtech ISS\\Unit 2-6 - Computation Intelligence I\\CA\\data\\winequality-white.csv", header=TRUE)
data = read.csv("~/Workspace/ComIntel1-CA/data/winequality-white.csv", header=TRUE)

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
wine_testing <- wine_data[-wine_positions,1:wine_length-1]

##Function to predict and verify prediction of GRNN based on sigma
wine_grnn_accuracy <- function(sigma, data, positions, length, training, testing) {
  result = data[-positions,]
  result$actual = result[,length]
  result$predict = -1
  
  grnn <- learn(training, variable.column=length)
  grnn <- smooth(grnn, sigma = sigma)

  ###GRNN Testing for Wine
  for(i in 1:nrow(testing))
  {
    vec <- as.matrix(testing[i,])
    res <- guess(grnn, vec)

    if(is.nan(res))
    {
      cat("Entry ",i," Generated NaN result!\n")
    }
    else
    {
      result$predict[i] <- res
    }
  }
  
  grnn_result = data.frame(sigma=sigma, size=0, correct=0, accuracy=0)
  grnn_result$size = nrow(result)
  grnn_result$correct = nrow(result[round(result$predict) == result$actual,])
  cat("-------------- Sigma = ", sigma, "----------------------", "\n")
  cat("No of test cases = ", grnn_result$size ,"\n")
  cat("Correct predictions = ", grnn_result$correct ,"\n")
  grnn_result$accuracy = grnn_result$correct / grnn_result$size * 100;
  cat("Accuracy = ", grnn_result$accuracy, "\n")
  
  return(grnn_result)
}

### Using list of sigma to choose the best prediction
sigma_list <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
wine_result <- data.frame(sigma=sigma_list, correct=0, size=0, accuracy=0)
for(i in 1:length(sigma_list)) {
  grnn_result <- wine_grnn_accuracy(sigma_list[i], wine_data, wine_positions, wine_length, wine_training, wine_testing)
  wine_result$sigma[i] = grnn_result$sigma
  wine_result$size[i] = grnn_result$size
  wine_result$correct[i] = grnn_result$correct
  wine_result$accuracy[i] = grnn_result$accuracy
}