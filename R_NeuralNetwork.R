#-------------------
install.packages('grnn')


library(grnn)

set.seed(123)

### Setup data for yeast
#yeast_data = read.csv("F:\\Mtech ISS\\Unit 2-6 - Computation Intelligence I\\CA\\data\\yeast.data", header=FALSE)
#yeast_data = read.csv("/Volumes/HIEU/Mtech ISS/Unit 2-6 - Computation Intelligence I/CA/data/yeast.data", header=FALSE)
yeast_data = read.csv("~/Workspace/ComIntel1-CA/data/yeast.data", header=FALSE)

yeast_size=nrow(yeast_data)
yeast_length=ncol(yeast_data)
yeast_index <- 1:yeast_size

yeast_positions <- sample(yeast_index, trunc(yeast_size * 0.75))
yeast_training <- yeast_data[yeast_positions,]
yeast_testing <- yeast_data[-yeast_positions,1:yeast_length-1]

### Setup data for winequality
install.packages('grnn')
library(grnn)
set.seed(123)

#wine_data = read.csv("F:\\Mtech ISS\\Unit 2-6 - Computation Intelligence I\\CA\\data\\winequality-white.csv", header=TRUE)
wine_data = read.csv("~/Workspace/ComIntel1-CA/data/winequality-white.csv", header=TRUE)

wine_size=nrow(wine_data)
wine_length=ncol(wine_data)
wine_index <- 1:wine_size

wine_positions <- sample(wine_index, trunc(wine_size * 0.75))

wine_training <- wine_data[wine_positions,]
wine_testing <- wine_data[-wine_positions,1:wine_length-1]

##GRNN Training and classification for Wine
wine_result = wine_data[-wine_positions,]
wine_result$actual = wine_result[,wine_length]
wine_result$predict = -1

wine_grnn <- learn(wine_training, variable.column=wine_length)
wine_grnn <- smooth(wine_grnn, sigma = 0.9)

###GRNN Testing for Wine
for(i in 1:nrow(wine_testing))
{  
  vec <- as.matrix(wine_testing[i,])
  res <- guess(wine_grnn, vec)
  
  if(is.nan(res))
  {
    cat("Entry ",i," Generated NaN result!\n")
  }
  else
  {
    wine_result$predict[i] <- res
  }
}

wine_result_size = nrow(wine_result)
wine_result_correct = nrow(wine_result[round(wine_result$predict) == wine_result$actual,])
cat("No of test cases = ",wine_result_size,"\n")
cat("Correct predictions = ", wine_result_correct ,"\n")
cat("Accuracy = ", wine_result_correct / wine_result_size * 100 ,"\n")
