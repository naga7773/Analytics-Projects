getwd()
rm(list=ls())
setwd("/Users/naga/Downloads/My files/My code files/R/")

OriginalData <- read.csv("winequality-white.csv", sep = ';')

od <- OriginalData[,]

summary(od)

library(ggplot2)
library(dplyr)
library(randomForest)
library(MASS)
library(gridExtra)

od$quality_factor <- as.factor(od$quality)


od <- od %>%
  mutate(rating = ifelse(quality<=5,"bad",ifelse(quality<=7,"average","good")))
od$rating <- factor(od$rating, levels = c("bad","average","good"))


#Split data in to train and test 
split_ratio <-  0.7
n_obs <- dim(od)[1]
train_index <- sample(c(1:n_obs), floor(split_ratio * n_obs), replace = F)
train_data <- od[train_index, ]
test_data <- od[-train_index, ]


#### RandomForest ######
rf.fit <- randomForest(rating~.-quality-quality_factor,data = train_data, mtry = 3, importance = T)
rf.pred <- predict(rf.fit, newdata = test_data)
table(rf.pred, test_data$rating)
rf_acc <- mean(rf.pred == test_data$rating)
rf_acc


### LDA #####
lda.fit <- lda(data = train_data, rating~.-quality-quality_factor)
lda.class <- predict(lda.fit, newdata = test_data)$class
table(lda.class, test_data$rating)
lda_acc <- mean(lda.class == test_data$rating)
lda_acc

### QDA ###
qda.fit <- qda(data = train_data, rating~.-quality-quality_factor)
qda.class <- predict(qda.fit, newdata = test_data)$class
table(qda.class, test_data$rating)
qda_acc <- mean(qda.class == test_data$rating)
qda_acc

### Multinomial Logistic regression ###
require(nnet)
mlr.fit <- multinom(data = train_data, rating~.-quality-quality_factor, family = "binomial")
mlr.pred <- predict(mlr.fit, newdata = test_data, type = "class") #type="probs" gives prob of every class per obs
table(mlr.pred, test_data$rating)
mlr_acc <- mean(mlr.pred == test_data$rating)
mlr_acc

### SVC ###
library(e1071)
#svm_linear_tune<- tune(svm,rating~.-quality-quality_factor, data = od, kernel = "linear",
#                 range=list(cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100)))

#radial kernel has gamma permeter for rbf and degree parameter for polynomial kernal
#svm_radial_tune<- tune(svm,rating~.-quality-quality_factor, data = od, kernel = "radial",
#                       range=list(cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100),
#                                  gamma = c(0.6, 0.7, 0.8, 0.9, 1)))

#svm_linear_model <- svm_linear_tune$best.model
#svm_radial_model <- svm_radial_tune$best.model
svm.fit <- svm(data = train_data, rating~.-quality-quality_factor, kernel = "radial", cost = 10, gamma = 1)
svm.pred <- predict(svm.fit, newdata = test_data)
table(svm.pred, test_data$rating)
svm_acc <- mean(svm.pred == test_data$rating)
svm_acc

### neural networks model ###
library(nnet)
response_matrix <- class.ind(train_data$rating)
nn.fit <- nnet(train_data[,c(1:11)], response_matrix, size =10, softmax = T)
nn.pred <- predict(nn.fit, test_data[,c(1:11)] , type="class")
table(nn.pred, test_data$rating)
nn_acc <- mean(nn.pred == test_data$rating)
nn_acc


importance(rf.fit)


#bad  average     good MeanDecreaseAccuracy MeanDecreaseGini
#fixed.acidity        37.31477 34.20959 22.06420             49.53575         116.8136
#volatile.acidity     67.62890 67.87021 35.77009             87.03264         201.6612
#citric.acid          36.05474 42.17557 23.06347             52.28254         137.4568
#residual.sugar       36.83541 44.47620 17.68827             59.18256         142.0443
#chlorides            40.91082 38.18869 23.07675             50.04954         142.5658
#free.sulfur.dioxide  50.05817 48.52988 28.41387             70.62745         167.4730
#total.sulfur.dioxide 36.20287 39.19218 22.70450             51.64052         145.9411
#density              31.54109 41.94301 27.64068             54.79205         182.0533
#pH                   40.82955 40.25407 24.89518             55.76853         135.8934
#sulphates            33.43826 31.06024 29.62987             47.48371         118.8915
#alcohol              68.41668 48.06367 48.96977             76.86050         222.3791

# As higher accuracy is achieved using random forest algorithm, cross validation is performed using a for loop. 


#### RandomForest ######
previous_rf_acc = 0
for (i in c(1:20)){
  
  split_ratio <-  0.7
  n_obs <- dim(od)[1]
  train_index <- sample(c(1:n_obs), floor(split_ratio * n_obs), replace = F)
  train_data <- od[train_index, ]
  test_data <- od[-train_index, ]
  
  rf.fit <- randomForest(rating~.-quality-quality_factor,data = train_data, mtry = 3, importance = T)
  rf.pred <- predict(rf.fit, newdata = test_data)
  table(rf.pred, test_data$rating)
  rf_acc <- mean(rf.pred == test_data$rating)
  if(rf_acc > previous_rf_acc){
    previous_rf_acc = rf_acc
    bestfit = rf.fit
  }
  
}

#best fit is saved as rf.fit.1
rf.fit.1 <- bestfit
# rf.fit.1 
rf.pred <- predict(rf.fit.1, newdata = test_data)
table(rf.pred, test_data$rating)
rf_acc <- mean(rf.pred == test_data$rating)
rf_acc
