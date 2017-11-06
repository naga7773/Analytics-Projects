rm(list = ls())
setwd("/Users/naga/Downloads/My files/ML Data Sets")


df1_train <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header = F, na.strings = ' ?',sep = ',')

str(df1_train)

names(df1_train) <- c("age","workclass ","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","Income")

df1_train$age <- as.numeric(df1_train$age)

ConvertToContinous <- function(x){
  x <- as.numeric(x)
}

df2_train <- df1_train[,]

df2_train$age <- ConvertToContinous(df2_train$age)
df2_train$fnlwgt <- ConvertToContinous(df2_train$fnlwgt)
df2_train$'education-num' <- ConvertToContinous(df2_train$'education-num')
df2_train$`capital-gain` <- ConvertToContinous(df2_train$`capital-gain`)
df2_train$`capital-loss` <- ConvertToContinous(df2_train$`capital-loss`)
df2_train$Income <- ifelse(df2_train$Income == '>50K',1,0)

str(df2_train)


df2_train$`education-num` <-as.ordered(df2_train$`education-num`)


str(df2_train$`education-num`)


summary(df2_train)

names(df2_train)

names(df2_train)[5] <- "education_num"

names(df2_train)

names(df2_train)[6] <- "marital_status"
names(df2_train)[11] <- "capital_gain"
names(df2_train)[12] <- "capital_loss"
names(df2_train)[13] <- "hours_per_week"
names(df2_train)[14] <- "native_country"


summary(df2_train)



attach(df2_train)

hist(age,col = "cyan",main =  "Distribution of Age", xlab = "Age")
BackupDataSet <- df2_train[,]

write.csv(df2_train,row.names = FALSE,"/users/naga/Downloads/My files/ML Data Sets/Adult.csv")

df2_train$income <- as.factor(df2_train$Income)

df2_train <- df2_train[,-15]
str(df2_train)
