# Exploring wine qulaity data set 

rm(list=ls())
setwd("/Users/naga/Downloads/My files/My code files/R/")

OriginalData <- read.csv("winequality-white.csv", sep = ';')

od <- OriginalData[,]

od$quality_factor <- as.factor(od$quality)

od <- od %>%
  mutate(rating = ifelse(quality<=5,"bad",ifelse(quality<=7,"average","good")))
od$rating <- factor(od$rating, levels = c("bad","average","good"))










split_ratio <-  0.7
n_obs <- dim(od)[1]
train_index <- sample(c(1:n_obs), floor(split_ratio * n_obs), replace = F)
train_data <- od[train_index, ]
test_data <- od[-train_index, ]

train_data %>%
  group_by(rating) %>%
  summarise(avg = mean(fixed.acidity),
            dev = sd(fixed.acidity),
            obs = n())



train_data %>%
  group_by(rating) %>%
  summarise(avg = mean(fixed.acidity),
            dev = sd(fixed.acidity),
            obs = n())


t(fixed.acidity)



train_data %>%
  dplyr::select(volatile.acidity,rating)%>%
  ggplot()+
  geom_density(aes(x = volatile.acidity, fill = rating, alpha = 0.3))

train_data %>%
  dplyr::select(fixed.acidity,rating)%>%
  ggplot()+
  geom_density(aes(x = fixed.acidity, fill = rating, alpha = 0.3))

train_data %>%
  dplyr::select(density,rating)%>%
  ggplot()+
  geom_density(aes(x = density, fill = rating, alpha = 0.3))










