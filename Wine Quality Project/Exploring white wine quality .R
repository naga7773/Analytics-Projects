# Exploring wine qulaity data set 

rm(list=ls())
setwd("/Users/naga/Downloads/My files/My code files/R/")

OriginalData <- read.csv("winequality-white.csv", sep = ';')

od <- OriginalData[,]

od$quality_factor <- as.factor(od$quality)

od <- od %>%
  mutate(rating = ifelse(quality<=5,"bad",ifelse(quality<=7,"average","good")))
od$rating <- factor(od$rating, levels = c("bad","average","good"))


#analyzing sub data in each category

od_9 <- od%>%
  filter(quality==9)

od_8 <- od%>%
  filter(quality==8)

od_7 <- od%>%
  filter(quality==7)

od_6 <- od%>%
  filter(quality==6)


p9 <- ggplot(data = od_9)+
  geom_density(aes(x = fixed.acidity, fill = quality_factor))

p8 <- ggplot(data = od_8)+
  geom_density(aes(x = fixed.acidity, fill = quality_factor, alpha = 0.5))

p7 <- ggplot(data = od_7)+
  geom_density(aes(x = fixed.acidity, fill = quality_factor, alpha = 0.5))

p6 <- ggplot(data = od_6)+
  geom_density(aes(x = fixed.acidity, fill = quality_factor, alpha = 0.5))

grid.arrange(p6,p7,p8,p9,ncol = 2)


od %>%
  dplyr::select(quality_factor,fixed.acidity)%>%
  group_by(quality_factor)%>%
  summarise(avg = mean(fixed.acidity), count = n())

od %>%
  dplyr::select(quality_factor,volatile.acidity)%>%
  group_by(quality_factor)%>%
  summarise(avg = mean(volatile.acidity), count = n())

ggplot(data=od)+
  geom_boxplot(aes(x = quality_factor,y = fixed.acidity,fill = quality_factor))

ggplot(data=od)+
  geom_boxplot(aes(x = quality_factor,y = volatile.acidity,fill = quality_factor))

ggplot(data=od)+
  geom_boxplot(aes(x = quality_factor,y = citric.acid,fill = quality_factor))

ggplot(data=od)+
  geom_boxplot(aes(x = quality_factor,y = residual.sugar,fill = quality_factor))


ggplot(data =od)+
  geom_point(aes(x = residual.sugar, y = fixed.acidity, col = quality_factor))

rs_lt30 <- od%>%
  filter(residual.sugar<=30)

ggplot(data =rs_lt30)+
  geom_point(aes(x = residual.sugar, y = alcohol, col = quality_factor))


#Splitting data in to train and test

split_ratio <-  0.7
n_obs <- dim(od)[1]
train_index <- sample(c(1:n_obs), floor(split_ratio * n_obs), replace = F)
train_data <- od[train_index, ]
test_data <- od[-train_index, ]

#Analyzing the rating variable 
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










