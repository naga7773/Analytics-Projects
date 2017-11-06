#the dataframe is downloaded from the website and read in to R using read.table command
pd=read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",sep=",",header=F,col.names=c("age", "type_employer", "fnlwgt", "education","education_num","marital", "occupation", "relationship", "race","sex","capital_gain", "capital_loss", "hr_per_week","country", "income"),fill=FALSE,strip.white=T)
#to get the names of the variables
names(pd)
#to get the dimensions of the data frame
dim(pd)
#finding out the class of a variable using lapply()
lapply(pd,class)
#findout the missing values in a column using table(pd$variablename)
#oserved that there are rows with "?" in type_employer, occupation and country.
#removemissingvalues
pd<-pd[pd$type_employer!="?",]
pd<-pd[pd$occupation!="?",]
pd<-pd[pd$country!="?",]
dim(pd)
#changing the income values <=50K to "0" and >50K to "1"
pd$income=ifelse(pd$income=="<=50K",0,1)
#class probability for "0" is 0.7510775
#class probability for "1" is 0.2489225
#removing the variable education_num and fnlwgt
pd[["education_num"]]=NULL
#removed this variable because it reperesents same data as education variable
#and intial analysis showed that it cluster the analysis. 
pd[["fnlwgt"]]=NULL
summary(pd)
dim(pd)
pd$education=as.character(pd$education)
pd$education = gsub("^10th","<=10th",pd$education)
pd$education = gsub("^11th","highschool",pd$education)
pd$education = gsub("^12th","highschool",pd$education)
pd$education = gsub("^1st-4th","<=10th",pd$education)
pd$education = gsub("^5th-6th","<=10th",pd$education)
pd$education = gsub("^7th-8th","<=10th",pd$education)
pd$education = gsub("^9th","<=10th",pd$education)
pd$education = gsub("^Assoc-acdm","Associates",pd$education)
pd$education = gsub("^Assoc-voc","Associates",pd$education)
pd$education = gsub("^Bachelors","Bachelors",pd$education)
pd$education = gsub("^Doctorate","Doctorate",pd$education)
pd$education = gsub("^HS-Grad","HS-Grad",pd$education)
pd$education = gsub("^Masters","Masters",pd$education)
pd$education = gsub("^Preschool","<=10th",pd$education)
pd$education = gsub("^Prof-school","Prof-School",pd$education)
pd$education = gsub("^Some-college","highschool",pd$education)
pd$education=as.factor(pd$education)
summary(pd$education)
#country is by default taken as factor. 
#to modify its content first it is converted to character type
pd$country=as.character(pd$country)
pd$country[pd$country=="United-States"]="US"
pd$country[pd$country=="Vietnam"]="Vietnam"
pd$country[pd$country=="South"]="South"
pd$country[pd$country=="Puerto-Rico"]="Puerto-Rico"
pd$country[pd$country=="Philippines"]="Philippines"
pd$country[pd$country=="Mexico"]="Mexico"
pd$country[pd$country=="Japan"]="Japan"
pd$country[pd$country=="Jamaica"]="Jamaica"
pd$country[pd$country=="Italy"]="Italy"
pd$country[pd$country=="India"]="India"
pd$country[pd$country=="Guatemala"]="Guatemala"
pd$country[pd$country=="Germany"]="Germany"
pd$country[pd$country=="England"]="England"
pd$country[pd$country=="El-Salvador"]="El-Salvador"
pd$country[pd$country=="Cuba"]="Cuba"
pd$country[pd$country=="Dominican-Republic"]="Dominican-Republic"
pd$country[pd$country=="Columbia"]="Columbia"
pd$country[pd$country=="China"]="China"
pd$country[pd$country=="Canada"]="Canada"
pd$country[pd$country=="Cambodia"]="Other"
pd$country[pd$country=="Ecuador"]="Other"
pd$country[pd$country=="France"]="Other"
pd$country[pd$country=="Greece"]="Other"
pd$country[pd$country=="Haiti"]="Other"
pd$country[pd$country=="Holand-Netherlands"]="Other"
pd$country[pd$country=="Honduras"]="Other"
pd$country[pd$country=="Hong"]="Other"
pd$country[pd$country=="Hungary"]="Other"
pd$country[pd$country=="Iran"]="Other"
pd$country[pd$country=="Ireland"]="Other"
pd$country[pd$country=="Laos"]="Other"
pd$country[pd$country=="Nicaragua"]="Other"
pd$country[pd$country=="Outlying-US(Guam-USVI-etc)"]="Other"
pd$country[pd$country=="Peru"]="Other"
pd$country[pd$country=="Poland"]="Other"
pd$country[pd$country=="Portugal"]="Other"
pd$country[pd$country=="Scotland"]="Other"
pd$country[pd$country=="Taiwan"]="Other"
pd$country[pd$country=="Thailand"]="Other"
pd$country[pd$country=="Trinadad&Tobago"]="Other"
pd$country[pd$country=="Yugoslavia"]="Other"
pd$country=as.factor(pd$country)
#percent of males=67.56846
#percent of females=32.43154
#changing capital gain in to three factors none,low and high.
pd[["capital_gain"]] <- ordered(cut(pd$capital_gain,c(-Inf, 0,median(pd[["capital_gain"]][pd[["capital_gain"]] >0]),Inf)),labels = c("None", "Low", "High"))
pd[["capital_loss"]] <- ordered(cut(pd$capital_loss,c(-Inf, 0,median(pd[["capital_loss"]][pd[["capital_loss"]] >0]),Inf)),labels = c("None", "Low", "High"))
pd$income=as.factor(pd$income)