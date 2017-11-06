pdtest=read.csv(file.choose(),sep=",",header=F,col.names=c("age", "type_employer", "fnlwgt", "education","education_num","marital", "occupation", "relationship", "race","sex","capital_gain", "capital_loss", "hr_per_week","country", "income"),fill=FALSE,strip.white=T)
names(pdtest)
dim(pdtest)
lapply(pdtest,class)
pdtest<-pdtest[pdtest$type_employer!="?",]
pdtest<-pdtest[pdtest$occupation!="?",]
pdtest<-pdtest[pdtest$country!="?",]
dim(pdtest)
pdtest$income=ifelse(pdtest$income=="<=50K.",0,1)
pdtest[["education_num"]]=NULL
pdtest[["fnlwgt"]]=NULL
summary(pdtest)
dim(pdtest)
pdtest$education=as.character(pdtest$education)
pdtest$education = gsub("^10th","<=10th",pdtest$education)
pdtest$education = gsub("^11th","highschool",pdtest$education)
pdtest$education = gsub("^12th","highschool",pdtest$education)
pdtest$education = gsub("^1st-4th","<=10th",pdtest$education)
pdtest$education = gsub("^5th-6th","<=10th",pdtest$education)
pdtest$education = gsub("^7th-8th","<=10th",pdtest$education)
pdtest$education = gsub("^9th","<=10th",pdtest$education)
pdtest$education = gsub("^Assoc-acdm","Associates",pdtest$education)
pdtest$education = gsub("^Assoc-voc","Associates",pdtest$education)
pdtest$education = gsub("^Bachelors","Bachelors",pdtest$education)
pdtest$education = gsub("^Doctorate","Doctorate",pdtest$education)
pdtest$education = gsub("^HS-Grad","HS-Grad",pdtest$education)
pdtest$education = gsub("^Masters","Masters",pdtest$education)
pdtest$education = gsub("^Preschool","<=10th",pdtest$education)
pdtest$education = gsub("^Prof-school","Prof-School",pdtest$education)
pdtest$education = gsub("^Some-college","highschool",pdtest$education)
pdtest$education=as.factor(pdtest$education)
summary(pdtest$education)
#country is by default taken as factor. 
#to modify its content first it is converted to character type
pdtest$country=as.character(pdtest$country)
pdtest$country[pdtest$country=="United-States"]="US"
pdtest$country[pdtest$country=="Vietnam"]="Vietnam"
pdtest$country[pdtest$country=="South"]="South"
pdtest$country[pdtest$country=="Puerto-Rico"]="Puerto-Rico"
pdtest$country[pdtest$country=="Philippines"]="Philippines"
pdtest$country[pdtest$country=="Mexico"]="Mexico"
pdtest$country[pdtest$country=="Japan"]="Japan"
pdtest$country[pdtest$country=="Jamaica"]="Jamaica"
pdtest$country[pdtest$country=="Italy"]="Italy"
pdtest$country[pdtest$country=="India"]="India"
pdtest$country[pdtest$country=="Guatemala"]="Guatemala"
pdtest$country[pdtest$country=="Germany"]="Germany"
pdtest$country[pdtest$country=="England"]="England"
pdtest$country[pdtest$country=="El-Salvador"]="El-Salvador"
pdtest$country[pdtest$country=="Cuba"]="Cuba"
pdtest$country[pdtest$country=="Dominican-Republic"]="Dominican-Republic"
pdtest$country[pdtest$country=="Columbia"]="Columbia"
pdtest$country[pdtest$country=="China"]="China"
pdtest$country[pdtest$country=="Canada"]="Canada"
pdtest$country[pdtest$country=="Cambodia"]="Other"
pdtest$country[pdtest$country=="Ecuador"]="Other"
pdtest$country[pdtest$country=="France"]="Other"
pdtest$country[pdtest$country=="Greece"]="Other"
pdtest$country[pdtest$country=="Haiti"]="Other"
pdtest$country[pdtest$country=="Holand-Netherlands"]="Other"
pdtest$country[pdtest$country=="Honduras"]="Other"
pdtest$country[pdtest$country=="Hong"]="Other"
pdtest$country[pdtest$country=="Hungary"]="Other"
pdtest$country[pdtest$country=="Iran"]="Other"
pdtest$country[pdtest$country=="Ireland"]="Other"
pdtest$country[pdtest$country=="Laos"]="Other"
pdtest$country[pdtest$country=="Nicaragua"]="Other"
pdtest$country[pdtest$country=="Outlying-US(Guam-USVI-etc)"]="Other"
pdtest$country[pdtest$country=="Peru"]="Other"
pdtest$country[pdtest$country=="Poland"]="Other"
pdtest$country[pdtest$country=="Portugal"]="Other"
pdtest$country[pdtest$country=="Scotland"]="Other"
pdtest$country[pdtest$country=="Taiwan"]="Other"
pdtest$country[pdtest$country=="Thailand"]="Other"
pdtest$country[pdtest$country=="Trinadad&Tobago"]="Other"
pdtest$country[pdtest$country=="Yugoslavia"]="Other"
pdtest$country=as.factor(pdtest$country)
pdtest[["capital_gain"]] <- ordered(cut(pdtest$capital_gain,c(-Inf, 0,median(pdtest[["capital_gain"]][pdtest[["capital_gain"]] >0]),Inf)),labels = c("None", "Low", "High"))
pdtest[["capital_loss"]] <- ordered(cut(pdtest$capital_loss,c(-Inf, 0,median(pdtest[["capital_loss"]][pdtest[["capital_loss"]] >0]),Inf)),labels = c("None", "Low", "High"))
pdtest$income=as.factor(pdtest$income)

