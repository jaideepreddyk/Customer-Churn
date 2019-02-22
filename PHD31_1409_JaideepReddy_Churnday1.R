custid_churn_traindata=read.csv("Train.csv")
head(custid_churn_traindata)
colnames(custid_churn_traindata)

str(custid_churn_traindata)

acc_info_traindata=read.csv("Train_AccountInfo.csv")
str(acc_info_traindata)

demo_traindata=read.csv("Train_Demographics.csv")
str(demo_traindata)

services_traindata=read.csv("train_ServicesOptedFor.csv")
str(services_traindata)

colnames(demo_traindata)[1]="CustomerID"

str(services_traindata)
length(which(services_traindata$CustomerID=="Cust2034"))

colnames(services_traindata)
# View(reshape(services_traindata,idvar = "CustomerID", timevar = "TypeOfService",direction = "wide"))

library(reshape2)

new_services_traindata=dcast(services_traindata, CustomerID ~ TypeOfService)

final_traindata=merge(acc_info_traindata,demo_traindata,by = "CustomerID")

final_traindata=merge(final_traindata,new_services_traindata,by = "CustomerID")

final_traindata=merge(final_traindata,custid_churn_traindata,by = "CustomerID")

str(final_traindata)
summary(final_traindata)

final_traindata[final_traindata=='?']<-NA
final_traindata[final_traindata=='']<-NA
final_traindata[final_traindata=='MISSINGVAL']<-NA
sum(is.na(final_traindata$TotalCharges))

summary(final_traindata)

final_traindata$TotalCharges<-as.numeric(levels(final_traindata$TotalCharges))[final_traindata$TotalCharges]
unique(final_traindata$DeviceProtection)

charvars<-c('Country','State','Education','Gender','DeviceProtection','HasPhoneService','InternetServiceCategory','MultipleLines','OnlineBackup','OnlineSecurity','StreamingMovies','StreamingTelevision','TechnicalSupport','Retired','HasDependents','HasPartner')
final_traindata[charvars]<-lapply(final_traindata[charvars], factor)
str(final_traindata)

sum(is.na(final_traindata))

final_traindata<-na.omit(final_traindata)

summary(final_traindata$Education)

final_traindata$DOE<-as.Date(final_traindata$DOE,format= "%d-%b-%y")

summary(final_traindata)

# Removing unecessary columns
final_traindata$DOE<-NULL
final_traindata$DOC<-NULL
final_traindata$Country<-NULL
final_traindata$State<-NULL
final_traindata$CustomerID<-NULL

####### PLOTS ######################
char_df<-subset(final_traindata,select =c('Education','Gender','DeviceProtection','HasPhoneService','InternetServiceCategory','MultipleLines','OnlineBackup','OnlineSecurity','StreamingMovies','StreamingTelevision','TechnicalSupport','Retired','HasDependents','HasPartner'))

par(mfrow=c(2,2))
for(i in 1:length(char_df)){
  plot(char_df[i],main=names(char_df[i]))
}

par(mfrow=c(1,2))
hist(final_traindata$BaseCharges,main = 'BaseCharges')
hist(final_traindata$TotalCharges,main = 'TotalCharges')

boxplot(final_traindata$BaseCharges)
boxplot(final_traindata$TotalCharges)


################## TEST-TRAIN SPLIT #####################
library(caret)
inTrain <- createDataPartition(y=final_traindata$Churn, p=0.7, list=FALSE) 
mytrain_train<-final_traindata[inTrain,]
mytrain_valid<-final_traindata[-inTrain,]

######################## 1. MODEL C50 #######################
library(C50)
names(mytrain_train[20])

model_c50<-C5.0(x=mytrain_train[,-20],y=mytrain_train[,20],rules = T,trials = 3)

summary(model_c50)

testmodel_c50<-predict(model_c50,mytrain_valid[,-20])
t<-table(mytrain_valid[,20],testmodel_c50)

confusionMatrix(data = testmodel_c50,reference = mytrain_valid[,20],positive = "Yes")

###################### 2.MODEL KNN  ###########################

charvars2<-c('Education','Gender','DeviceProtection','HasPhoneService','InternetServiceCategory','MultipleLines','OnlineBackup','OnlineSecurity','StreamingMovies','StreamingTelevision','TechnicalSupport','Retired','HasDependents','HasPartner','ElectronicBilling','ContractType','PaymentMethod')
library(dummies)

final_traindata[charvars2]<-lapply(final_traindata[charvars2], dummy)

length(charvars2)

inTrain <- createDataPartition(y=final_traindata$Churn, p=0.7, list=FALSE)
mytrain_train<-final_traindata[inTrain,]
mytrain_valid<-final_traindata[-inTrain,]


library(class)
knn_model<-knn(train = mytrain_train[,-20],test = mytrain_valid[,-20],cl = mytrain_train$Churn,k = 5)

confusionMatrix(knn_model,mytrain_valid[,20],positive = "Yes")

######################### 3.Model rpart  ##################################### 

summary(final_traindata)
library(rpart)
library(rpart.plot)

model_rpart<-rpart(Churn~.,data = mytrain_train)
summary(model_rpart)

rpart.plot(model_rpart)
plotcp(model_rpart)

testmodel_rpart<-predict(model_rpart,mytrain_valid[,-20],'class')
confusionMatrix(testmodel_rpart,mytrain_valid[,20],positive = "Yes")

############################## 4.Logistic Regression (BEST MODEL) #############################################
library(MASS)

model_logi<-glm(Churn~.,data = mytrain_train,family = "binomial")

summary(model_logi)
model_logi_stepaic<-stepAIC(model_logi)
summary(model_logi_stepaic)

log_preds<-predict(model_logi_stepaic,mytrain_valid[,-20],type = "response")
log_preds
length(log_preds)

log_preds<-ifelse(log_preds<0.5, 'No', 'Yes')

confusionMatrix(log_preds,mytrain_valid[,20],positive = "Yes")

###  ROC  ###
library(ROCR)
ROCRpred = prediction(log_preds,mytrain_valid[,20])
# as.numeric(performance(ROCRpred, "auc")@y.values)
ROCRperf <- performance(ROCRpred, "tpr", "fpr")
par(mfrow=c(1,1))
plot(ROCRperf, colorize = TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))

log_preds<-ifelse(log_preds<0.27, 'No', 'Yes')

confusionMatrix(log_preds,mytrain_valid[,20],positive = "Yes")

######################### 5.XGBoost ##############################

xgb.ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 3, search='random', allowParallel=T)

xgb.tune <-train(Churn~., data = mytrain_train, method="xgbTree", 
                 trControl=xgb.ctrl,
                 tuneLength=20, verbose=T, metric="Accuracy", nthread=3)

xg_preds<-predict(xgb.tune,mytrain_valid[,-20])

confusionMatrix(xg_preds,mytrain_valid[,20],positive = "Yes")

############################## 6.SVM #####################################
library(e1071)
charvars2<-c('Education','Gender','DeviceProtection','HasPhoneService','InternetServiceCategory','MultipleLines','OnlineBackup','OnlineSecurity','StreamingMovies','StreamingTelevision','TechnicalSupport','Retired','HasDependents','HasPartner','ElectronicBilling','ContractType','PaymentMethod')
library(dummies)

final_traindata[charvars2]<-lapply(final_traindata[charvars2], dummy)

length(charvars2)

inTrain <- createDataPartition(y=final_traindata$Churn, p=0.7, list=FALSE)
mytrain_train<-final_traindata[inTrain,]
mytrain_valid<-final_traindata[-inTrain,]

model_svm<-svm(Churn~.,data = mytrain_train,kernel="linear")

preds_svm<-predict(model_svm,mytrain_valid[,-20])

confusionMatrix(preds_svm,mytrain_valid[,20],positive = "Yes")

########### TEST DATA PREPROCESSING AND PREDICTIONS ################

custid_churn_testdata=read.csv("Test.csv")
head(custid_churn_testdata)
colnames(custid_churn_traindata)

str(custid_churn_traindata)

acc_info_testdata=read.csv("Test_AccountInfo.csv")
str(acc_info_traindata)

demo_testdata=read.csv("Test_Demographics.csv")
str(demo_testdata)

services_testdata=read.csv("test_ServicesOptedFor.csv")
str(services_testdata)

colnames(demo_testdata)[1]="CustomerID"

str(services_testdata)
# length(which(services_traindata$CustomerID=="Cust2034"))

colnames(services_testdata)
# View(reshape(services_testdata,idvar = "CustomerID", timevar = "TypeOfService",direction = "wide"))

library(reshape2)

new_services_testdata=dcast(services_testdata, CustomerID ~ TypeOfService)

final_testdata=merge(acc_info_testdata,demo_testdata,by = "CustomerID")

final_testdata=merge(final_testdata,new_services_testdata,by = "CustomerID")

final_testdata=merge(final_testdata,custid_churn_testdata,by = "CustomerID")

str(final_testdata)
summary(final_testdata)

final_testdata[final_testdata=='?']<-NA
final_testdata[final_testdata=='']<-NA
final_testdata[final_testdata=='MISSINGVAL']<-NA

# final_testdata<-na.omit(final_testdata)
str(final_testdata)

final_testdata$TotalCharges<-as.numeric(levels(final_testdata$TotalCharges))[final_testdata$TotalCharges]

# which(final_testdata$CustomerID=="Cust1450")
# final_testdata[250,]

sum(is.na(final_testdata))


str(final_testdata)

unique(final_testdata$DeviceProtection)

charvars<-c('Country','State','Education','Gender','DeviceProtection','HasPhoneService','InternetServiceCategory','MultipleLines','OnlineBackup','OnlineSecurity','StreamingMovies','StreamingTelevision','TechnicalSupport','Retired','HasDependents','HasPartner')
final_testdata[charvars]<-lapply(final_testdata[charvars], factor)
str(final_testdata)

sum(is.na(final_testdata))

summary(final_testdata)
#### Central Imputing #############
library(DMwR)
final_testdata<-centralImputation(final_testdata)

final_testdata$DOE<-as.Date(final_testdata$DOE,format= "%d-%b-%y")


summary(final_testdata)
str(final_testdata)
sum(is.na(final_testdata))

#final_traindata$Churn<-NULL

# Removing unecessary columns
final_testdata$DOE<-NULL
final_testdata$Country<-NULL
final_testdata$State<-NULL
final_testdata$DOC<-NULL
final_testdata$CustomerID<-NULL

str(final_traindata)
str(final_testdata)
############### Taking care of level mismatch

levels(final_testdata$PaymentMethod)<-levels(final_traindata$PaymentMethod)

test_preds<-predict(model_logi,final_testdata,type = "response")

final_predictions<-ifelse(test_preds<0.27, 'No', 'Yes')
## The threshold is obtained from ROC curve by applying logistic model on train

Predictions<-data.frame(CustomerID=custid_churn_testdata$CustomerID,Churn=final_predictions)
write.csv(Predictions,'prediction.csv',row.names = F)

################################## TEST DATA PRE_PROCESSING ENDS HERE #######################

######### Association Rule Mining #######################################
library("arules")
##install.packages("arulesViz")
library("arulesViz")

############## Pre processing for Association Rules ############################
services_traindata$SeviceDetails[which(services_traindata$SeviceDetails=="No internet service")]="No"

services_traindata$SeviceDetails[which(services_traindata$SeviceDetails=="No phone service")]="No"

services_traindata$SeviceDetails[which(services_traindata$SeviceDetails==0)]="No"

services_traindata$SeviceDetails[which(services_traindata$SeviceDetails==1)]="Yes"

services_traindata$TypeOfService<-as.character(services_traindata$TypeOfService)
services_traindata$TypeOfService[which(services_traindata$SeviceDetails=="DSL")]="DSL"
services_traindata$TypeOfService[which(services_traindata$SeviceDetails=="Fiber optic")]="Fiber optic"

services_traindata<-services_traindata[!(services_traindata$SeviceDetails=="No"),]

services_traindata$TypeOfService<-as.factor(services_traindata$TypeOfService)
services_traindata$SeviceDetails<-NULL


str(services_traindata$CustomerID)

colnames(services_traindata)
colnames(custid_churn_traindata)

colnames(services_traindata)<-c('CustomerID','Transaction')

colnames(custid_churn_traindata)<-c('CustomerID','Transaction')
rules_df<-rbind(services_traindata,custid_churn_traindata)


finalrulesdata<-final_traindata

########### Binning Numerical Attributes ############

finalrulesdata$BaseCharges<-cut(finalrulesdata$BaseCharges,seq(0,600,100))
str(finalrulesdata$BaseCharges)
basecharges_df<-data.frame(CustomerID=finalrulesdata$CustomerID,Transaction=finalrulesdata$BaseCharges)
rules_df<-rbind(rules_df,basecharges_df)


finalrulesdata$TotalCharges<-cut(finalrulesdata$TotalCharges,seq(0,50000,10000))
str(finalrulesdata$TotalCharges)
totalcharges_df<-data.frame(CustomerID=finalrulesdata$CustomerID,Transaction=finalrulesdata$TotalCharges)
rules_df<-rbind(rules_df,totalcharges_df)

################# Categorical   Attributes ###################

## Electronic BIlling

elecbilling_df<-data.frame(CustomerID=finalrulesdata$CustomerID,Transaction=finalrulesdata$ElectronicBilling)

elecbilling_df<-elecbilling_df[!(elecbilling_df$Transaction=="No"),]
elecbilling_df$Transaction="ElectronicBilling"

rules_df<-rbind(rules_df,elecbilling_df)
finalrulesdata$ElectronicBilling<-NULL

## Contract Type
contractype_df<-data.frame(CustomerID=finalrulesdata$CustomerID,Transaction=finalrulesdata$ContractType)

rules_df<-rbind(rules_df,contractype_df)

finalrulesdata$ContractType<-NULL

## Payment Method

paymentmethod_df<-data.frame(CustomerID=finalrulesdata$CustomerID,Transaction=finalrulesdata$PaymentMethod)

rules_df<-rbind(rules_df,paymentmethod_df)
finalrulesdata$PaymentMethod<-NULL

## Retired

retired_df<-data.frame(CustomerID=finalrulesdata$CustomerID,Transaction=finalrulesdata$Retired)


retired_df<-retired_df[!(retired_df$Transaction==0),]
retired_df$Transaction="Retired"
rules_df<-rbind(rules_df,paymentmethod_df)
finalrulesdata$Retired<-NULL

## Has Partner
partner_df<-data.frame(CustomerID=finalrulesdata$CustomerID,Transaction=finalrulesdata$HasPartner)

partner_df<-partner_df[!(partner_df$Transaction==2),]
partner_df$Transaction<-"HasPartner"
rules_df<-rbind(rules_df,partner_df)
finalrulesdata$HasPartner<-NULL

## Has Dependent
dependent_df<-data.frame(CustomerID=finalrulesdata$CustomerID,Transaction=finalrulesdata$HasDependents)

dependent_df<-dependent_df[!(dependent_df$Transaction==2),]
dependent_df$Transaction<-"HasDepedents"
rules_df<-rbind(rules_df,dependent_df)
finalrulesdata$HasDependents<-NULL

###  Education 
education_df<-data.frame(CustomerID=finalrulesdata$CustomerID,Transaction=finalrulesdata$Education)

rules_df<-rbind(rules_df,education_df)
finalrulesdata$Education<-NULL

### gender

gender_df<-data.frame(CustomerID=finalrulesdata$CustomerID,Transaction=finalrulesdata$Gender)
rules_df<-rbind(rules_df,gender_df)
finalrulesdata$Gender<-NULL
########  Reading in Transaction Form ######################

write.csv(rules_df,'Transactions.csv',row.names = F)

transdata = read.transactions("Transactions.csv", format = "single", sep = ",", cols = c("CustomerID", "Transaction"))

inspect(transdata[5])


###################### RULE MINING  ###################################

############# 'NO' Rules #################


norules1<-apriori(transdata,parameter = list(supp = 0.001, conf = 0.7, target = "rules",minlen=2),appearance = list(rhs=c("No"),default='lhs'))
top.lift <- sort(norules1, decreasing = TRUE, na.last = NA, by = "lift")
inspect(head(top.lift,10))





############## 'YES' Rules #################
yesrules1<-apriori(transdata,parameter = list(supp = 0.1, conf = 0.6, target = "rules",minlen=2),appearance = list(rhs=c("Yes"),default='lhs'))

top.lift <- sort(yesrules1, decreasing = TRUE, na.last = NA, by = "lift")

inspect(head(top.lift,10))



yesrules2<-apriori(transdata,parameter = list(supp = 0.02, conf = 0.80, target = "rules",minlen=2,maxlen=20),appearance = list(rhs=c("Yes"),default='lhs'))

top.lift <- sort(yesrules2, decreasing = TRUE, na.last = NA, by = "lift")

inspect(head(top.lift,10))


yesrules3<-apriori(transdata,parameter = list(supp = 0.01, conf = 0.86, target = "rules",minlen=2,maxlen=20),appearance = list(rhs=c("Yes"),default='lhs'))
top.lift <- sort(yesrules3, decreasing = TRUE, na.last = NA, by = "lift")

inspect(top.lift)


yesrules4<-apriori(transdata,parameter = list(supp = 0.005, conf = 0.90, target = "rules",minlen=2,maxlen=20),appearance = list(rhs=c("Yes"),default='lhs'))
top.lift <- sort(yesrules4, decreasing = TRUE, na.last = NA, by = "lift")

inspect(head(top.lift,10))

#############################################################################



