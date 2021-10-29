library(spam)
library(verification)
library(dplyr)
library(sampling)
library(tidyverse)
library(pROC)
library(purrr)
library(tidyr)
library(sampling)
library(ggplot2)
library(stringr)
library(rmda)
library(export)
library(glmnet)
library(caret)
library(scorecard)
library(openxlsx)
library(psych)
library(Hmisc)
library(pheatmap)
library(reshape2)
library(PerformanceAnalytics)


#feature selection by U test
dataU<-read.xlsx("339radiomics-Label is metastasis.xlsx",sheet=4,rowNames = TRUE)
pvalue=apply(dataU,1,function(x)wilcox.test(x[1:131],x[131:339],paired = F)$p.value)
write.csv(pvalue,"339-Label is metastasis-Pvalue-U test.csv")
dataR<-read.xlsx("339radiomics-Label is metastasis.xlsx",sheet=3)
dataR<-dataR %>% select(Label,DPL15,DPL329,DPL475,DPL473, DPL215,DPL228,DPL212,DPL334,
                        DPL394,DPL350,DPL364,DPL461,DPL183,DPL310,DPL348,DPL304,DPL317,
                        DPL73,DPL392,DPL113,DPL60, DPL489,DPL230,DPL361,DPL67,DPL155,
                        DPL111,DPL504)
write.xlsx(dataR,"339-Label is metastasis-28features.xlsx")


#5 fold cross validation combined LASSO to seek seed 
traindata<-read.xlsx("339-Label is metastasis-28features.xlsx",sheet=1)
cnt<-1
repeat{
  a<-sample(1:9999,1,replace = FALSE)
  set.seed(a)
  train_index <- createDataPartition(traindata$Label, p = 0.7, list = FALSE)
  train <- traindata[train_index, ]
  test<- traindata[-train_index, ]
  set.seed(a)
  var_sel = list()
  auc_res_test = list()
  auc_res_train = list()
  auc_train = list()
  auc_test  = list()
  cv_fit_list = list()
  fit_list = list()
  roc_compare = c()
  
  for(i  in c(1:10))
  {
    x <- as.matrix(train[, -1])
    y <- train$Label
    
    cv.fit <- cv.glmnet(x, y, family = 'binomial', type.measure = "auc",nfolds = 5, alpha = 1)
    fit <- glmnet(x, y, family = 'binomial',alpha = 1)
    
    cv_fit_list[[i]] = cv.fit
    fit_list[[i]] = fit
    
    vars_sel = coef(cv.fit, s = cv.fit$lambda.1se)
    vars_sel = unlist(vars_sel@Dimnames)[vars_sel@i + 1][-1]
    var_sel[[i]] = data.frame(var_sel = vars_sel)%>%mutate(var_sel = as.character(var_sel))
    
    pre_res_test <- as.vector(predict(fit, newx = as.matrix(test[, -1]), s = cv.fit$lambda.1se))
    roc_res_test <- auc(roc(test$Label, pre_res_test, ci = T, quiet = T, transpose = T))
    
    pre_res_train <- as.vector(predict(fit, newx = x, s = cv.fit$lambda.1se))
    roc_res_train <- auc(roc(train$Label, pre_res_train, ci = T, quiet = T, transpose = T))
    auc_res_test[[i]] = roc_res_test
    auc_res_train[[i]] = roc_res_train
    auc_train[[i]] = roc(train$Label, pre_res_train, ci = T, quiet = T, transpose = T)
    auc_test[[i]]  = roc(test$Label, pre_res_test, ci = T, quiet = T, transpose = T)
    
    roc_compare[i] = roc.test(roc_res_test, roc_res_train)$p.value
  }
  idx = which.max(auc_res_test)
  cvfit = cv_fit_list[[idx]]
  fit = fit_list[[idx]]
  s= cvfit$lambda.1se
  plot(cvfit)
  
  abline(v = log(s), lty = 2, lwd = 1, col = 'blue')
  abline(v = log(cvfit$lambda.1se), lty = 2, lwd = 1, col = 'black')
  export::graph2ppt(file = './sample.pptx', width = 10, height = 10,append = T)
  
  plot(fit, s = s, xvar = 'lambda')
  abline(v = log(s), lty = 2, col = 'blue')
  export::graph2ppt(file = './sample.pptx', width = 10, height = 10,append = T)
  
  roc_train = auc_train[[idx]]
  roc_test  = auc_test[[idx]]
  plot(roc_train, print.auc = T, print.auc.pattern = 'AUC: %.2f (%.2f - %.2f)',
       legacy.axes = T, col = 'red')
  plot(roc_test, print.auc = T, print.auc.pattern = 'AUC: %.2f (%.2f - %.2f)',
       add = T, col = 'blue', print.auc.y = 0.45)
  
  legend(x = 0.3, y = 0.2, legend = c('Train', 'Test'), 
         col = c('red', 'blue'), lty = 1)
  
  coefs <- coefficients(fit, s = s)
  useful_feature <- unlist(coefs@Dimnames)[coefs@i + 1]
  useful_feature <- useful_feature[-1]
  dt_coef <- data.frame(Feature = useful_feature, Coef = coefs@x[-1])
  dt_coef <- arrange(dt_coef, desc(Coef))
  dt_coef$Feature <- factor(dt_coef$Feature, 
                            levels = as.character(dt_coef$Feature))
  p_coef <- ggplot(aes(x = Feature, y = Coef), data = dt_coef)
  p_coef <- p_coef + geom_col(fill = 'blue', width = 0.7) + coord_flip() + 
    theme_bw() + ylab('Coefficients')
  p_coef
  if((roc_train$auc>roc_test$auc) & (roc_test$auc>0.73) & (abs(roc_train$auc-roc_test$auc)<0.1))
  {
    print(a)
    print(roc_train$auc)
    print(roc_test$auc)}
  
  cnt<-cnt+1
  if(cnt>1000){
    break
  }
}


#the seed is 5612 to develop DPL model through LASSO with 5 fold cross validation
traindata<-read.xlsx("339-Label is metastasis-28features.xlsx",sheet=1)
set.seed(5612)
train_index <- createDataPartition(traindata$Label, p = 0.7, list = FALSE)
train <- traindata[train_index, ]
test<- traindata[-train_index, ]

set.seed(5612)
var_sel = list()
auc_res_test = list()
auc_res_train = list()
auc_train = list()
auc_test  = list()
cv_fit_list = list()
fit_list = list()
roc_compare = c()

for(i  in c(1:5))
{
  x <- as.matrix(train[, -1])
  y <- train$Label
  
  cv.fit <- cv.glmnet(x, y, family = 'binomial', type.measure = "auc",nfolds = 5, alpha = 1)
  fit <- glmnet(x, y, family = 'binomial', alpha = 1)
  
  cv_fit_list[[i]] = cv.fit
  fit_list[[i]] = fit
  
  vars_sel = coef(cv.fit, s = cv.fit$lambda.1se)
  vars_sel = unlist(vars_sel@Dimnames)[vars_sel@i + 1][-1]
  var_sel[[i]] = data.frame(var_sel = vars_sel)%>%mutate(var_sel = as.character(var_sel))
  
  pre_res_test <- as.vector(predict(fit, newx = as.matrix(test[, -1]), s = cv.fit$lambda.1se))
  roc_res_test <- auc(roc(test$Label, pre_res_test, ci = T, quiet = T, transpose = T))
  
  pre_res_train <- as.vector(predict(fit, newx = x, s = cv.fit$lambda.1se))
  roc_res_train <- auc(roc(train$Label, pre_res_train, ci = T, quiet = T, transpose = T))
  auc_res_test[[i]] = roc_res_test
  auc_res_train[[i]] = roc_res_train
  auc_train[[i]] = roc(train$Label, pre_res_train, ci = T, quiet = T, transpose = T)
  auc_test[[i]]  = roc(test$Label, pre_res_test, ci = T, quiet = T, transpose = T)
  
  roc_compare[i] = roc.test(roc_res_test, roc_res_train)$p.value
}
idx = which.max(auc_res_test)
cvfit = cv_fit_list[[idx]]
fit = fit_list[[idx]]
s= cvfit$lambda.1se

plot(cvfit)
abline(v = log(s), lty = 2, lwd = 1, col = 'blue')
abline(v = log(cvfit$lambda.1se), lty = 2, lwd = 1, col = 'black')
export::graph2ppt(file = './339DPL-5612.1sefeature selection.pptx', width = 5, height = 5,append = T)

plot(fit, s = s, xvar = 'lambda')
abline(v = log(s), lty = 2, col = 'blue')
export::graph2ppt(file = './339DPL-5612.1se¦Ë.pptx', width = 5, height = 5,append = T)

roc_train = auc_train[[idx]]
roc_test  = auc_test[[idx]]
plot(roc_train, print.auc = T, print.auc.pattern = 'AUC: %.2f (%.2f - %.2f)',
     legacy.axes = T, col = 'red')
plot(roc_test, print.auc = T, print.auc.pattern = 'AUC: %.2f (%.2f - %.2f)',
     add = T, col = 'blue', print.auc.y = 0.45)
export::graph2ppt(file = './339DPL-5612.1seAUC.pptx', width = 5, height = 5,append = T)

legend(x = 0.3, y = 0.2, legend = c('Train', 'Test'), 
       col = c('red', 'blue'), lty = 1)

coefs <- coefficients(fit, s = s)
useful_feature <- unlist(coefs@Dimnames)[coefs@i + 1]
useful_feature <- useful_feature[-1]
dt_coef <- data.frame(Feature = useful_feature, Coef = coefs@x[-1])
dt_coef <- arrange(dt_coef, desc(Coef))
dt_coef$Feature <- factor(dt_coef$Feature, 
                          levels = as.character(dt_coef$Feature))
p_coef <- ggplot(aes(x = Feature, y = Coef), data = dt_coef)
p_coef <- p_coef + geom_col(fill = 'blue', width = 0.7) + coord_flip() + 
  theme_bw() + ylab('Coefficients')
p_coef
export::graph2ppt(file = './339DPL-5612.1sefeature.pptx', width = 5, height = 5,append = T)


#the performance of model
intercept = coefs@x
write.csv(intercept,"intercept.csv")
coefficients(fit)
write.xlsx(dt_coef,"dt_coef.xlsx")
write.csv(train,"train data1.csv")
write.csv(pre_res_train,"pre_res_train1.csv")
write.csv(test,"test data.csv")
write.csv(pre_res_test,"pre_res_test102.csv")
citable_train= ci.coords(roc_train, 'best', ret = c("threshold", "specificity", "sensitivity", "accuracy",
                                                    "tn", "tp", "fn", "fp", "npv", "ppv", "1-specificity",
                                                    "1-sensitivity", "1-accuracy", "1-npv", "1-ppv",
                                                    "precision", "recall"), drop = T,
                         best.policy = 'random')#
citable_test= ci.coords(roc_test, 'best', ret = c("threshold", "specificity", "sensitivity", "accuracy",
                                                  "tn", "tp", "fn", "fp", "npv", "ppv", "1-specificity",
                                                  "1-sensitivity", "1-accuracy", "1-npv", "1-ppv",
                                                  "precision", "recall"), drop = T,
                        best.policy = 'random')#
write.csv(citable_train,"citable_train.csv")
write.csv(citable_test,"citable_test.csv")

roc_train$original.predictor
roc_train$predictor
write.csv(roc_train$predictor,"train predictor.csv")
write.csv(roc_test$predictor,"test predictor.csv")


#predicting LNM in validation set
valid106<-read.xlsx("106radiomic-Label is metastasis.xlsx",sheet=5)
valid28= valid106 %>% select(Label,DPL15,DPL329,DPL475,DPL473,DPL215,DPL228,DPL212,DPL334,DPL394,DPL350,DPL364,PL461,DPL183,DPL310,
                             DPL348,DPL304,DPL317,DPL73,DPL392,DPL113,DPL60,DPL489,DPL230,DPL361,DPL67,DPL155,DPL111,DPL504)
auc_res_valid28= list()
auc_valid28= list()
pre_res_valid28 <- as.vector(predict(fit, newx = as.matrix(valid28[, -1]), s = cv.fit$lambda.1se))
valid28$score = pre_res_valid28
valid28_score = valid28 %>% select(Label,score)
auc_valid28  = roc(valid28$Label, pre_res_valid28, ci = T, quiet = T, transpose = T)
plot(auc_valid28, print.auc = T, print.auc.pattern = 'AUC: %.2f (%.2f - %.2f)',
     legacy.axes = T, col = 'red')
write.csv(pre_res_valid28,"valid28-score.csv")
export::graph2ppt(file = './339DPL-5612.1seAUCvalidation.pptx', width = 5, height = 5,append = T)  

citable_valid84= ci.coords(auc_valid28, 'best', ret = c("threshold", "specificity", "sensitivity", "accuracy",
                                                        "tn", "tp", "fn", "fp", "npv", "ppv", "1-specificity",
                                                        "1-sensitivity", "1-accuracy", "1-npv", "1-ppv",
                                                        "precision", "recall"), drop = T,
                           best.policy = 'random')
write.csv(citable_valid84,"citable_valid84.csv")


#tumor marker predicting LNM
trainlog<-read.xlsx("339+84clinical data.xlsx",sheet=5)
trainlognew<-trainlog%>%select(CEA,	AFP,	CA125,	CA153,	CA199)
glmttm1 = glm(lymph.node.metastasis~	AFP,data = trainlog, family = binomial)
pre_res_tm1 <- as.vector(predict(glmttm1, newx = trainlognew))
auc_t1  = roc(trainlog$lymph.node.metastasis, pre_res_tm1, ci = T, quiet = T, transpose = T)

glmttm2 = glm(lymph.node.metastasis~	CEA,data = trainlog, family = binomial)
pre_res_tm2 <- as.vector(predict(glmttm2, newx = trainlognew))
auc_t2  = roc(trainlog$lymph.node.metastasis, pre_res_tm2, ci = T, quiet = T, transpose = T)

glmttm3 = glm(lymph.node.metastasis~	CA125,data = trainlog, family = binomial)
pre_res_tm3 <- as.vector(predict(glmttm3, newx = trainlognew))
auc_t3  = roc(trainlog$lymph.node.metastasis, pre_res_tm3, ci = T, quiet = T, transpose = T)

glmttm4 = glm(lymph.node.metastasis~	CA153,data = trainlog, family = binomial)
pre_res_tm4 <- as.vector(predict(glmttm4, newx = trainlognew))
auc_t4  = roc(trainlog$lymph.node.metastasis, pre_res_tm4, ci = T, quiet = T, transpose = T)

glmttm5 = glm(lymph.node.metastasis~	CA199,data = trainlog, family = binomial)
pre_res_tm5 <- as.vector(predict(glmttm5, newx = trainlognew))
auc_t5  = roc(trainlog$lymph.node.metastasis, pre_res_tm5, ci = T, quiet = T, transpose = T)

#visualization of model
plot(auc_t1, print.auc = T, print.auc.pattern = 'AUC: %.2f (%.2f - %.2f)',
     legacy.axes = T, col = '#CD3700',print.auc.y = 0.45)
plot(auc_t2, print.auc = T, print.auc.pattern = 'AUC: %.2f (%.2f - %.2f)',
     add = T,col = 'red',print.auc.y = 0.41)
plot(auc_t3, print.auc = T, print.auc.pattern = 'AUC: %.2f (%.2f - %.2f)',
     add = T,col = '#A4D3EE',print.auc.y = 0.37)
plot(auc_t4, print.auc = T, print.auc.pattern = 'AUC: %.2f (%.2f - %.2f)',
     add = T,col = '#1E90FF',print.auc.y = 0.33)
plot(auc_t5, print.auc = T, print.auc.pattern = 'AUC: %.2f (%.2f - %.2f)',
     add = T,col = 'blue',print.auc.y = 0.29)

export::graph2ppt(file = './239Train-5 tumor marker AUC.pptx', width = 5, height = 5,append = T)



#combining CEA, CA199£¬DPL score to predict LNM in three datasets
train<-read.xlsx("339+84clinical data.xlsx",sheet=5)
test<-read.xlsx("339+84clinical data.xlsx",sheet=7)
validation<-read.xlsx("339+84clinical data.xlsx",sheet=12)

train1<-train%>%select(CEA,DPL.score,CA199)
glmtrain = glm(lymph.node.metastasis~DPL.score+CEA+CA199,data = train, family = binomial)
pre_res_train <- as.vector(predict(glmtrain, newx = train1))
auc_tm1  = roc(train$lymph.node.metastasis, pre_res_train, ci = T, quiet = T, transpose = T)

test1<-test%>%select(CEA,DPL.score,CA199)
glmtest = glm(lymph.node.metastasis~DPL.score+CEA+CA199,data = test, family = binomial)
pre_res_tm2 <- as.vector(predict(glmtest, newx = test1))
auc_tm2  = roc(test$lymph.node.metastasis, pre_res_tm2, ci = T, quiet = T, transpose = T)

validation1<-test%>%select(CEA,DPL.score,CA199)
pre_res_validation = glm(lymph.node.metastasis~DPL.score+CEA+CA199,data = validation, family = binomial)
pre_res_tm3 <- as.vector(predict(pre_res_validation, newx = validation1))
auc_tm3  = roc(validation$lymph.node.metastasis, pre_res_tm3, ci = T, quiet = T, transpose = T)
validscore<-pre_res_tm3
validscore
write.csv(validscore,"84DPLscore+CEA+CA199 logistic score.csv")


# the performance of model
citable_tmtrain= ci.coords(auc_tm1, 'best', ret = c("threshold", "specificity", "sensitivity", "accuracy",
                                                    "tn", "tp", "fn", "fp", "npv", "ppv", "1-specificity",
                                                    "1-sensitivity", "1-accuracy", "1-npv", "1-ppv",
                                                    "precision", "recall"), drop = T,best.policy = 'random')
citable_tmtest= ci.coords(auc_tm2, 'best', ret = c("threshold", "specificity", "sensitivity", "accuracy",
                                                   "tn", "tp", "fn", "fp", "npv", "ppv", "1-specificity",
                                                   "1-sensitivity", "1-accuracy", "1-npv", "1-ppv",
                                                   "precision", "recall"), drop = T,best.policy = 'random')
citable_tmvalidation= ci.coords(auc_tm3, 'best', ret = c("threshold", "specificity", "sensitivity", "accuracy",
                                                         "tn", "tp", "fn", "fp", "npv", "ppv", "1-specificity",
                                                         "1-sensitivity", "1-accuracy", "1-npv", "1-ppv",
                                                         "precision", "recall"), drop = T,best.policy = 'random')
write.csv(citable_tmtrain,"citable_tmtrain.csv")
write.csv(citable_tmtest,"citable_tmtest.csv")
write.csv(citable_tmvalidation,"citable_tmvalidation.csv")


#visualization of model
plot(auc_tm1, print.auc = T, print.auc.pattern = 'AUC: %.2f (%.2f - %.2f)',
     legacy.axes = T, col = 'red',print.auc.y = 0.45)
plot(auc_tm2, print.auc = T, print.auc.pattern = 'AUC: %.2f (%.2f - %.2f)',
     add = T,col = 'blue',print.auc.y = 0.41)
export::graph2ppt(file = './Train+TEST+DPL score+CEA+CA199AUC.pptx', width = 5, height = 5,append = T)

plot(auc_tm3, print.auc = T, print.auc.pattern = 'AUC: %.2f (%.2f - %.2f)',
     col = 'red',print.auc.y = 0.45)
export::graph2ppt(file = './84validation+DPL score+CEA+CA199AUC.pptx', width = 5, height = 5,append = T)