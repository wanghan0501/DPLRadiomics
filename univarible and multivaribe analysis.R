#library packages
library(openxlsx)
library(glmnet)


#Multivariate analysis for DPL features
data<-read.xlsx("339-Label is metastasis-28 features.xlsx",sheet=3)
GOR1<-glm(Label~.,data=data)
GSUM1<-summary(GOR1)
OR1<-exp(coef(GOR1))
IC1<-exp(confint(GOR1))
Pvalue1<-round(GSUM1$coefficients[,4],3)


#univariate analysis for DPL features
data<-read.xlsx("339-Label is metastasis-28 features.xlsx",sheet=3)
GOR2<-glm(Label~DPL15,data=data)
GSUM2<-summary(GOR2)
OR2<-exp(coef(GOR2))
IC2<-exp(confint(GOR2))
Pvalue2<-round(GSUM2$coefficients[,4],3)

GOR3<-glm(Label~DPL329,data=data)
GSUM3<-summary(GOR3)
OR3<-exp(coef(GOR3))
IC3<-exp(confint(GOR3))
Pvalue3<-round(GSUM3$coefficients[,4],3)

GOR4<-glm(Label~DPL475,data=data)
GSUM4<-summary(GOR4)
OR4<-exp(coef(GOR4))
IC4<-exp(confint(GOR4))
Pvalue4<-round(GSUM4$coefficients[,4],3)

GOR5<-glm(Label~DPL473,data=data)
GSUM5<-summary(GOR5)
OR5<-exp(coef(GOR5))
IC5<-exp(confint(GOR5))
Pvalue5<-round(GSUM5$coefficients[,4],3)

GOR6<-glm(Label~DPL215,data=data)
GSUM6<-summary(GOR6)
OR6<-exp(coef(GOR6))
IC6<-exp(confint(GOR6))
Pvalue6<-round(GSUM6$coefficients[,4],3)

GOR7<-glm(Label~DPL228,data=data)
GSUM7<-summary(GOR7)
OR7<-exp(coef(GOR7))
IC7<-exp(confint(GOR7))
Pvalue7<-round(GSUM7$coefficients[,4],3)

GOR8<-glm(Label~DPL212,data=data)
GSUM8<-summary(GOR8)
OR8<-exp(coef(GOR8))
IC8<-exp(confint(GOR8))
Pvalue8<-round(GSUM8$coefficients[,4],3)

GOR9<-glm(Label~DPL394,data=data)
GSUM9<-summary(GOR9)
OR9<-exp(coef(GOR9))
IC9<-exp(confint(GOR9))
Pvalue9<-round(GSUM9$coefficients[,4],3)

GOR10<-glm(Label~DPL461,data=data)
GSUM10<-summary(GOR10)
OR10<-exp(coef(GOR10))
IC10<-exp(confint(GOR10))
Pvalue10<-round(GSUM10$coefficients[,4],3)

GOR11<-glm(Label~DPL183,data=data)
GSUM11<-summary(GOR11)
OR11<-exp(coef(GOR11))
IC11<-exp(confint(GOR11))
Pvalue11<-round(GSUM11$coefficients[,4],3)

GOR12<-glm(Label~DPL317,data=data)
GSUM12<-summary(GOR12)
OR12<-exp(coef(GOR12))
IC12<-exp(confint(GOR12))
Pvalue12<-round(GSUM12$coefficients[,4],3)

GOR13<-glm(Label~DPL73,data=data)
GSUM13<-summary(GOR13)
OR13<-exp(coef(GOR13))
IC13<-exp(confint(GOR13))
Pvalue13<-round(GSUM13$coefficients[,4],3)

GOR14<-glm(Label~DPL392,data=data)
GSUM14<-summary(GOR14)
OR14<-exp(coef(GOR14))
IC14<-exp(confint(GOR14))
Pvalue14<-round(GSUM14$coefficients[,4],3)

GOR15<-glm(Label~DPL60,data=data)
GSUM15<-summary(GOR15)
OR15<-exp(coef(GOR15))
IC15<-exp(confint(GOR15))
Pvalue15<-round(GSUM15$coefficients[,4],3)

GOR16<-glm(Label~DPL489,data=data)
GSUM16<-summary(GOR16)
OR16<-exp(coef(GOR16))
IC16<-exp(confint(GOR16))
Pvalue16<-round(GSUM16$coefficients[,4],3)

GOR17<-glm(Label~DPL230,data=data)
GSUM17<-summary(GOR17)
OR17<-exp(coef(GOR17))
IC17<-exp(confint(GOR17))
Pvalue17<-round(GSUM17$coefficients[,4],3)

GOR18<-glm(Label~DPL361,data=data)
GSUM18<-summary(GOR18)
OR18<-exp(coef(GOR18))
IC18<-exp(confint(GOR18))
Pvalue18<-round(GSUM18$coefficients[,4],3)

GOR19<-glm(Label~DPL67,data=data)
GSUM19<-summary(GOR19)
OR19<-exp(coef(GOR19))
IC19<-exp(confint(GOR19))
Pvalue19<-round(GSUM19$coefficients[,4],3)

GOR20<-glm(Label~DPL155,data=data)
GSUM20<-summary(GOR20)
OR20<-exp(coef(GOR20))
IC20<-exp(confint(GOR20))
Pvalue20<-round(GSUM20$coefficients[,4],3)

GOR21<-glm(Label~DPL504,data=data)
GSUM21<-summary(GOR21)
OR21<-exp(coef(GOR21))
IC21<-exp(confint(GOR21))
Pvalue21<-round(GSUM21$coefficients[,4],3)


#univariate analysis for DEGs
data<-read.xlsx("84-12 gene TPM.xlsx",sheet=2)
head(data)
GOR1<-glm(Metastasis~AC132217.1,data=data)
GSUM1<-summary(GOR1)
OR1<-exp(coef(GOR1))
IC1<-exp(confint(GOR1))
Pvalue1<-round(GSUM1$coefficients[,4],3)

GOR2<-glm(Metastasis~MYOC,data=data)
GSUM2<-summary(GOR2)
OR2<-exp(coef(GOR2))
IC2<-exp(confint(GOR2))
Pvalue2<-round(GSUM2$coefficients[,4],3)

GOR3<-glm(Metastasis~IGHV1OR164,data=data)
GSUM3<-summary(GOR3)
OR3<-exp(coef(GOR3))
IC3<-exp(confint(GOR3))
Pvalue3<-round(GSUM3$coefficients[,4],3)

GOR4<-glm(Metastasis~IGLVI70,data=data)
GSUM4<-summary(GOR4)
OR4<-exp(coef(GOR4))
IC4<-exp(confint(GOR4))
Pvalue4<-round(GSUM4$coefficients[,4],3)

GOR5<-glm(Metastasis~AC084337.2,data=data)
GSUM5<-summary(GOR5)
OR5<-exp(coef(GOR5))
IC5<-exp(confint(GOR5))
Pvalue5<-round(GSUM5$coefficients[,4],3)

GOR6<-glm(Metastasis~HAS1,data=data)
GSUM6<-summary(GOR6)
OR6<-exp(coef(GOR6))
IC6<-exp(confint(GOR6))
Pvalue6<-round(GSUM6$coefficients[,4],3)

GOR7<-glm(Metastasis~IGLC7,data=data)
GSUM7<-summary(GOR7)
OR7<-exp(coef(GOR7))
IC7<-exp(confint(GOR7))
Pvalue7<-round(GSUM7$coefficients[,4],3)

GOR8<-glm(Metastasis~ADAMTS9AS1,data=data)
GSUM8<-summary(GOR8)
OR8<-exp(coef(GOR8))
IC8<-exp(confint(GOR8))
Pvalue8<-round(GSUM8$coefficients[,4],3)

GOR9<-glm(Metastasis~VEGFD,data=data)
GSUM9<-summary(GOR9)
OR9<-exp(coef(GOR9))
IC9<-exp(confint(GOR9))
Pvalue9<-round(GSUM9$coefficients[,4],3)

GOR10<-glm(Metastasis~PPP1R1A,data=data)
GSUM10<-summary(GOR10)
OR10<-exp(coef(GOR10))
IC10<-exp(confint(GOR10))
Pvalue10<-round(GSUM10$coefficients[,4],3)

GOR11<-glm(Metastasis~THRSP,data=data)
GSUM11<-summary(GOR11)
OR11<-exp(coef(GOR11))
IC11<-exp(confint(GOR11))
Pvalue11<-round(GSUM11$coefficients[,4],3)


#Multivariate analysis for DEGs
data<-read.xlsx("84-12 gene TPM.xlsx",sheet=2)
GOR<-glm(Metastasis~.,data=data)
GSUM<-summary(GOR)
OR<-exp(coef(GOR))
IC<-exp(confint(GOR))
Pvalue<-round(GSUM$coefficients[,4],3)