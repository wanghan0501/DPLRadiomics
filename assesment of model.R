#library packages
library(rms)
library(calibrate)
library(export)

#calibration curve#
train<-read.csv("train.csv")
f1 <- lrm(Label~DPL15+DPL73+DPL230+DPL67+DPL317+DPL461+DPL329+DPL475+DPL361+DPL183+DPL504+DPL60+DPL394+
          DPL489+DPL155+DPL228+DPL473+DPL215+DPL212+DPL392,x=T, y=T, data = train) 
cal1 <- rms::calibrate(f1,cmethod="KM",method="boot",m=115,B=1000,legend=FALSE)
plot(cal1,lwd=2,lty=1,xlab=" Predicted",ylab=list("Actual "),xlim=c(0,1),ylim=c(0,1))
export::graph2ppt(file = './339DPL-5612.1se calibration curve train.pptx', width = 5, height = 5,append = T)

test<-read.csv("test.csv")
f2 <- lrm(Label~DPL15+DPL73+DPL230+DPL67+DPL317+DPL461+DPL329+DPL475+DPL361+DPL183+DPL504+DPL60+DPL394+
          DPL489+DPL155+DPL228+DPL473+DPL215+DPL212+DPL392,x=T, y=T, data = test) 
cal2 <- rms::calibrate(f2,cmethod="KM",method="boot",m=115,B=1000,legend=FALSE)
plot(cal2,lwd=2,lty=1,xlab=" Predicted",ylab=list("Actual "),xlim=c(0,1),ylim=c(0,1))
export::graph2ppt(file = './100DPL-5612.1se calibration curve test.pptx', width = 5, height = 5,append = T)

valid84<-read.xlsx("106radiomic-Label is metastasis.xlsx",sheet=5)
f3 <- lrm(Label~DPL15+DPL73+DPL230+DPL67+DPL317+DPL461+DPL329+DPL475+DPL361+DPL183+DPL504+DPL60+DPL394+
          DPL489+DPL155+DPL228+DPL473+DPL215+DPL212+DPL392,x=T, y=T, data =valid84) 
cal3 <- rms::calibrate(f3,cmethod="KM",method="boot",m=115,B=1000,legend=FALSE)
plot(cal3,lwd=2,lty=1,xlab=" Predicted",ylab=list("Actual "),xlim=c(0,1),ylim=c(0,1))
export::graph2ppt(file = './84DPL-5612.1se calibration curve valid.pptx', width = 5, height = 5,append = T)


#decision curve#
library(rmda)
train<-read.csv("train.csv")
DPL<- decision_curve(Label~DPL15+DPL73+DPL230+DPL67+DPL317+DPL461+DPL329+DPL475+DPL361+DPL183+DPL504+DPL60+DPL394+
                       DPL489+DPL155+DPL228+DPL473+DPL215+DPL212+DPL392,data = train,
                     family = binomial(link ='logit'), thresholds = seq(0,1, by = 0.01),
                     confidence.intervals= 0.95,study.design = 'case-control',
                     population.prevalence= 0.3)
plot_decision_curve(DPL,curve.names= c('complex'),
                    cost.benefit.axis =FALSE,col = c('red'),
                    confidence.intervals =FALSE,standardize = FALSE)
traincom<-read.xlsx("train+CEA+CA199.xlsx",sheet=3)
combined<- decision_curve(Label ~ DPL.score+CEA+CA199,data = traincom,
                          family = binomial(link ='logit'), thresholds = seq(0,1, by = 0.01),
                          confidence.intervals= 0.95,study.design = 'case-control',
                          population.prevalence= 0.3)
plot_decision_curve(combined,curve.names= c('Combined'),
                    cost.benefit.axis =FALSE,col = c('red'),
                    confidence.intervals =FALSE,standardize = FALSE)
#plot two decision curves
List<- list(DPL,combined)
plot_decision_curve(list(DPL,combined),
                    curve.names = c("DPL model", "Combined model"),
                    confidence.intervals = FALSE,
                    cost.benefit.axis = FALSE,
                    lwd = 1.2,
                    col = c("blue", "red"),
                    cex.lab = 1.5,  
                    cex.axis = 1.5, 
                    mgp = c(4,1,0))
graph2ppt(file = 'decision curve DPL and combined.pptx', width = 5, height =5, append = T)