#library packages
library(openxlsx)
library(export)
library(dplyr)
library(pheatmap)
library(corrplot)
library(RColorBrewer)

#the correaltion of 339Label,DPL score and clinical characteristic#
cordata<-read.xlsx("339 correlation heatmap.xlsx",sheet=1,rowNames = T)
cordata<-data.frame(cordata)
cor_matr = cor(cordata)
corrplot(cor_matr, type="upper", order="hclust", tl.col="black", tl.srt=45)
export::graph2ppt(file = './339correlation heatmap.pptx', width = 10, height = 10,append = T)
chart.Correlation(cordata,histogram = TRUE,pch=19)
export::graph2ppt(file = './84+341correlation heatmap33.pptx', width = 20, height = 20,append = T)


#the correlation of DPL features and gene#
gene<-read.xlsx("130 CRC  TPM value+gene name-T.xlsx",sheet=1)
gene84<-gene%>%select(Gene.name,T218,T123, T32,T101,T103,T33,T219,T229,T191,T39,T217,T165,T189,T125,T221,T105,T182,T24,T40,T92,
                      T53,T111,T188,T21,T26,T117,T94,T195,T43,T29,T31,T106,T52,T28,T73,T38,T36,T50,T80,T74,T181,T54,T55,T71,
                      T201,T82,T77,T142,T176,T104,T108,T109,T127,T130,T132,T133,T136,T134,T138,T143,T150,T160,T163,T161,T173,T167,
                      T174,T175,T186,T178,T177,T179,T197,T200,T202,T205,T208,T209,T215,T216,T220,T222,T223,T226)
as.matrix(gene84)
write.xlsx(gene84,"84TPM.xlsx")

datacor1<-read.xlsx("84TPM.xlsx",sheet=2,rowNames = T)
datacor1<-as.matrix(datacor1)
datacor1<-datacor1[rowSums(datacor1)>84,]
datacor1<-t(datacor1)
write.xlsx(datacor1,"datacor1.xlsx")

datacor22<-read.xlsx("106radiomic-Label is metastasis.xlsx",sheet=5)
datacor2<-datacor22%>%select(  DPL15,DPL73,DPL230,DPL67,DPL317,DPL461,DPL329,DPL475,DPL361,DPL183,DPL504,DPL60,DPL394,DPL489,
                               DPL155,DPL228,DPL473,DPL215,DPL212,DPL392)
write.xlsx(datacor2,"84-20¸öDPL features.xlsx")

datacor1<-read.xlsx("datacor1.xlsx",sheet=1)
datacor2<-read.xlsx("84-20¸öDPL features.xlsx",sheet=1)
corhmisc<-corr.test(datacor1, datacor2, method = "spearman",adjust="none")
cmt <-corhmisc$r
pmt <-corhmisc$p
head(cmt)

cmt.out<-cbind(rownames(cmt),cmt)
write.table(cmt.out,file="cor.txt",sep="\t",row.names=F)
pmt.out<-cbind(rownames(pmt),pmt)
write.table(pmt.out,file="pvalue.txt",sep="\t",row.names=F)
df<-melt(cmt,value.name="corhmisc")
df$pvalue <-as.vector(pmt)
head(df)

write.table(df,file="cor-p.xlsx",sep="\t")
#write.table(df,file="cor-p.txt",sep="\t")
if (!is.null(pmt)){
  ssmt <- pmt< 0.01
  pmt[ssmt] <-'**'
  smt <- pmt >0.01& pmt <0.05
  pmt[smt] <- '*'
  pmt[!ssmt&!smt]<- ''
}else {
  pmt <- F
}
mycol<-colorRampPalette(c("blue","white","red"))(800)
pheatmap(cmt,scale = "none",cluster_row = T, cluster_col = T, border=NA,
         display_numbers = pmt,fontsize_number = 12, number_color = "white",
         cellwidth = 20, cellheight =20,color=mycol)
graph2ppt(file = 'correlation.pptx', width = 8, height = 8, append = T)


#16384*84 heatmap#
datacor1<-read.xlsx("datacor1.xlsx",sheet=1)
datacor1t<-t(datacor1)
library(pheatmap)
library(RColorBrewer)
color<-colorRampPalette(c('red','white','blue'))(100)   
pheatmap(datacor1t,scale="row",color = color,clustering_method = "complete",cluster_cols = FALSE,fontsize = 10, show_rownames = FALSE,cex=1)


#349*84 heatmap#
DEGdata<-read.xlsx("84-349DEGs.xlsx",sheet=5)
library(pheatmap)
library(RColorBrewer)
color<-colorRampPalette(c('blue','white','red'))(100)   
pheatmap(DEGdata,scale="row",color = color,clustering_method = "complete",cluster_cols = FALSE,fontsize = 10, show_rownames = FALSE,cex=1)

#297*84heatmap in immune pathways#
Immudataclusgene<-read.csv("297 genes enriched in immune pathways.csv",header=T,row.names=1,stringsAsFactors = F)  
pheatmap(Immudataclusgene,scale="row",color = color,clustering_method = "complete",
         fontsize = 5, cex=1, )
annotation_col <-read.xlsx("297 genes enriched in immune pathways.xlsx",sheet=9)
annotation_col
rownames(annotation_col) = paste("Immudataclusgene", 1:10, sep = "")
pheatmap(Immudataclusgene, annotation_col = annotation_col)

#correlation of 1751 gene and 20 DPL features#
datacor11<-read.xlsx("84-1751gene.xlsx",sheet=3)
datacor22<-read.xlsx("84-20 DPLfeatures.xlsx",sheet=1)
corhmisc11<-corr.test(datacor11, datacor22, method = "spearman",adjust="none")
cmt11 <-corhmisc11$r
pmt11 <-corhmisc11$p
head(cmt)

cmt.out11<-cbind(rownames(cmt11),cmt11)
write.table(cmt.out11,file="cor11.txt",sep="\t",row.names=F)
pmt.out11<-cbind(rownames(pmt11),pmt11)
write.table(pmt.out,file="pvalue.txt",sep="\t",row.names=F)
df11<-melt(cmt11,value.name="corhmisc11")
df11$pvalue <-as.vector(pmt11)
head(df)

write.table(df,file="cor-p.xlsx",sep="\t")
#write.table(df,file="cor-p.txt",sep="\t")
if (!is.null(pmt)){
  ssmt <- pmt< 0.01
  pmt[ssmt] <-'**'
  smt <- pmt >0.01& pmt <0.05
  pmt[smt] <- '*'
  pmt[!ssmt&!smt]<- ''
}else {
  pmt <- F
}
mycol<-colorRampPalette(c("blue","white","red"))(800)
pheatmap(cmt11,scale = "none",cluster_row = T, cluster_col = T, border=NA,
         display_numbers = pmt11,fontsize_number = 2, number_color = "white",
         cellwidth = 20, cellheight =2,color=mycol)
pheatmap(cmt11,scale = "none",cluster_row = T, cluster_col = T, border=NA,
         cellwidth = 10, cellheight =0.15,color=mycol)
graph2ppt(file = 'correlation.pptx', width = 8, height = 8, append = T)


#the heatmap of DEGs
heatmap_data<-read.csv("84-45 gene heatmap.csv",header=T,row.names=1,stringsAsFactors = F)  
color<-colorRampPalette(c('#436eee','white','#EE0000'))(100)     
pheatmap(heatmap_data,scale="row",color = color,clustering_method = "complete",cluster_cols = FALSE,fontsize = 10, cex=1)
png(filename="heatmap.png",height=4000,width=4000,res=500,units="px")  
dev.off()