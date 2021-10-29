#library packages
library(openxlsx)
library(ggplot2)
library(DESeq2)


#The DEGs analysis
countData=read.csv("related gene 130 CRC  counts value+ENSG-T.csv")
coutData=countData%>%select(genename,T218,T123, T32,T101,T103,T33,T219,T229,T191,T39,T217,T165,T189,T125,T221,T105,T182,T24,T40,T92,
                            T53,T111,T188,T21,T26,T117,T94,T195,T43,T29,T31,T106,T52,T28,T73,T38,T36,T50,T80,T74,T181,T54,T55,T71,
                            T201,T82,T77,T142,T176,T104,T108,T109,T127,T130,T132,T133,T136,T134,T138,T143,T150,T160,T163,T161,T173,T167,
                            T174,T175,T186,T178,T177,T179,T197,T200,T202,T205,T208,T209,T215,T216,T220,T222,T223,T226)
write.csv(coutData,"84-counts.csv")
countData<-read.csv("84-counts.csv",row.names=1)
as.matrix(countData)
countData<-round(as.matrix(countData))
countData = countData[rowSums(countData)>84, ]
head(countData)
colData=read.csv("group list.csv",row.names = 1)
colData
dds<-DESeqDataSetFromMatrix(countData=countData,colData=colData,design=~group)
dds$group <- relevel(dds$group, ref = "nonmetastasis")
dds2 <- DESeq(dds)
resultsNames(dds2)
res <- results(dds2, name="group_metastasis_vs_nonmetastasis")
summary(res)
sum(res$padj < 0.05, na.rm=TRUE)
write.csv(res,file = "DEGs Data-84.csv")


#DPL related genes-GO analysis visualization#
go_enrich_df<-read.xlsx("GO_KEGG.xlsx",sheet = 8)
as.data.frame(go_enrich_df)
shorten_names <- function(x, n_word=3, n_char=30){
  if (length(strsplit(x, " ")[[1]]) > n_word || (nchar(x) > 30))
  {
    if (nchar(x) > 30) x <- substr(x, 1, 30)
    x <- paste(paste(strsplit(x, " ")[[1]][1:min(length(strsplit(x," ")[[1]]), n_word)],
                     collapse=" "), "...", sep="")
    return(x)
  } 
  else
  {
    return(x)
  }
}
go_enrich_df$Description=factor(go_enrich_df$Description,levels = rev(go_enrich_df$Description))
labels=(sapply(  levels(go_enrich_df$Description)[as.numeric(go_enrich_df$Description)],
  shorten_names))
names(labels) = rev(1:nrow(go_enrich_df))
CPCOLS <- c("red", "blue", "#66C3A5")

p<-ggplot(go_enrich_df, aes(x=Description, y=GeneInGOAndHitList, fill=Category)) +
  geom_bar(stat="identity", width=0.8) + coord_flip() + scale_fill_manual(values = CPCOLS) + theme_bw() + 
  scale_x_discrete(labels=labels) + xlab("GO term") + ylab("GeneCount")+theme(axis.text=element_text(face = "bold", color="gray50"))
  +labs(title = "The Most Enriched GO Terms")
export::graph2ppt(file = 'DPLrelated genes-GO.pptx', width = 10, height = 10, append = T)


pathway<-read.xlsx("GO_KEGG.xlsx",sheet = 11)
p=ggplot(pathway,aes(Enrichment,Description))
p=p+geom_point(aes(size=GeneInGOAndHitList))
p=p+geom_point(aes(size=GeneInGOAndHitList,color=qvalue))
p=p+scale_color_gradient(low="red",high = "blue")+labs(color="qvalue",size="Count",x="Rich Ratio",y="KEGG Pathway")
p=p+theme_bw()
export::graph2ppt(file = 'DPLrelated genes-KEGG.pptx', width = 10, height =10, append = T)



#heatmap of related gene#
library(DESeq2)
countData=read.csv("related gene 130 CRC  counts value+ENSG-T.csv")
coutData=countData%>%select(genename,T218,T123, T32,T101,T103,T33,T219,T229,T191,T39,T217,T165,T189,T125,T221,T105,T182,T24,T40,T92,
                            T53,T111,T188,T21,T26,T117,T94,T195,T43,T29,T31,T106,T52,T28,T73,T38,T36,T50,T80,T74,T181,T54,T55,T71,
                            T201,T82,T77,T142,T176,T104,T108,T109,T127,T130,T132,T133,T136,T134,T138,T143,T150,T160,T163,T161,T173,T167,
                            T174,T175,T186,T178,T177,T179,T197,T200,T202,T205,T208,T209,T215,T216,T220,T222,T223,T226)
write.csv(coutData,"84-counts.csv")
countData<-read.csv("84-counts.csv",row.names=1)
as.matrix(countData)
countData<-round(as.matrix(countData))
countData = countData[rowSums(countData)>84, ]
colData=read.csv("group list.csv",row.names = 1)
dds<-DESeqDataSetFromMatrix(countData=countData,colData=colData,design=~group)
dds$group <- relevel(dds$group, ref = "nonmetastasis")
dds2 <- DESeq(dds)
resultsNames(dds2)
res <- results(dds2, name="group_metastasis_vs_nonmetastasis")
sum(res$padj < 0.05, na.rm=TRUE)
write.csv(res,file = "DEGs Data-84.csv")

library(pheatmap)
library(RColorBrewer)
heatmap_data<-read.csv("84-45 gene heatmap.csv",header=T,row.names=1,stringsAsFactors = F)  
color<-colorRampPalette(c('#436eee','white','#EE0000'))(100)     
pheatmap(heatmap_data,scale="row",color = color,clustering_method = "complete",cluster_cols = FALSE,fontsize = 10, cex=1)
png(filename="heatmap.png",height=4000,width=4000,res=500,units="px")  
dev.off()