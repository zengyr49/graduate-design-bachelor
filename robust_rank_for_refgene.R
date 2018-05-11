library(RobustRankAggreg)
com.feature <- function(x,y,method="merge"){
  com=c()
  if(method == "merge"){
    com=unique(c(x,y))
  }
  if(method=="overlap"){
    com=intersect(x,y)
  }
  return(com)
}

##use rra algorithm to aggregate genenames##
prefix<-"/data1/data/TCGA/RNA_Seq_FPKM_2017_4_1/"
namelist<-c('ACC1','BLCA1','BRCA1','CESC1','CHOL1','COAD1','DLBC1','ESCA1','GBM1','HNSC1','KICH1','KIRC1','KIRP1','LAML1','LGG1','LIHC1','LUAD1','LUSC1','MESO1','OV1','PAAD1','PCPG1','PRAD1','READ1','SARC1','SKCM1','STAD1','TGCT1','THCA1','THYM1','UCEC1','UCS1','UVM1')
print('now loading microarray data...')
microarray.data<-read.table('/data1/data/zengyanru/R_find_maxexpr/Cell_line_RMA_proc_basalExp_clear.txt',sep='\t',header = TRUE,na.strings = 0,fill = TRUE)
rownames(microarray.data)<-microarray.data[,1]
microarray.data<-microarray.data[,c(-1,-2)]

for (i in namelist){
  outputname<-paste('top20_',i,'.txt',sep='',collapse = '')
  ##find out common gene##
  info<-paste('now loading ',i,sep = '',collapse = '')
  print(info)
  wholename<-paste(prefix,i,'.txt',sep='',collapse = '')
  rnaseq.data<-read.table(wholename,sep='\t',header = TRUE,na.strings = 0,fill = TRUE)
  rnaseq.data<-rnaseq.data[!rnaseq.data[,2]=='',]
  rnaseq.data<-rnaseq.data[!duplicated(rnaseq.data[,2]),]
  rownames(rnaseq.data)<-rnaseq.data[,2]
  rnaseq.data<-rnaseq.data[,c(-1,-2,-3)]
  gene.com<-com.feature(rownames(microarray.data),rownames(rnaseq.data),method="overlap")
  rnaseq.data.com<-as.matrix(rnaseq.data[gene.com,])
  ##finding finished##
  ##rank the gene
  print('now begin to rank the genes')
  genename<-apply(rnaseq.data.com,2,function(x){odr<-order(x,decreasing = TRUE);return(rownames(rnaseq.data.com)[odr])})
  genename50<-head(genename,n=50)
  print('now using package to aggregate genes')
  rankorder<-rankMatrix(genename50)
  aggregate_rank<-aggregateRanks(rmat = rankorder)
  write.table(head(aggregate_rank,n=20),file=outputname)
  print('write completed.')
}

microarray.data.com<-as.matrix(microarray.data[gene.com,])
##finding finished##
##rank the gene
print('now begin to rank the genes this is microarray')
ma_genename<-apply(microarray.data.com,2,function(x){odr<-order(x,decreasing = TRUE);return(rownames(microarray.data.com)[odr])})
genename50<-head(ma_genename,n=50)
print('now using package to aggregate genes')
rankorder<-rankMatrix(genename50)
aggregate_rank<-aggregateRanks(rmat = rankorder)
write.table(head(aggregate_rank,n=20),file='top20_microarray.txt')
print('write completed.')
