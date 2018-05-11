library("DOSE")
library(clusterProfiler)
#upregulation gene enrichment
genesymbol<-read.table("D:\\zengyr\\about_drug_sensitivity\\classification_mode\\升华挑选出来基因的生物学意义\\上下调基因的富集分析\\genes_for_GO_upreg.txt",header=F)
#you can choose your gene to enrich. one line for one symbol.
genesymbol<-as.character(genesymbol$V1)
entrezid<-bitr(genesymbol, fromType="SYMBOL", toType="ENTREZID", OrgDb="org.Hs.eg.db")
toenrich<-enrichKEGG(gene=entrezid$ENTREZID, organism = "hsa", pvalueCutoff = 0.05, pAdjustMethod = "none",
                     use_internal_data = F)

dotplot(toenrich)

tofile<-"D:\\zengyr\\about_drug_sensitivity\\classification_mode\\升华挑选出来基因的生物学意义\\上下调基因的富集分析\\sensi_to_not_upreg.txt"
#you can select your path to save
write.table(toenrich,tofile,quote=F,row.names = F,sep="\t")

#上调GO，先要跑前面的
ego <- enrichGO(gene=entrezid$ENTREZID, OrgDb = "org.Hs.eg.db", ont = "ALL", pvalueCutoff = 0.05, pAdjustMethod = "fdr",
                readable = T)
dotplot(ego)

tofilego<-"D:\\zengyr\\about_drug_sensitivity\\classification_mode\\升华挑选出来基因的生物学意义\\上下调基因的富集分析\\go_sensi_to_not_upreg.txt"
write.table(ego,tofilego,quote=F,row.names = F,sep="\t")

################################################################
#downregulation gene enrichment
genesymbol<-read.table("D:\\zengyr\\about_drug_sensitivity\\classification_mode\\升华挑选出来基因的生物学意义\\上下调基因的富集分析\\genes_for_GO_downreg.txt",header=F)
genesymbol<-as.character(genesymbol$V1)
entrezid<-bitr(genesymbol, fromType="SYMBOL", toType="ENTREZID", OrgDb="org.Hs.eg.db")
toenrich<-enrichKEGG(gene=entrezid$ENTREZID, organism = "hsa", pvalueCutoff = 0.05, pAdjustMethod = "none",
                     use_internal_data = F)
dotplot(toenrich)

tofile<-"D:\\zengyr\\about_drug_sensitivity\\classification_mode\\升华挑选出来基因的生物学意义\\上下调基因的富集分析\\sensi_to_not_downreg.txt"
#you can select your path to save
write.table(toenrich,tofile,quote=F,row.names = F,sep="\t")

#下调GO，先要跑前面的
ego <- enrichGO(gene=entrezid$ENTREZID, OrgDb = "org.Hs.eg.db", ont = "ALL", pvalueCutoff = 0.05, pAdjustMethod = "fdr",
                readable = T)

dotplot(ego)

tofilego<-"D:\\zengyr\\about_drug_sensitivity\\classification_mode\\升华挑选出来基因的生物学意义\\上下调基因的富集分析\\go_sensi_to_not_downreg.txt"
write.table(ego,tofilego,quote=F,row.names = F,sep="\t")
