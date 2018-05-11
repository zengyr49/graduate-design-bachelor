setwd("D:\\zengyr\\about_drug_sensitivity\\classification_mode")
filename<-"134_followup_joindate_ep100.txt"
data_sensi<-read.table(filename,header = T)
library(survival)
library(survminer)
mfit=survfit(Surv(days_to_last_followup,vital_status=="Dead")~sensi_not,data=data_sensi)
# plot(mfit)
ggsurvplot(mfit,conf.int = F,pval = T,risk.table = T)
