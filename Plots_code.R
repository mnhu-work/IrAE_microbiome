
rm(list=ls())
setwd('D:/data/IrAE_microbiome/IrAE_analysis/PD1_IrAE/')
load('../IrAE_imput_data.RData')
load('IrAE_all_PD1_wilcox.RData')

feat<-feat[which(rowSums(feat) > 0),]
feat.all<-feat[,rownames(meta)]
feat.all<-apply(feat, 2, function(x) x/sum(x))

library(tidyverse)
meta<-meta%>%filter(meta$Treatment=='anti-PD1')


load('../PD1_IrAE_wilcox.RData')
feat.all<-as.matrix(feat.all)
feat.all<-feat.all[,rownames(meta)]

# Packages
library("tidyverse")
library("coin")
library("pROC")
library("yaml")

##wilcox.test for differential genus
studies <- meta %>% pull(Study) %>% unique
meta$IrAE<-ifelse(meta$IrAE==0,'nonIrAE','IrAE')
meta$IrAE<-as.factor(meta$IrAE)
sum(is.na(feat.all))
# block for colonoscopy and study as well
meta <- meta %>%
  mutate(block=Study)
meta$block<-as.factor(meta$block)
meta$Sample_ID<-rownames(meta)
feat.all <- feat[,meta$Sample_ID]
feat.all<-as.matrix(feat.all)
studies <- meta %>% pull(Study) %>% unique
#
#feat.all<-apply(feat, 2, function(x) x/sum(x))
# ##############################################################################
# calculate pval
p.val <- matrix(NA, nrow=nrow(feat.all), ncol=length(studies)+1, 
                dimnames=list(row.names(feat.all), c(studies, 'all')))
pb <- txtProgressBar(max=nrow(feat.all), style=3)
fc <- p.val
aucs.mat <- p.val
aucs.all  <- vector('list', nrow(feat.all))
log.n0 <-as.numeric(1e-05)
mult.corr <- 'fdr'
studies <- meta %>% pull(Study) %>% unique

cat("Calculating effect size for every feature...\n")
pb <- txtProgressBar(max=nrow(feat.all), style=3)

# caluclate wilcoxon test and effect size for each feature and study
for (f in row.names(feat.all)) {
  
  # for each study
  for (s in studies) {
    
    x <- feat.all[f, meta %>% filter(Study==s) %>% 
                    filter(IrAE=='IrAE') %>% pull(Sample_ID)]
    y <- feat.all[f, meta %>% filter(Study==s) %>% 
                    filter(IrAE=='nonIrAE') %>% pull(Sample_ID)]
    
    # Wilcoxon
    p.val[f,s] <- wilcox.test(x, y, exact=FALSE)$p.value
    
    # AUC
    aucs.all[[f]][[s]] <- c(roc(controls=y, cases=x, 
                                direction='<', ci=TRUE, auc=TRUE)$ci)
    aucs.mat[f,s] <- c(roc(controls=y, cases=x, 
                           direction='<', ci=TRUE, auc=TRUE)$ci)[2]
    
    # FC
    q.p <- quantile(log10(x+log.n0), probs=seq(.1, .9, .05),na.rm=TRUE)
    q.n <- quantile(log10(y+log.n0), probs=seq(.1, .9, .05),na.rm=TRUE)
    fc[f, s] <- sum(q.p - q.n)/length(q.p)
  }
  
  # calculate effect size for all studies combined
  # Wilcoxon + blocking factor
  d <- data.frame(y=feat.all[f,], 
                  x=meta$IrAE, block=meta$block)
  p.val[f,'all'] <- pvalue(wilcox_test(y ~ x | block, data=d))
  # other metrics
  x <- feat.all[f, meta %>% filter(IrAE=='IrAE') %>% pull(Sample_ID)]
  y <- feat.all[f, meta %>% filter(IrAE=='nonIrAE') %>% pull(Sample_ID)]
  # FC
  fc[f, 'all'] <- mean(fc[f, studies])
  # AUC
  aucs.mat[f,'all'] <- c(roc(controls=y, cases=x, 
                             direction='<', ci=TRUE, auc=TRUE)$ci)[2]
  
  # progressbar
  setTxtProgressBar(pb, (pb$getVal()+1))
}
cat('\n')

# multiple hypothesis correction
p.adj <- data.frame(apply(p.val, MARGIN=2, FUN=p.adjust, method=mult.corr),
                    check.names = FALSE)
# save results for meta-analysis in supplementary table
df.all <- tibble(mOTUs=rownames(feat.all),
                 fc=fc[,'all'],
                 auroc=aucs.mat[,'all'],
                 p.val=p.val[,'all'],
                 p.adj=p.adj[,'all'])

class(df.all)
save.image(file='IrAE_all_PD1_wilcox.RData')
#
p.val<-as.data.frame(p.val)
P_0.1<-subset(p.val,all<0.1)
#write.table(P_0.05,file='f_P_0.05.txt',quote=FALSE, sep='\t')

feat_1<-feat.all

##Confounder analysis

# ##############################################################################
# preprocess confounder variables to test later
colnames(meta)
# ##############################################################################
#  variance explained by ICB_response status
ss.response <- apply(feat.all, 1, FUN=function(x, label){
  rank.x <- rank(x)/length(x)
  ss.tot <- sum((rank.x - mean(rank.x))^2)/length(rank.x)
  ss.o.i <- sum(vapply(unique(label), function(l){
    sum((rank.x[label==l] - mean(rank.x[label==l]))^2)
  }, FUN.VALUE = double(1)))/length(rank.x)
  return(1-ss.o.i/ss.tot)
}, label=meta %>% pull(IrAE))

# calculate trimmed mean abundance
t.mean <- apply(feat.all, 1, mean, trim=0.1)

df.plot.all <- tibble(
  species=rownames(feat.all),
  IrAE=ss.response,
  t.mean=t.mean,
  p.value=p.val[rownames(feat.all), 'all'],
  meta.significance=p.val[rownames(feat.all), 'all'] < 0.05)

# ##############################################################################
# Test all possible confounder variables
df.list <- list()
for (meta.var in c( 'Study','Tumor')){
  cat('###############################\n', meta.var, '\n')
  meta.c <- meta %>%
    filter(!is.na(eval(parse(text=meta.var))))
  
  cat('After filtering, the distribution of variables is:\n')
  print(table(meta.c$IrAE, meta.c %>% pull(meta.var)))
  print(table(meta.c$Study))
  feat.red <- feat.all[,meta.c$Sample_ID]
  
  cat('Calculating variance explained by meta-variable...\n')
  ss.var <- apply(feat.red, 1, FUN=function(x, label){
    rank.x <- rank(x)/length(x)
    ss.tot <- sum((rank.x - mean(rank.x))^2)/length(rank.x)
    ss.o.i <- sum(vapply(unique(label), function(l){
      sum((rank.x[label==l] - mean(rank.x[label==l]))^2)
    }, FUN.VALUE = double(1)))/length(rank.x)
    return(1 - ss.o.i/ss.tot)
  }, label=meta.c %>% pull(meta.var))
  df.plot.all[[meta.var]] <- ss.var
  
  cat('Calculating association with the meta-variable...\n')
  if (meta.c %>% pull(meta.var) %>% unique %>% length > 2){
    meta.significance <- apply(feat.red, 1, FUN=function(x, var){
      kruskal.test(x~as.factor(var))$p.value
    }, var=meta.c %>% pull(meta.var))
  } else {
    meta.significance <- apply(feat.red, 1, FUN=function(x, var){
      wilcox.test(x~as.factor(var))$p.value
    }, var=meta.c %>% pull(meta.var))
  }
  meta.significance <- p.adjust(meta.significance, method='fdr')
  df.plot.all[[paste0(meta.var, '.significance')]] <- meta.significance
  cat('\n')
}
# ##############################################################################
# plot

g <- df.plot.all %>%
  gather(key=type, value=meta, -species, -IrAE,
         -t.mean, -p.value, -meta.significance) %>%
  filter(!str_detect(type, '.significance')) %>%
  filter(complete.cases(.)) %>%
  mutate(facet=case_when(type=='Study' ~ 'Study',
                         type=='Tumor' ~ 'Tumor',
                         TRUE ~ type)) %>%
  ggplot(aes(x=IrAE, y=meta, size=t.mean+1e-08, col=meta.significance)) +
  geom_point(shape=19) +
  xlab('Variance explained by IrAE status') +
  ylab('Variance explained by metadata variable') +
  theme_bw() +
  facet_wrap(~facet, ncol=3) +
  theme(strip.background = element_blank(),
        panel.grid.minor = element_blank()) +
  scale_x_continuous(breaks = seq(from=0, to=0.32, by=0.1)) +
  scale_y_continuous(breaks=seq(from=0, to=0.6, by=0.1)) +
  scale_colour_manual(values = alpha(c('black', '#CC071E'),
                                     alpha=c(0.1, .75)),
                      name=paste0('Significance\n(', '0.05', ' FDR)')) +
  scale_size_area(name='Trimmed mean\nabundance',
                  breaks=c(1e-05, 1e-03, 1e-02)) +
  guides( size = "legend", colour='legend')
g

### plot only study
df.plot.study <- df.plot.all %>%
  gather(key=type, value=meta, -species, -IrAE,
         -t.mean, -p.value, -meta.significance) %>%
  filter(!str_detect(type, '.significance')) %>%
  filter(complete.cases(.)) %>%
  filter(type=='Study')

g2 <- df.plot.study %>%
  ggplot(aes(x=IrAE, y=meta)) +
  geom_point(aes(size=t.mean, fill=meta.significance), shape=21,
             col=alpha(c('black'), alpha=0.4)) +
  xlab(paste0('Variance explained by IrAE\n',' average: ',
              formatC(mean(df.plot.study$IrAE)*100, digits=2), '%')) +
  ylab(paste0('Variance explained by Study\n',' average: ',
              formatC(mean(df.plot.study$meta)*100, digits=2), '%')) +
  theme_bw() +
  theme(panel.grid.minor = element_blank()) +
  scale_x_continuous(breaks = seq(from=0, to=0.1, by=0.05)) +
  scale_y_continuous(breaks=seq(from=0, to=0.6, by=0.1)) +
  scale_fill_manual(values = alpha(c('grey', '#CC071E'),
                                   alpha=c(0.4, .8)),
                    name=paste0('Significance\n(', '0.05', ' FDR)')) +
  scale_size_area(name='Trimmed mean abundance',
                  breaks=c(1e-05, 1e-03, 1e-02)) +
  guides(size = "legend", colour='legend')
g2
meta$Tumor<-as.factor(meta$Tumor)
df.plot.tumor <- df.plot.all %>%
  gather(key=type, value=meta, -species, -IrAE,
         -t.mean, -p.value, -meta.significance) %>%
  filter(!str_detect(type, '.significance')) %>%
  filter(complete.cases(.)) %>%
  filter(type=='Tumor')

g2 <- df.plot.tumor %>%
  ggplot(aes(x=IrAE, y=meta)) +
  geom_point(aes(size=t.mean, fill=meta.significance), shape=21,
             col=alpha(c('black'), alpha=0.4)) +
  xlab(paste0('Variance explained by IrAE\n',' average: ',
              formatC(mean(df.plot.tumor$IrAE)*100, digits=2), '%')) +
  ylab(paste0('Variance explained by Tumor\n',' average: ',
              formatC(mean(df.plot.tumor$meta)*100, digits=2), '%')) +
  theme_bw() +
  theme(panel.grid.minor = element_blank()) +
  scale_x_continuous(breaks = seq(from=0, to=0.1, by=0.05)) +
  scale_y_continuous(breaks=seq(from=0, to=0.6, by=0.1)) +
  scale_fill_manual(values = alpha(c('grey', '#CC071E'),
                                   alpha=c(0.4, .8)),
                    name=paste0('Significance\n(', '0.05', ' FDR)')) +
  scale_size_area(name='Trimmed mean abundance',
                  breaks=c(1e-05, 1e-03, 1e-02)) +
  guides( size = "legend", colour='legend')
g2
