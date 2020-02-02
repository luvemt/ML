
require(DESeq2)
require(ggplot2)

#load transcript matrix
transcript_matrix = read.csv(file = "transcript_count_matrix3.csv")
transcript_name = transcript_matrix[,1]
transcript_matrix = transcript_matrix[,-1]

#load experimental assay
sample_tb = read.table("drought-samples2.txt", sep="\t", header = TRUE)

sample_index = colnames(transcript_matrix)
sample_index = gsub("mcotton","",sample_index)
sample_index = as.numeric(gsub(".gtf","",sample_index))
sample_order = order(sample_index, decreasing = FALSE)

transcript_matrix = transcript_matrix[,sample_order]
sample_tb = sample_tb[order(sample_tb$Sample),]

# create cotton cts
cottoncts = transcript_matrix
rownames(cottoncts) = as.character(transcript_name)
rm(transcript_matrix)

# setup sample info as coldata
coldata = sample_tb
coldata$Total = colSums(cottoncts)
coldata$Replicate = as.character(coldata$Replicate)

# assign references
coldata$Timepoint = relevel(coldata$Timepoint, "12 hrs")
coldata$Phenotype = relevel(coldata$Phenotype, "Drought sensitive")

rownames(coldata) = colnames(cottoncts)

cottonctsdds = DESeqDataSetFromMatrix(countData = cottoncts,
                                      colData = coldata,
                                      design = ~1)
# setup gene names as featuredata
featuredata = data.frame(rownames(cottoncts))
mcols(cottonctsdds) = DataFrame(mcols(cottonctsdds), featuredata)

# filter genes with aggregate count >= 10
cottonctsdds = cottonctsdds[rowSums(counts(cottonctsdds))>=10,]


# complex model b
cmodel <- ~ Timepoint + Phenotype + Timepoint:Phenotype
cottonctscmodel <- cottonctsdds
design(cottonctscmodel) <- cmodel
cottonctscmodel <- DESeq(cottonctscmodel)

# contrast interaction term
head(coefficients(cottonctscmodel))


# timepoint contrast

# 12 hrs vs 0 hrs

res_time1 <- results(cottonctscmodel,
                          contrast = c("Timepoint","0 hrs","12 hrs"))
png("ma_plot_time12vs0.png", width = 720, height = 720)
plotMA(object = res_time1)
dev.off()

res_time1 = res_time1[which(res_time1$padj<0.05),]
order_degs_time1 <- res_time1[order(res_time1$padj),]

head(order_degs_time1)

nrow(order_degs_time1)


# 12 hrs vs 48 hrs

res_time2 <- results(cottonctscmodel,
                     contrast = c("Timepoint","48 hrs","12 hrs"))
png("ma_plot_time12vs48.png", width = 720, height = 720)
plotMA(object = res_time2)
dev.off()

res_time2 = res_time2[which(res_time2$padj<0.05),]
order_degs_time2 <- res_time2[order(res_time2$padj),]

head(order_degs_time2)

nrow(order_degs_time2)


common_genes = intersect(rownames(order_degs_time1), rownames(order_degs_time2))

for(i in 1:10){
  png(paste0("top10genes_time_intersection",i,".png"),width = 720, height = 720)
  degexpr_time1 <- plotCounts(cottonctscmodel, 
                              gene=common_genes[i], 
                              intgroup="Timepoint", returnData=TRUE)
  
  degexpr_time1$Timepoint = ordered(degexpr_time1$Timepoint, 
                                    levels=c("0 hrs", "12 hrs", "48 hrs"))
  degexpr_time1$Phenotype = cottonctscmodel$Phenotype
  p1 = match(common_genes[i], rownames(order_degs_time1))
  p2 = match(common_genes[i], rownames(order_degs_time2))
  p = ggplot(degexpr_time1, aes(x=Timepoint, y=count, color=Phenotype, 
                                shape=Timepoint)) +
    geom_point(size=4, position=position_jitter(w=0.1, h=0)) +
    scale_y_log10() + ggtitle(paste("Timepoint:",
                                    common_genes[i],
                                    "\nRank=",i,"\nP_12vs0=",
                                    format(order_degs_time1[p1,6],
                                           digits = 2), "\nP_12vs48=",
                                    format(order_degs_time2[p2,6])))
  print(p)
  dev.off()
}



# phenotype contrast
res_phenotype <- results(cottonctscmodel,
                          contrast = c("Phenotype","Drought tolerant","Drought sensitive"))
png("ma_plot_phenotype_timepoint_pheno_interaction_model.png", width = 720, height = 720)
plotMA(object = res_phenotype)
dev.off()

order_degs_phenotype <- order(res_phenotype$padj)
print(sum(res_phenotype$padj<0.05, na.rm = TRUE))
head(res_phenotype[order_degs_phenotype,])

####################################### Do this ###############################3
# Plot top 10 genes
###############################################################################


# Interaction Contrast 
# 12 hrs Drought Sens. vs 0 hrs Drought Tol.

res_int1 = results(cottonctscmodel,
                   contrast = list("Timepoint0.hrs.PhenotypeDrought.tolerant"))

####### Do this ######
png("ma_plot_time12vs0.png", width = 720, height = 720)
plotMA(object = res_int1)
dev.off()
######################


res_int1 = res_int1[which(res_int1$padj<0.01),]
order_degs_int1 <- res_int1[order(res_int1$padj),]

head(order_degs_int1)

nrow(order_degs_int1)


# 12 hrs Drought Sens. vs 48 hrs Drought Tol.

res_int2 = results(cottonctscmodel,
                   contrast = list("Timepoint48.hrs.PhenotypeDrought.tolerant"))
####### Do this ######
png("ma_plot_time12vs0.png", width = 720, height = 720)
plotMA(object = res_int1)
dev.off()
######################

res_int2 = res_int2[which(res_int2$padj<0.01),]

order_degs_int2 <- res_int2[order(res_int2$padj),]

head(order_degs_int2)

nrow(order_degs_int2)

common_genes = intersect(rownames(order_degs_int1), rownames(order_degs_int2))

for(i in 1:10){
  png(paste0("top10genes_time_pheno_interaction",i,".png"),width = 720, height = 720)
  degexpr_int <- plotCounts(cottonctscmodel, 
                            gene=common_genes[i], 
                            intgroup="Timepoint", returnData=TRUE)
  
  degexpr_int$Timepoint = ordered(degexpr_int$Timepoint, 
                                  levels=c("0 hrs", "12 hrs", "48 hrs"))
  degexpr_int$Phenotype = cottonctscmodel$Phenotype
  
  p1 = match(common_genes[i], rownames(order_degs_int1))
  p2 = match(common_genes[i], rownames(order_degs_int2))
  
  p = ggplot(degexpr_int, aes(x=Timepoint, y=count, color=Phenotype)) +
    geom_point(size=4, position=position_jitter(w=0.1, h=0)) +
    scale_y_log10() + ggtitle(paste("Timepoint : Phenotype",
                                    common_genes[i],
                                    "\nSignificance=",i,"\nPval_12Drought Sens_vs_0Drought Tol=",
                                    format(order_degs_int1[p1,6],
                                           digits = 2), "\nPval_12Drought Sen_vs_48Drought Tol=",
                                    format(order_degs_int2[p2,6])))
  print(p)
  dev.off()
}


##################### Q3 ##########################
################### part 3 ########################

# remove phenotype effect
dds = cottonctscmodel
ald = DESeqTransform(dds)
assay(ald)= log2(1 + counts(dds, normalized=TRUE))


head(coef(cottonctscmodel),2)
assay(ald)["Gohir.D05G027600.1.v1.1",]

dds$Timepoint


matrix.no.time <- sapply(seq(ncol(ald)), function(j) {
  if(dds$Timepoint[j] == "0 hrs") {
    assay(ald)[, j] - coef(dds)[, "Timepoint_0.hrs_vs_12.hrs"]
  }
  
  else if(dds$Timepoint[j] == "48 hrs"){
    assay(ald)[, j] - coef(dds)[, "Timepoint_48.hrs_vs_12.hrs"]
  }
  
  else {
    assay(ald)[, j]
  } } )

#ald.no.time  <- DESeqTransform(dds)

#assay(ald.no.time) <- matrix.no.time




#cottonctstimephenorem  <- DESeqTransform(ald.no.time)


matrix.no.phen <- sapply(seq(ncol(ald)), function(j) {
  if(dds$Phenotype[j] == "Drought tolerant") {
   assay(ald)[, j] - coef(dds)[, "Phenotype_Drought.tolerant_vs_Drought.sensitive"]
  } else {
 assay(ald)[, j]
  } } )

ald.no.time.no.phen  <- DESeqTransform(dds)

assay(ald.no.time.no.phen) <- matrix.no.phen



cottonmatrix_remphenotype <- sapply(seq(ncol(cottonctscmodel)), function(j) {
  # log data should be used
  log2(assay(cottonctscmodel)[, j]+1) - 
    coef(cottonctscmodel)[, "Phenotype_Drought.tolerant_vs_Drought.sensitive"]
  })

# remove timepoint effect
cottonmatrix_remtime <- sapply(seq(ncol(cottonmatrix_remphenotype)), function(j) {
  cottonmatrix_remphenotype[, j] - 
    (coef(cottonctscmodel)[, "Timepoint_0.hrs_vs_12.hrs"] + 
    coef(cottonctscmodel)[, "Timepoint_48.hrs_vs_12.hrs"])
})

# PCA after timepoint and phenotype effect have been removed

cottonctstimephenorem  <- DESeqTransform(cottonctscmodel)
assay(cottonctstimephenorem) <- cottonmatrix_remtime
png("cotton_cts_after_time_pheno_removal.png", width = 720, height = 720)
data <- plotPCA(cottonctstimephenorem, 
                ntop=nrow(cottonctstimephenorem), 
                intgroup=c("Timepoint", "Phenotype"), returnData=TRUE)
percentVar <- round(100 * attr(data, "percentVar"))
ggplot(data, aes(PC1, PC2, color=Phenotype, size=5)) +
  geom_point() + 
  xlab(paste0("PC1: ", percentVar[1], "% variance")) +
  ylab(paste0("PC2: ", percentVar[2], "% variance")) +
  ggtitle("Normalized log transformed data after removing timepoint effect")
dev.off()

####################### Q3 part 4 ############################################

res_int1 = results(cottonctscmodel,
                   contrast = list("Timepoint0.hrs.PhenotypeDrought.tolerant"))
res_int1 = res_int1[which(res_int1$padj<0.01),]
order_degs_int1 <- res_int1[order(res_int1$padj),]


res_int2 = results(cottonctscmodel,
                   contrast = list("Timepoint48.hrs.PhenotypeDrought.tolerant"))
res_int2 = res_int2[which(res_int2$padj<0.01),]
order_degs_int2 <- res_int2[order(res_int2$padj),]

common_genes = intersect(rownames(order_degs_int1), rownames(order_degs_int2))

for(i in 1:10){
  #png(paste0("top10_time_pheno_removed_rank_",i,".png"),width = 720, height = 720)
  g = match(common_genes[i], rownames(cottonctscmodel))
  degexpr_intremoved <- as.data.frame(assay(ald.no.time.no.phen)[g,])
  
  degexpr_intremoved$Timepoint = ordered(cottonctscmodel$Timepoint, 
                                         levels=c("0 hrs", "12 hrs", "48 hrs"))
  degexpr_intremoved$Phenotype = cottonctscmodel$Phenotype
  colnames(degexpr_intremoved)[1] = "count"
  
  p1 = match(common_genes[i], rownames(order_degs_int1))
  p2 = match(common_genes[i], rownames(order_degs_int2))
  
  p = ggplot(degexpr_intremoved, aes(x=Timepoint, y=count, color=Phenotype, 
                                     shape=Timepoint)) +
    geom_point(size=4, position=position_jitter(w=0.1, h=0)) +
    ggtitle(paste("Timepoint:Phenotype",
                  common_genes[i],
                  "\nSignificance=",i,"\nPval_12Drought Sens_vs_0Drought Tol=",
                  format(order_degs_int1[p1,6],
                         digits = 2), "\nPval_12Drought Sen_vs_48Drought Tol=",
                  format(order_degs_int2[p2,6])))
  print(p)
  #dev.off()
}