### process data ----

data <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/master_data.csv")

# -log10 scaling
data[, 11:114] <- -log10(data[, 11:114] + 1e-10)

bl <- data[data$EXAMDATE_RANK==0,]
blAD <- bl[bl$DX_VALS==4,]

### corr mat ----

bile_acids <- colnames(blAD)[11:114]
lab_values <- colnames(blAD)[118:122]
cor_matrix <- matrix(0, nrow = length(bile_acids), ncol = length(lab_values))
rownames(cor_matrix) <- bile_acids
colnames(cor_matrix) <- lab_values

covariates <- c("BMI", "fast")
for (bile in bile_acids) {
  for (lab in lab_values) {
    complete_data <- na.omit(blAD[, c(bile, lab, covariates)])
    
    # result <- try(pcor.test(complete_data[[bile]],
    #                         complete_data[[lab]],
    #                         complete_data[, covariates]),
    #               silent = TRUE)
    result <- cor.test(complete_data[[bile]], complete_data[[lab]], method = "spearman")
    cor_matrix[bile, lab] <- result$estimate
  }
}
cor_df <- melt(cor_matrix)
colnames(cor_df) <- c("BileAcid", "LabValue", "pValue")

# add in class data
class <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/raw/class.csv")
all <- merge(cor_df, class, by.x="BileAcid", by.y="bile_acids")
all$class[all$class == "alpha_amino_acid"] <- "amino_acid"
all$class[all$class == "beta_amino_acid"] <- "amino_acid"
all$class[all$class %in% c("dicarboxylic_acid",
                           "tricarboxylic_acid", "hydroxyl_acid",
                           "lactam", "auxin", "keto_acid",
                           "indole_derivative", "aryl_sulfate",
                           "alpha_hydroxyl_acid","carboxylic_acid")] <- "other"

### visualization ----

all_other <- all[all$class %in% c("fatty_acid", "amino_acid", "other"),]
all_ba <- all[all$class %in% c("primary", "primary_conjugated",
                               "secondary", "secondary_conjugated"),]
ggplot(all, aes(x = LabValue, y = BileAcid, fill = pValue)) +
  geom_tile() +
  theme_bw() +
  scale_fill_gradientn(
    colors = c("royalblue4", "white", "brown4"), 
    values = scales::rescale(c(-1, 0, 1)),
    limits = c(-1, 1)
  ) +
  labs(fill = "Correlation", y="", x="") +
  theme(panel.grid = element_blank(),
    axis.ticks.x = element_blank(),
    axis.ticks.y = element_blank()) +
  facet_grid(class ~ ., scales = "free_y", space = "free_y")

