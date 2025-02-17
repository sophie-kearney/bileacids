### ----
# IMPORTS
### ----

library(tidyverse)
library(ppcor)
library(reshape2)
library(gridExtra)

### ----
# DATA WRANGLING
### -----

# data <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/master_data.csv")
# data$TAU <- as.numeric(data$TAU)
# data$ABETA <- as.numeric(data$ABETA)

# data <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/master_data_minmax.csv")
data <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/master_data_log10.csv")
# data <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/master_data_standardscaler.csv")

# set constants
lab_values <- c("ABeta42", "pTau", "TAU", "FDG", "ABETA", "WholeBrain", "AV45")
covariates <- c("BMI", "PTGENDER", "APOE_e2e4", "AGE", "fast")
begin_BA_cols <- which(colnames(data) == "CA")
end_BA_cols <- which(colnames(data)=="BUDCA")
begin_BA_ratio <- which(colnames(data) == "CA_CDCA")
end_BA_ratio <- which(colnames(data) == "GLCA_CDCA")

### -------
# Do BA move with known markers of AD pathology?
### -------

bl <- data[data$EXAMDATE_RANK==0,]
blHCAD <- bl
# blHCAD <- bl[bl$DX_VALS %in% c(1,4),]

bile_acids <- colnames(blHCAD)[c(begin_BA_cols:end_BA_cols, begin_BA_ratio:end_BA_ratio)]

# create empty matrices
q_matrix <- matrix(1, nrow = length(bile_acids), ncol = length(lab_values))
beta_matrix <- matrix(0, nrow = length(bile_acids), ncol = length(lab_values))
rownames(q_matrix) <- bile_acids
colnames(q_matrix) <- lab_values
rownames(beta_matrix) <- bile_acids
colnames(beta_matrix) <- lab_values

# run lm for every pair of bile acid and lab value
for (bile in bile_acids) {
  for (lab in lab_values) {
    # remove any NA
    complete_data <- na.omit(blHCAD[, c(bile, lab, covariates)])
    # remove any infinite
    complete_data <- complete_data[!apply(complete_data, 1, function(row) any(is.infinite(row))), ]

    model <- lm(as.formula(paste(lab, "~", bile, "+", paste(covariates, collapse = " + "))),
                data = complete_data)

    q_matrix[bile, lab] <- summary(model)$coefficients[bile, "Pr(>|t|)"]
    beta_matrix[bile, lab] <- summary(model)$coefficients[bile, "Estimate"]
  }
}

# FDR correction with benjamini-hochberg
q_matrix <- matrix(p.adjust(q_matrix, method = "BH"),
                   nrow = length(bile_acids),
                   dimnames = list(bile_acids, lab_values))

# -log10(qvalue) * sign(beta)
heatmaq_matrix <- -log10(q_matrix) * sign(beta_matrix)
heatmaq_matrix[q_matrix > 0.05] <- 0
q_df <- melt(heatmaq_matrix)
colnames(q_df) <- c("BileAcid", "LabValue", "qValue")

# merge in all classes
class <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/raw/class.csv")
all <- merge(q_df, class, by.x="BileAcid", by.y="bile_acids")
all$class[all$class == "alpha_amino_acid"] <- "amino_acid"
all$class[all$class == "beta_amino_acid"] <- "amino_acid"
all$class[all$class %in% c("dicarboxylic_acid",
                           "tricarboxylic_acid", "hydroxyl_acid",
                           "lactam", "auxin", "keto_acid",
                           "indole_derivative", "aryl_sulfate",
                           "alpha_hydroxyl_acid","carboxylic_acid")] <- "other"

# set order of facets with factor
all$class <- factor(all$class, levels = c("primary", "secondary",
                                              "primary_conjugated", "secondary_conjugated",
                                              "ratio"),
                    labels=c("Primary", "Secondary","Primary Conj", "Secondary Conj", "Ratio"))
all$LabValue <- factor(all$LabValue, levels=c("ABeta42", "ABETA", "pTau","TAU","FDG","WholeBrain", "AV45"))
ggplot(all, aes(x = LabValue, y = BileAcid, fill = qValue)) +
  geom_tile(color="gray") +
  theme_bw() +
  scale_fill_gradientn(
    colors = c("#2b550e", "white", "#9c2324"),
    values = scales::rescale(c(-6, 0, 6)),
    limits = c(-6, 6), ,
    guide = guide_colorbar(reverse = TRUE)
  ) +
  labs(fill = "-log10(qValue) * sign(Beta)",
       x="", y="") +
  theme(panel.grid = element_blank()) +
  facet_grid(class ~ ., scales = "free_y", space = "free_y")


### -------
# Do BA move with known markers of AD pathology?
### -------

all_metabolites <- colnames(data)[19:122]

cor_matrix <- matrix(NA, nrow = length(all_metabolites), ncol = length(all_metabolites),
                   dimnames = list(all_metabolites, all_metabolites))
p_matrix <- matrix(NA, nrow = length(all_metabolites), ncol = length(all_metabolites),
                   dimnames = list(all_metabolites, all_metabolites))

for (m1 in all_metabolites) {
  for (m2 in all_metabolites) {
    if (m1 != m2) {
      complete_data <- na.omit(data[, c(m1, m2, covariates)])
      pcor_result <- pcor.test(complete_data[[m1]], complete_data[[m2]], complete_data[, covariates])

      cor_matrix[m1, m2] <- pcor_result$estimate
      p_matrix[m1, m2] <- pcor_result$p.value
    }
  }
}

q_matrix <- matrix(p.adjust(p_matrix, method = "BH"), 
                   nrow = length(all_metabolites), 
                   dimnames = list(all_metabolites, all_metabolites))

cor_matrix[q_matrix > 0.05] <- 0
cor_df <- melt(cor_matrix)
colnames(cor_df) <- c("Metabolite1", "Metabolite2", "Correlation")

# import and aggregate classes
class <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/raw/class.csv")
class$class[class$class %in% c("primary", "secondary", "primary_conjugated",
                               "secondary_conjugated", "ratio")] <- "bile_acid"
class$class[class$class == "alpha_amino_acid"] <- "amino_acid"
class$class[class$class == "beta_amino_acid"] <- "amino_acid"
class$class[class$class %in% c("dicarboxylic_acid",
                           "tricarboxylic_acid", "hydroxyl_acid",
                           "lactam", "auxin", "keto_acid",
                           "indole_derivative", "aryl_sulfate",
                           "alpha_hydroxyl_acid","carboxylic_acid")] <- "other"


all <- merge(cor_df, class, by.x="Metabolite1", by.y="bile_acids")
colnames(all)[4] <- "metabolite1_class"
all <- merge(all, class, by.x="Metabolite2", by.y="bile_acids")
colnames(all)[5] <- "metabolite2_class"

ggplot(all, aes(x = Metabolite1, y = Metabolite2, fill = Correlation)) +
  geom_tile() +
  theme_minimal() +
  scale_fill_gradientn(
    colors = c("royalblue4", "white", "brown4"), 
    values = scales::rescale(c(-1, 0, 1)),
    limits = c(-1, 1)) +
  labs(fill = "Correlation", y="", x="") +
  theme(panel.grid = element_blank())
