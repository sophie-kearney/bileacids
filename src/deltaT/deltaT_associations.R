library(tidyverse)

data <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/blMCI-6dt.csv")

# set constants
lab_values <- c("ABeta42", "pTau", "TAU", "FDG", "ABETA", "WholeBrain", "AV45")
covariates <- c("BMI", "PTGENDER", "APOE_e2e4", "AGE", "fast")
begin_BA_cols <- which(colnames(data) == "CA")
end_BA_cols <- which(colnames(data)=="BUDCA")
begin_BA_ratio <- which(colnames(data) == "CA_CDCA")
end_BA_ratio <- which(colnames(data) == "GLCA_CDCA")

bile_acids <- colnames(data)[c(begin_BA_cols:end_BA_cols, begin_BA_ratio:end_BA_ratio)]

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
    complete_data <- na.omit(data[, c(bile, lab, covariates)])
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

heatmaq_matrix <- -log10(q_matrix) * sign(beta_matrix)
heatmaq_matrix[q_matrix > 0.05] <- 0
q_df <- melt(heatmaq_matrix)
colnames(q_df) <- c("BileAcid", "LabValue", "qValue")

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
       x="", y="", title="Baseline pMCI and 24 months after AD Onset") +
  theme(panel.grid = element_blank(),
        plot.title = element_text(hjust = 0.5)) +
  facet_grid(class ~ ., scales = "free_y", space = "free_y")


