library(tidyverse)
library(ppcor)
library(reshape2)
library(gridExtra)

### process data -------

data <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/master_data.csv")

# -log10 scaling
data[, 11:114] <- log10(data[, 11:114] + 1e-10)

# min-max
minmax_scale <- function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
}
data[, 11:114] <- apply(data[, 11:114], 2, minmax_scale)

#### correlation heatmap ---------

bl <- data[data$EXAMDATE_RANK==0,]
blAD <- data[data$DX_VALS==4,]

bile_acids <- colnames(blAD)[11:114]
lab_values <- colnames(blAD)[118:122]
cor_matrix <- matrix(0, nrow = length(bile_acids), ncol = length(lab_values))
rownames(cor_matrix) <- bile_acids
colnames(cor_matrix) <- lab_values

covariates <- c("BMI", "fast")
for (bile in bile_acids) {
  for (lab in lab_values) {
    complete_data <- na.omit(blAD[, c(bile, lab, covariates)])
    
    result <- try(pcor.test(complete_data[[bile]],
                            complete_data[[lab]],
                            complete_data[, covariates]),
                  silent = TRUE)
    cor_matrix[bile, lab] <- result$estimate
  }
}

print(cor_matrix) 

cor_df <- melt(cor_matrix)
colnames(cor_df) <- c("BileAcid", "LabValue", "pValue")

class <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/raw/class.csv")

all <- merge(cor_df, class, by.x="BileAcid", by.y="bile_acids")

### ALL VALUES
ggplot(all, aes(x = LabValue, y = BileAcid, fill = pValue)) +
  geom_tile() +
  # scale_fill_gradient(low = "blue", high = "white") +
  theme_minimal() +
  scale_fill_gradientn(
    colors = c("royalblue4", "white", "brown4"), 
    values = scales::rescale(c(-1, 0, 1)),  # Ensure -1, 0, and 1 are the key values for color scaling
    limits = c(-1, 1)  # Set the color scale limits between -1 and 1
  ) +
  labs(fill = "Correlation", y="", x="") +
  theme(panel.grid = element_blank())


##### logistic effect size heatmap  ---------

hcad <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/HCAD_cov.csv")
bl <- hcad[hcad$EXAMDATE_RANK==0,]
blAD <- bl[bl$DX_VALS==4,]
blAD$TAU <- as.numeric(blAD$TAU)

bile_acids <- colnames(blAD)[c(19:122, 131:137)]
lab_values <- colnames(blAD)[c(129, 130, 5, 9)]
p_matrix <- matrix(1, nrow = length(bile_acids), ncol = length(lab_values))
beta_matrix <- matrix(0, nrow = length(bile_acids), ncol = length(lab_values))

rownames(p_matrix) <- bile_acids
colnames(p_matrix) <- lab_values

rownames(beta_matrix) <- bile_acids
colnames(beta_matrix) <- lab_values

covariates <- c("BMI", "PTGENDER", "APOE_e2e4", "AGE", "fast")

for (bile in bile_acids) {
  for (lab in lab_values) {
    complete_data <- na.omit(blAD[, c(bile, lab, covariates)])
    
    model <- lm(as.formula(paste(lab, "~", bile, "+", paste(covariates, collapse = " + "))),
                data = complete_data)
    
    p_matrix[bile, lab] <- summary(model)$coefficients[bile, "Pr(>|t|)"]
    beta_matrix[bile, lab] <- summary(model)$coefficients[bile, "Estimate"]
  }
}

q_matrix <- matrix(p.adjust(p_matrix, method = "BH"), 
                   nrow = length(bile_acids), 
                   dimnames = list(bile_acids, lab_values))


# heatmap_matrix <- -log10(q_matrix) * sign(beta_matrix)
# q_df <- melt(heatmap_matrix)
# colnames(q_df) <- c("BileAcid", "LabValue", "qValue")
# ggplot(q_df, aes(x = LabValue, y = BileAcid, fill = qValue)) +
#   geom_tile() +
#   theme_minimal() +
#   scale_fill_gradientn(
#     colors = c("royalblue4", "white", "brown4"),
#     values = scales::rescale(c(min(heatmap_matrix), 0, max(heatmap_matrix))),
#     limits = c(min(heatmap_matrix), max(heatmap_matrix))) +
#   labs(fill = "qValue") +
#   theme(panel.grid = element_blank())

# q_df <- melt(q_matrix)
# colnames(q_df) <- c("BileAcid", "LabValue", "qValue")
# ggplot(q_df, aes(x = LabValue, y = BileAcid, fill = qValue)) +
#   geom_tile() +
#   theme_minimal() +
#   scale_fill_gradientn(
#     colors = c("royalblue4","white", "white"),
#     values = scales::rescale(c(0, 0.05, 1)),
#     limits = c(0, 1)) +
#   labs(fill = "qValue") +
#   theme(panel.grid = element_blank())

heatmap_matrix <- -log10(q_matrix) * sign(beta_matrix)
heatmap_matrix[q_matrix > 0.05] <- 0
q_df <- melt(heatmap_matrix)
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

# ggplot(all, aes(x = LabValue, y = BileAcid, fill = qValue)) +
#   geom_tile() +
#   theme_minimal() +
#   scale_fill_gradientn(
#     colors = c("#2b550e", "white", "#9c2324"),
#     values = scales::rescale(c(-6, 0, 6)),
#     limits = c(-6, 6)
#   ) +
#   labs(fill = "-log10(q) * sign(Beta)") +
#   theme(panel.grid = element_blank())

all_other <- all[all$class %in% c("fatty_acid", "amino_acid", "other"),]
all_ba <- all[all$class %in% c("primary", "primary_conjugated",
                               "secondary", "secondary_conjugated",
                               "ratio"),]
all_ba$class <- factor(all_ba$class, levels = c("primary", "secondary", 
                                                "primary_conjugated", "secondary_conjugated",
                                                "ratio"))

ggplot(all_other, aes(x = LabValue, y = BileAcid, fill = qValue)) +
  geom_tile() +
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

### plot values ------


bile_acids <- colnames(blAD)[c(100:110)]
lab_values <- colnames(blAD)[c(129, 130, 5, 9)]

plot_list <- list()
for (bile in bile_acids) {
  for (lab in lab_values) {
    complete_data <- na.omit(blAD[, c(bile, lab, covariates)])
    
    p <- ggplot(blAD, aes_string(x = lab, y = bile)) + 
      geom_point() + 
      theme_minimal()
    plot_list <- append(plot_list, list(p))
  }
}

do.call(grid.arrange, c(plot_list, ncol = 4))

