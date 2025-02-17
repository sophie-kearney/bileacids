library(tidyverse)
library(ppcor)
library(reshape2)
library(gridExtra)

data <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/master_data.csv")

bl <- data[data$VISCODE2=="bl",]
bl_clean <- na.omit(bl[, c("ABeta42", "L_HISTIDINE")])

###---------------
## test correlation metrics
### test normality of values

shapiro.test(bl_clean$ABeta42)
shapiro.test(bl_clean$L_HISTIDINE)

# conc - data is not normal

### spearman correlation

cor.test(bl_clean$ABeta42, bl_clean$L_HISTIDINE, method = "spearman")

#  conc - very weak positive correlation

### partial correlation
### uses pearson - may not be helpful

bl_clean <- na.omit(bl[, c("ABeta42", "L_HISTIDINE", "BMI", "fast")])
result <- pcor.test(bl_clean$ABeta42, bl_clean$L_HISTIDINE, bl_clean[, c("BMI", "fast")])
print(result)

# conc - no significant correlation
#     estimate    p.value statistic   n gp  Method
# 1 0.05847701 0.09879285  1.652668 800  2 pearson


### ------------------
# get sample numbers

# prevalent AD
data %>%
  group_by(RID) %>%
  summarise(all_four = all(DX_VALS == 4)) %>%
  summarise(count = sum(all_four))

# healthy
data %>%
  group_by(RID) %>%
  summarise(all_1 = all(DX_VALS == 1)) %>%
  summarise(count = sum(all_1))

# MCI
data %>%
  group_by(RID) %>%
  summarise(all_2_or_3 = all(DX_VALS %in% c(2, 3))) %>%
  summarise(count = sum(all_2_or_3))


# healthy to MCI
data$DX_VALS[data$DX_VALS == 3] <- 2

rids <- c()
for (rid in unique(data$RID)){
  patient <- data[data$RID == rid,]
  dxs <- patient$DX_VALS
  
  if (any(dxs == 1) && any(dxs[which(dxs == 1)[1]:length(dxs)] == 2)) {
    if (dxs[1] != 2 && dxs[length(dxs)] != 1 && !any(dxs == 4)){
      rids <- c(rids, rid)
    }
  }
}

rids <- rids[rids != 420]
rids <- rids[rids != 4388]
length(rids)


# MCI to AD

rids <- c()
for (rid in unique(data$RID)){
  patient <- data[data$RID == rid,]
  dxs <- patient$DX_VALS
  
  if (any(dxs == 2) && any(dxs[which(dxs == 2)[1]:length(dxs)] == 4)) {
    if (dxs[1] != 4 && dxs[length(dxs)] != 2 && !any(dxs == 1)){
      rids <- c(rids, rid)
    }
  }
}

length(rids)
t <- data[data$RID %in% rids[0:50],]
rids <- rids[rids != 566]
rids <- rids[rids != 2274]
rids <- rids[rids != 4293]
rids <- rids[rids != 4430]
length(rids)

ggplot(t, aes(x = EXAMDATE_RANK, y = DX_VALS, group = RID)) +
  geom_line() +
  facet_wrap(~ RID, scales = "free_y") +
  theme_minimal() +
  labs(x = "Exam Date Rank", y = "DX Value", title = "DX Values over Time by Patient (RID)")

length(unique(data$RID))

# Healthy -> AD

rids2 <- c()
for (rid in unique(data$RID)){
  patient <- data[data$RID == rid,]
  dxs <- patient$DX_VALS
  
  if (any(dxs == 1) && any(dxs[which(dxs == 1)[1]:length(dxs)] == 4)) {
    if (dxs[1] != 4 && dxs[1] != 2 && dxs[length(dxs)] != 1 && dxs[length(dxs)] != 2){
      rids2 <- c(rids2, rid)
    }
  }
}
length(rids2)
t2 <- data[data$RID %in% rids2,]
ggplot(t, aes(x = EXAMDATE_RANK, y = DX_VALS, group = RID)) +
  geom_line() +
  facet_wrap(~ RID, scales = "free_y") +
  theme_minimal() +
  labs(x = "Exam Date Rank", y = "DX Value", title = "DX Values over Time by Patient (RID)")


### -------------
# correlation matrix all values
#####

bile_acids <- colnames(data)[11:114]
lab_values <- colnames(data)[118:122]
cor_matrix <- matrix(0, nrow = length(bile_acids), ncol = length(lab_values))
rownames(cor_matrix) <- bile_acids
colnames(cor_matrix) <- lab_values

### SPEARMAN
# for (bile in bile_acids) {
#   for (lab in lab_values) {
#     complete_data <- na.omit(data[, c(bile, lab)])
#   
#     result <- cor.test(complete_data[[bile]], complete_data[[lab]], method = "spearman")
#     cor_matrix[bile, lab] <- result$estimate
#   }
# }

### PEARSON PARTIAL
covariates <- c("BMI", "fast")
for (bile in bile_acids) {
  for (lab in lab_values) {
    complete_data <- na.omit(data[, c(bile, lab, covariates)])

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
  theme(
        panel.grid = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.line.x = element_blank())

### corr matrix BL AD

bl <- data[data$EXAMDATE_RANK==0,]
blAD <- bl[bl$DX_VALS==4,]

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
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.line.x = element_blank())

###### ------------
# get counts
######
sec <- all[all$X == "secondary",]
ggplot(sec, aes(x = LabValue, y = BileAcid, fill = pValue)) +
  geom_tile() +
  # scale_fill_gradientn(
  #   colors = c("royalblue4", "white", "brown4"), 
  #   values = scales::rescale(c(-1,0, 1))) +
  theme_minimal() +
  scale_fill_gradientn(
    colors = c("royalblue4", "white", "brown4"), 
    values = scales::rescale(c(-1, 0, 1)),  # Ensure -1, 0, and 1 are the key values for color scaling
    limits = c(-1, 1)  # Set the color scale limits between -1 and 1
  ) +
  labs(y="", x="") +
  theme(legend.position = "none", 
        panel.grid = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.line.x = element_blank())

secc <- all[all$X == "secondary_conjugated",]
ggplot(secc, aes(x = LabValue, y = BileAcid, fill = pValue)) +
  geom_tile() +
  # scale_fill_gradientn(
  #   colors = c("royalblue4", "white", "brown4"), 
  #   values = scales::rescale(c(-1,0, 1))) +
  theme_minimal() +
  scale_fill_gradientn(
    colors = c("royalblue4", "white", "brown4"), 
    values = scales::rescale(c(-1, 0, 1)),  # Ensure -1, 0, and 1 are the key values for color scaling
    limits = c(-1, 1)  # Set the color scale limits between -1 and 1
  ) +
  labs(y="", x="") +
  theme(legend.position = "none", ,
        panel.grid = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.line.x = element_blank())

pri <- all[all$X == "primary",]
ggplot(pri, aes(x = LabValue, y = BileAcid, fill = pValue)) +
  geom_tile() +
  # scale_fill_gradientn(
  #   colors = c("royalblue4", "white", "brown4"), 
  #   values = scales::rescale(c(-1,0, 1))) +
  scale_fill_gradientn(
    colors = c("royalblue4", "white", "brown4"), 
    values = scales::rescale(c(-1, 0, 1)),  # Ensure -1, 0, and 1 are the key values for color scaling
    limits = c(-1, 1)  # Set the color scale limits between -1 and 1
  ) +
  theme_minimal() +
  labs(y="", x="") +
  theme(legend.position = "none", ,
        panel.grid = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.line.x = element_blank())

pric <- all[all$X == "primary_conjugated",]
ggplot(pric, aes(x = LabValue, y = BileAcid, fill = pValue)) +
  geom_tile() +
  # scale_fill_gradientn(
  #   colors = c("royalblue4", "white", "brown4"), 
  #   values = scales::rescale(c(-1,0, 1))) +
  scale_fill_gradientn(
    colors = c("royalblue4", "white", "brown4"), 
    values = scales::rescale(c(-1, 0, 1)),  # Ensure -1, 0, and 1 are the key values for color scaling
    limits = c(-1, 1)  # Set the color scale limits between -1 and 1
  ) +
  theme_minimal() +
  labs(y="", x="", fill="Correlation") +
  theme(
        panel.grid = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.line.x = element_blank())


fa <- all[all$X == "fatty_acid",]
ggplot(fa, aes(x = LabValue, y = BileAcid, fill = pValue)) +
  geom_tile() +
  # scale_fill_gradientn(
  #   colors = c("royalblue4", "white", "brown4"), 
  #   values = scales::rescale(c(-1,0, 1))) +
  scale_fill_gradientn(
    colors = c("royalblue4", "white", "brown4"), 
    values = scales::rescale(c(-1, 0, 1)),  # Ensure -1, 0, and 1 are the key values for color scaling
    limits = c(-1, 1)  # Set the color scale limits between -1 and 1
  ) +
  theme_minimal() +
  labs(y="", x="") +
  theme(legend.position = "none", ,
        panel.grid = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.line.x = element_blank())


aa <- all[all$X %in% c( "beta_amino_acid", "alpha_amino_acid"),]
ggplot(aa, aes(x = LabValue, y = BileAcid, fill = pValue)) +
  geom_tile() +
  # scale_fill_gradientn(
  #   colors = c("royalblue4", "white", "brown4"), 
  #   values = scales::rescale(c(-1,0, 1))) +
  theme_minimal() +
  scale_fill_gradientn(
    colors = c("royalblue4", "white", "brown4"), 
    values = scales::rescale(c(-1, 0, 1)),  # Ensure -1, 0, and 1 are the key values for color scaling
    limits = c(-1, 1)  # Set the color scale limits between -1 and 1
  ) +
  labs(y="", x="") +
  theme(legend.position = "none", ,
        panel.grid = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.line.x = element_blank())


other <- all[all$X %in% c("dicarboxylic_acid",
                          "tricarboxylic_acid", "hydroxyl_acid",
                          "lactam", "auxin", "keto_acid",
                          "indole_derivative", "aryl_sulfate"),]
ggplot(other, aes(x = LabValue, y = BileAcid, fill = pValue)) +
  geom_tile() +
  theme_minimal() +
  scale_fill_gradientn(
    colors = c("royalblue4", "white", "brown4"), 
    values = scales::rescale(c(-1, 0, 1)),
    limits = c(-1, 1)
  ) +
  labs(y="", x="", fill="") +
  theme(legend.position="bottom" ,
        panel.grid = element_blank())

### -----------------
# linear regression for each combination
#####

bile_acids <- colnames(data)[11:114]
lab_values <- colnames(data)[121:122]
p_matrix <- matrix(1, nrow = length(bile_acids), ncol = length(lab_values))
beta_matrix <- matrix(0, nrow = length(bile_acids), ncol = length(lab_values))

rownames(p_matrix) <- bile_acids
colnames(p_matrix) <- lab_values

rownames(beta_matrix) <- bile_acids
colnames(beta_matrix) <- lab_values

covariates <- c("BMI", "fast")

for (bile in bile_acids) {
  for (lab in lab_values) {
    complete_data <- na.omit(data[, c(bile, lab, covariates)])
    
    model <- lm(as.formula(paste(lab, "~", bile, "+", paste(covariates, collapse = " + "))), 
                data = complete_data)
    
    p_matrix[bile, lab] <- summary(model)$coefficients[bile, "Pr(>|t|)"]
    beta_matrix[bile, lab] <- summary(model)$coefficients[bile, "Estimate"]
  }
}

q_matrix <- matrix(p.adjust(p_matrix, method = "BH"), 
                   nrow = length(bile_acids), 
                   dimnames = list(bile_acids, lab_values))

# q_df <- melt(q_matrix)

heatmap_matrix <- -log10(q_matrix) * sign(beta_matrix)
q_df <- melt(heatmap_matrix)

colnames(q_df) <- c("BileAcid", "LabValue", "qValue")

ggplot(q_df, aes(x = LabValue, y = BileAcid, fill = qValue)) +
  geom_tile() +
  theme_minimal() +
  scale_fill_gradientn(
    colors = c("royalblue4", "white", "brown4"),
    values = scales::rescale(c(-1, 0, 1)),
    limits = c(-1.051219, 1.051219)) +
  labs(fill = "-log10(qValue)*Sign(Beta)") +
  theme(panel.grid = element_blank())


