### ----
# IMPORTS
### ----

library(tidyverse)
library(glmnet)
library(knitr)

### ----
# DATA WRANGLING
### -----

data <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/master_data.csv")
# data$TAU <- as.numeric(data$TAU)
# data$ABETA <- as.numeric(data$ABETA)

bl <- data[data$EXAMDATE_RANK==0,]
blHCAD <- data[data$DX_VALS %in% c(1,4),]

### ----
# LASSO
### ----

begin_BA_cols <- which(colnames(blHCAD) == "L_HISTIDINE")
end_BA_cols <- which(colnames(blHCAD)=="BUDCA")
begin_BA_ratio <- which(colnames(blHCAD) == "CA_CDCA")
end_BA_ratio <- which(colnames(blHCAD) == "GLCA_CDCA")

X <- as.matrix(blHCAD[,c(begin_BA_cols:end_BA_cols, begin_BA_ratio:end_BA_ratio)]) 
y <- ifelse(blHCAD$DX_VALS == "4", 1, ifelse(blHCAD$DX_VALS == "1", 0, NA))

cv_fit <- cv.glmnet(X, y, family = "binomial", alpha = 1)
lambda_opt <- cv_fit$lambda.min

lasso_fit <- glmnet(X, y, family = "binomial", alpha = 1, lambda = lambda_opt)

selected_features <- rownames(coef(lasso_fit))[which(coef(lasso_fit)[, 1] != 0)][-1]
print(selected_features)

coef_matrix <- as.matrix(coef(lasso_fit))
selected_features <- coef_matrix[which(coef_matrix[, 1] != 0), , drop = FALSE]

feature_table <- data.frame(
  Feature = rownames(selected_features),
  Coefficient = selected_features[, 1]
)
View(feature_table)
write.csv(feature_table,"/Users/sophiekk/PycharmProjects/bileacids/raw/LASSO.csv",
          row.names = FALSE)
