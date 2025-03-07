library(tidyverse)
library(stringr)
library(data.table)

delta_t = -6
iAD <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/deltaT_iAD.csv")

###
# ONE VALUE
###

iad_6m <- iAD[iAD$DIFF == -6, c("RID", "CA")]
iad_0m <- iAD[iAD$DIFF == 0, c("RID", "CA")]
iad_matched <- merge(iad_6m, iad_0m, by = "RID", suffixes = c("_6m", "_0m"))

t.test(iad_matched$CA_6m, iad_matched$CA_0m, paired = TRUE)

###
# ALL BA
###

cols <- colnames(iAD)[21:124]
p_values <- sapply(cols, function(col) {
  iad_6m <- iAD[iAD$DIFF == delta_t, c("RID", col), drop = FALSE]
  iad_0m <- iAD[iAD$DIFF == 0, c("RID", col), drop = FALSE]
  iad_matched <- merge(iad_6m, iad_0m, by = "RID", suffixes = c("_6m", "_0m"))
  t.test(iad_matched[[2]], iad_matched[[3]], paired = TRUE)$p.value
})
p_adjusted <- p.adjust(p_values, method = "fdr")

results <- data.frame(Bile_Acid = colnames(iAD)[21:124], P_Value = p_values, FDR_Corrected = p_adjusted)

significant_results <- results[results$FDR_Corrected < 0.05, ]
print(significant_results)

###
# VISUALIZE
###

iad_long <- melt(iad_matched, id.vars = "RID", measure.vars = c("CA_6m", "CA_0m"))

ggplot(iad_long, aes(x = variable, y = value)) +
  geom_violin(trim = FALSE, fill = "lightblue") +
  geom_boxplot(width = 0.1, outlier.shape = NA) +
  labs(x = "Timepoint", y = "CA Levels", title = "Bile Acid CA Levels at -6m and 0m") +
  theme_minimal()
