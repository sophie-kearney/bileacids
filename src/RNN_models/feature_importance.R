library(tidyverse)
model <-
feature_importance <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/123_MaskedGRU_03111714_0.7782importance.csv")
feature_top <- feature_importance %>% 
  head(20)

ggplot(feature_top, aes(x = reorder(bileacid, contribution), y = contribution)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(y = "Importance", x="")

files <- list.files(path = "/Users/sophiekk/PycharmProjects/bileacids/processed", pattern = "^123_.*\\.csv$", full.names = TRUE)
models <- lapply(files, function(file) {
  data <- read.csv(file)
  head(data$bileacid, 20)
})
feature_counts <- table(unlist(models))

final_features <- names(feature_counts[feature_counts >= 3])
final_features

