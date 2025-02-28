library(tidyverse)
library(wesanderson)
library("RColorBrewer")
library(paletteer)

data <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/pMCIiAD/MaskedGRU_aucoverk.csv")
colnames(data)[colnames(data) == "aproc"] <- "auprc"
data_long <- tidyr::pivot_longer(data, cols = -k, names_to = "metric", values_to = "value")

ggplot(data_long, aes(x = k, y = value, color = metric)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks = data$k) +
  scale_y_continuous(limits = c(0.35, 0.88)) +
  scale_color_manual(values = wes_palette("Darjeeling1")) +
  labs(x = "k", y = "", color = "Metric", title="MaskedGRU on pMCIiAD") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))

simpleRNN_pMCIiAD <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/pMCIiAD/SimpleRNN_aucoverk.csv")
simpleRNN_pHCiAD <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/pHCiAD/SimpleRNN_aucoverk.csv")
simpleRNN_pMCIiAD_zp <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/pMCIiAD/not_imputed/simpleRNN_aucoverk.csv")
simpleRNN_pHCiAD_zp <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/pHCiAD/not_imputed/SimpleRNN_aucoverk.csv")
MaskedGRU_pMCIiAD <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/pMCIiAD/MaskedGRU_aucoverk.csv")
MaskedGRU_pHCiAD <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/pHCiAD/MaskedGRU_aucoverk.csv")

simpleRNN_pMCIiAD$model <- "SimpleRNN"
simpleRNN_pHCiAD$model <- "SimpleRNN"
simpleRNN_pMCIiAD_zp$model <- "SimpleRNN Zero Padded"
simpleRNN_pHCiAD_zp$model <- "SimpleRNN Zero Padded"
MaskedGRU_pMCIiAD$model <- "MaskedGRU"
MaskedGRU_pHCiAD$model <- "MaskedGRU"

simpleRNN_pMCIiAD$cohort <- "pMCIiAD"
simpleRNN_pMCIiAD_zp$cohort <- "pMCIiAD"
MaskedGRU_pMCIiAD$cohort <- "pMCIiAD"
simpleRNN_pHCiAD$cohort <- "pHCiAD"
simpleRNN_pHCiAD_zp$cohort <- "pHCiAD"
MaskedGRU_pHCiAD$cohort <- "pHCiAD"

combined_data <- bind_rows(simpleRNN_pMCIiAD, simpleRNN_pHCiAD, simpleRNN_pMCIiAD_zp, simpleRNN_pHCiAD_zp, MaskedGRU_pMCIiAD,MaskedGRU_pHCiAD)

ggplot(combined_data, aes(x = k, y = auroc, color = model)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks = unique(combined_data$k)) +
  scale_color_manual(values = wes_palette("FantasticFox1")) +
  labs(title = "AUROC Across Models", x = "k", y = "", color = "Model") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5)) +
  facet_wrap(~ cohort)

#### -----

data <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/metricsoverk_repeated.csv")

data_long <- data %>%
  pivot_longer(cols = c(AUROC, Accuracy, AUPRC), 
               names_to = "Metric", 
               values_to = "Value")

viscodes <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/first_AD.csv")
viscodes_long <- viscodes %>%
  bind_rows(viscodes, viscodes) %>%
  mutate(Metric = rep(c("AUROC", "Accuracy", "AURPC"), each = nrow(viscodes)))
viscodes_long$VISCODE2 <- factor(viscodes_long$VISCODE2, levels=c("bl","m06","m12","m18","m24","m36","m48","m60","m72","m84","m96","m108","m120"))


ggplot(data_long, aes(x = k, y = Value, color = Model, fill = Model)) +
  geom_point() +
  theme_bw() +
  scale_x_continuous(breaks = as.numeric(unique(data_long$k)), labels = unique(data_long$Visit)) +
  geom_smooth(method = "loess", se = TRUE, linewidth = 1, alpha = 0.2) +
  scale_color_manual(values = wes_palette("Chevalier1")) +
  scale_fill_manual(values = wes_palette("Chevalier1")) +
  facet_wrap(~ Metric) +
  theme(axis.text.x = element_text(angle = 90, hjust = 0.5))

## -------------

main_plot <- ggplot(data_long, aes(x = k, y = Value, color = Model, fill = Model)) +
  geom_point() +
  theme_bw() +
  scale_x_continuous(breaks = as.numeric(unique(data_long$k)), labels = unique(data_long$Visit)) +
  geom_smooth(method = "loess", se = TRUE, linewidth = 1, alpha = 0.2) +
  scale_color_manual(values = wes_palette("Chevalier1")) +
  scale_fill_manual(values = wes_palette("Chevalier1")) +
  facet_wrap(~ Metric) +
  labs(y="") +
  theme(axis.title.x = element_blank(), axis.text.x = element_blank(),
        legend.position = "top")
hist_plot <- ggplot(viscodes_long, aes(x = VISCODE2)) +
  geom_histogram(stat = "count", fill="#92a6a6") +
  theme_bw() +
  facet_wrap(~ Metric) +  
  labs(x="k", y="iAD") +
  theme(axis.text.x = element_text(angle = 90, hjust = 0.5),
        strip.text = element_blank(),
        axis.text.y = element_text(color = "white"),
        axis.ticks.y = element_blank(),
        panel.grid = element_blank())
grid.arrange(main_plot, hist_plot, ncol = 1, heights = c(2, .75))
