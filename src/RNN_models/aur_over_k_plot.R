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
