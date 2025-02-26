library(tidyverse)
library(wesanderson)

data <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/pMCIiAD/SimpleRNN_aucoverk.csv")

data_long <- tidyr::pivot_longer(data, cols = -k, names_to = "metric", values_to = "value")

ggplot(data_long, aes(x = k, y = value, color = metric)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks = data$k) +
  scale_y_continuous(limits = c(0.35, 0.88)) +
  scale_color_manual(values = wes_palette("Darjeeling1")) +
  labs(x = "k", y = "", color = "Metric", title="SimpleRNN on pMCIiAD") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))

simpleRNN_pMCIiAD <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/pMCIiAD/SimpleRNN_aucoverk.csv")
simpleRNN_pHCiAD <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/pHCiAD/SimpleRNN_aucoverk.csv")
simpleRNN_pMCIiAD_zp <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/pMCIiAD/not_imputed/simpleRNN_aucoverk.csv")
simpleRNN_pHCiAD_zp <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/pHCiAD/not_imputed/SimpleRNN_aucoverk.csv")
