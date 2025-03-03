library(tidyverse)
library(wesanderson)
library(gridExtra)

pMCIiAD <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/pMCIiAD.csv")

ggplot(pMCIiAD, aes(x=DIFF)) +
  geom_histogram()+
  theme_bw() +
  labs(x="Distribution of \u0394t from Visit of AD Incidence",
       y="Count (months)")+
  scale_x_continuous(breaks = unique(pMCIiAD$DIFF))

ggplot(pMCIiAD, aes(x=DIFF)) +
  geom_bar(fill="lightblue") +
  theme_bw() +
  labs(x="Distribution of \u0394t from Visit of AD Incidence (months)",
       y="Count") +
  scale_x_continuous(breaks = unique(pMCIiAD$DIFF))+
  geom_text(stat='count', aes(label=..count..), vjust=-0.5)

diffs <- as.data.frame(pMCIiAD$DIFF)
colnames(diffs) <- c("DIFF")
counts <- table(diffs$DIFF)
diffs$DIFF <- ifelse(counts[as.character(diffs$DIFF)] < 30, NA, diffs$DIFF)
diffs_long <- diffs %>%
  bind_rows(diffs, diffs) %>%
  mutate(Metric = rep(c("AUROC", "Accuracy", "AURPC"), each = nrow(diffs)))

ggplot(diffs_long, aes(x=DIFF)) +
  geom_bar(fill="lightblue") +
  theme_bw() +
  labs(x="Distribution of \u0394t from Visit of AD Incidence (months)",
       y="") +
  scale_x_continuous(breaks = unique(diffs$DIFF))+
  facet_wrap(~Metric)  +
  theme(strip.text = element_blank(),
        axis.text.y = element_text(color = "white"),
        axis.ticks.y = element_blank(),
        panel.grid = element_blank())

# --------------

metrics <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/LR_deltaT.csv")
metrics_long <- metrics %>%
  pivot_longer(cols = c(Accuracy, AUROC, AUPRC), names_to = "Metric", values_to = "Value")


main_plot <- ggplot(metrics_long, aes(x = Case, y = Value, color = Model, fill = Model)) +
  geom_point() +
  theme_bw() +
  scale_x_continuous(breaks = as.numeric(unique(metrics_long$Case)),
                     labels = as.numeric(unique(metrics_long$Case))) +
  geom_smooth(method = "loess", se = TRUE, linewidth = 1, alpha = 0.2) +
  scale_color_manual(values = wes_palette("Zissou1")) +
  scale_fill_manual(values = wes_palette("Zissou1")) +
  facet_wrap(~ Metric) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  labs(y="") +
  theme(axis.title.x = element_blank(), legend.position = "top",
        axis.text.x = element_blank())
hist_plot <- ggplot(diffs_long, aes(x=DIFF)) +
  geom_bar(fill="#d6e7ec") +
  theme_bw() +
  labs(x="Distribution of \u0394t from Visit of AD Incidence (months)",
       y="") +
  scale_x_continuous(breaks = unique(diffs$DIFF))+
  facet_wrap(~Metric)  +
  theme(strip.text = element_blank(),
        axis.text.y = element_text(color = "white"),
        axis.ticks.y = element_blank(),
        panel.grid = element_blank())
grid.arrange(main_plot, hist_plot, ncol = 1, heights = c(2, .75))
