library(tidyverse)
data <- data.frame("BARS Groups"=c("Low","Low","High","High"),
                   "PRS Groups"=c("Low","High","Low","High"),
                   "Odds_Ratio"=c(0.15858720338319368,0.1650390625,42.560090702947846,64.0))

ggplot(data, aes(x=BARS.Groups, y=PRS.Groups, fill=Odds_Ratio)) +
  geom_tile(color=NA) +
  theme_minimal() +
  geom_text(aes(label = sprintf("%.2f", Odds_Ratio)), color = "white") +
  scale_fill_gradientn(colors = c("#046c9a", "white", "#9b130d"), values = c(0,.1,1))+
  labs(x="BARS Group", y="PRS Group",fill="Odds Ratio") +
  theme(panel.grid.major = element_blank(),
    panel.grid.minor = element_blank())
