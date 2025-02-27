data <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/master_data.csv")

freq <- as.data.frame(table(data$RID))
ggplot(freq, aes(x=Freq))+
  geom_histogram(binwidth=1, color="darkblue", fill="lightblue") +
  theme_bw() +
  labs(y="",x="Distribution of Total Number of Visits") +
  scale_x_continuous(breaks=unique(freq$Freq))


unique(data$VISCODE2)
ggplot(freq, aes(x=Freq))+
  geom_histogram(binwidth=1, color="darkblue", fill="lightblue") +
  theme_bw() +
  labs(y="",x="Distribution of Total Number of Visits") +
  scale_x_continuous(breaks=unique(freq$Freq))


data$VISCODE2 <- factor(data$VISCODE2, levels = c("bl", "m06", "m12", "m18", "m24", "m36", "m48", "m60", "m72", "m84", "m96", "m108", "m120"), ordered = TRUE)

ggplot(data, aes(x = VISCODE2)) +
  geom_bar(color="darkblue", fill="lightblue") +
  labs(x = "Distribution of Visits at each Timepoint", y = "") +
  theme_bw()
