library(tidyverse)
library(ppcor)
library(reshape2)
library(gridExtra)

data <- read.csv("/Users/sophiekk/PycharmProjects/bileacids/processed/master_data.csv")

###
# HEALTHY V PREVALENT AD
###

# prevalent AD - 209 people, 566 samples
pAD <- data %>%
  group_by(RID) %>%
  filter(all(DX_VALS == 4)) %>%
  ungroup()

# healthy - 297 people, 938 samples
HC <- data %>%
  group_by(RID) %>%
  filter(all(DX_VALS == 1)) %>%
  ungroup()

# prevalent MCI - 374 people, 1040 samples
pMCI <- data %>%
  group_by(RID) %>%
  filter(all(DX_VALS %in% c(2,3))) %>%
  ungroup()


HCAD <- rbind(pAD, HC)
write.csv(HCAD, "/Users/sophiekk/PycharmProjects/bileacids/processed/HCAD.csv",
          row.names = FALSE)

HCMCI <- rbind(pMCI, HC)
write.csv(HCMCI, "/Users/sophiekk/PycharmProjects/bileacids/processed/HCMCI.csv",
          row.names = FALSE)

MCIAD <- rbind(pMCI, pAD)
write.csv(MCIAD, "/Users/sophiekk/PycharmProjects/bileacids/processed/MCIAD.csv",
          row.names = FALSE)
