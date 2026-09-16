# Directories and the model order, declared once. Every R script sources this
# file, so no path is written twice and none is absolute. Run R from anywhere
# inside the project.

OUT <- "output"     # committed pipeline inputs and outputs
FIG <- "figures"    # report figures, regenerated, not tracked

dir.create(OUT, showWarnings = FALSE)
dir.create(FIG, showWarnings = FALSE)

# eGARCH and ARMA orders of the univariate margin, used by the DCC fit. These
# are the orders the committed outputs were produced with. Section 3.1 of the
# report states (1,1) for both; the two disagree and are left as they are. Not
# to be confused with dccOrder, which is (1,1) and is never searched.
GARCH_ORDER <- c(1, 2)
ARMA_ORDER  <- c(1, 2)

read_pairs <- function() {
  pairs <- read.csv(file.path(OUT, "pair_df.csv"), row.names = 1)
  pairs$pair_name <- paste(pairs$Ticker.1, pairs$Ticker.2)
  pairs
}

# A spread panel converted to the rolling mean over each pair's own half life,
# then standardised. Both scripts need exactly this.
rolling_scaled <- function(spread_file, half_life_file, n_pairs) {
  spread <- as.xts(read.csv(file.path(OUT, spread_file), row.names = 1))
  half_life <- read.csv(file.path(OUT, half_life_file), row.names = 1)
  for (col in 1:n_pairs) {
    window <- half_life$Half.Life[col]
    spread[, col] <- rollmean(spread[, col], k = window, align = "right", fill = NA)
  }
  scale(na.omit(spread), center = TRUE, scale = TRUE)
}
