# Project-root paths, model orders and helpers used by R/dcc_egarch.R.

OUT <- "output"
FIG <- "figures"    # regenerated, not tracked

dir.create(OUT, showWarnings = FALSE)
dir.create(FIG, showWarnings = FALSE)

# Orders the committed outputs were fitted with; Section 3.1 reports the BIC
# minimum at p = 3, q = 1.
GARCH_ORDER <- c(1, 2)
ARMA_ORDER  <- c(1, 2)

read_pairs <- function() {
  pairs <- read.csv(file.path(OUT, "pair_df.csv"), row.names = 1)
  pairs$pair_name <- paste(pairs$Ticker.1, pairs$Ticker.2)
  pairs
}

# Rolling mean of each spread over its pair's half life, then standardised.
rolling_scaled <- function(spread_file, half_life_file, n_pairs) {
  spread <- as.xts(read.csv(file.path(OUT, spread_file), row.names = 1))
  half_life <- read.csv(file.path(OUT, half_life_file), row.names = 1)
  for (col in 1:n_pairs) {
    window <- half_life$Half.Life[col]
    spread[, col] <- rollmean(spread[, col], k = window, align = "right", fill = NA)
  }
  scale(na.omit(spread), center = TRUE, scale = TRUE)
}
