# DCC-eGARCH on the pair spreads, and the z scores the strategy trades on.
#
# The univariate margin is eGARCH with an ARMA mean and Student t
# innovations; the DCC layer is order (1,1) with a multivariate t. The training
# fit is done by dccfit. The testing volatilities are then propagated forward
# one day at a time from the fitted theta and the burn-in mean of Q, so the test
# period uses no information from its own future.
#
#   Rscript R/dcc_egarch.R
#
# The training fit is the slow step, around two hours on the original panel.
#
# Inputs   output/pair_df.csv
#          output/{train,test}_spread.csv, output/{train,test}_half_life.csv
# Outputs  output/{train,test}_spread_scaled.csv
#          output/{train,test}_spread_vol.csv
#          output/{train,test}_z_score.csv
#          figures/heatmap.jpg

# Run from anywhere: walk up from the working directory to the project root,
# the folder holding R/ and output/. RStudio opens in whatever directory it used
# last, so the script locates itself rather than assuming.
local({
  path <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  for (i in 1:6) {
    if (file.exists(file.path(path, "R", "00_paths.R"))) { setwd(path); return(invisible()) }
    parent <- dirname(path)
    if (parent == path) break
    path <- parent
  }
  stop("Cannot find the project root. setwd() to the folder containing R/ and output/, then rerun.")
})

source("R/00_paths.R")
library(xts)
library(zoo)
library(rmgarch)

pair_description <- read_pairs()
pairs.name <- pair_description$pair_name
pairs.no <- length(pairs.name); pairs.no

# Training ----
train.spread_scaled <- rolling_scaled("train_spread.csv", "train_half_life.csv", pairs.no)
train.length <- nrow(train.spread_scaled); train.length

# The order comes from R/00_paths.R and is the one this script was originally
# run with. Section 3.1 of the report states eGARCH(1,1) with an ARMA(1,1)
# mean; the two disagree and are left as they are.
cat(sprintf("eGARCH(%d,%d) margin, ARMA(%d,%d) mean, Student t\n",
            GARCH_ORDER[1], GARCH_ORDER[2], ARMA_ORDER[1], ARMA_ORDER[2]))
garch.spec <- ugarchspec(variance.model = list(model = "eGARCH", garchOrder = GARCH_ORDER),
                         mean.model = list(armaOrder = ARMA_ORDER, include.mean = FALSE),
                         distribution.model = "std")
dcc_spec <- dccspec(uspec = multispec(replicate(garch.spec, n = pairs.no)),
                    dccOrder = c(1, 1), model = "DCC", distribution = "mvt")
dcc_GARCH <- dccfit(spec = dcc_spec, data = train.spread_scaled)

train.cov_matrix <- vector("list", train.length)
train.vol <- matrix(NA, nrow = train.length, ncol = pairs.no,
                    dimnames = list(gsub("CST", "America/Chicago", index(train.spread_scaled)),
                                    pairs.name))
for (day in (1:train.length)) {
  train.cov_matrix[[day]] <- as.matrix(as.data.frame(dcc_GARCH@mfit$Q[day]))
  train.vol[day, ] <- diag(train.cov_matrix[[day]])
}

train.z_score <- train.spread_scaled / train.vol
colnames(train.z_score) <- pairs.name

write.zoo(as.zoo(train.spread_scaled), file = file.path(OUT, "train_spread_scaled.csv"), sep = ",")
write.zoo(as.zoo(as.xts(train.vol)),   file = file.path(OUT, "train_spread_vol.csv"),    sep = ",")
write.zoo(as.zoo(train.z_score),       file = file.path(OUT, "train_z_score.csv"),       sep = ",")

# Fitted DCC parameters, carried into the testing recursion ----
theta1 <- tail(dcc_GARCH@mfit$matcoef, 3)[1, 1]
theta2 <- tail(dcc_GARCH@mfit$matcoef, 2)[1, 1]
latest.Q <- as.data.frame(tail(dcc_GARCH@mfit$Q, 1))
one.third <- (train.length %/% 3)     # burn in the first third before averaging Q
Q_sum <- as.matrix(as.data.frame(dcc_GARCH@mfit$Q[one.third]))
for (day in (one.third:train.length)) {
  Q_sum <- Q_sum + as.matrix(as.data.frame(train.cov_matrix[day]))
}
Q_mean <- Q_sum / length(one.third:train.length)

jpeg(file = file.path(FIG, "heatmap.jpg"), quality = 90)
heatmap(Q_mean, col = colorRampPalette(c("green3", "yellow2", "red3"))(100),
        main = "Covariance Heatmap", xlab = "Pairs", ylab = "Pairs")
dev.off()

write.csv(data.frame(theta1 = theta1, theta2 = theta2),
          file.path(OUT, "dcc_theta.csv"), row.names = FALSE)

# Testing ----
test.spread_scaled <- rolling_scaled("test_spread.csv", "test_half_life.csv", pairs.no)
latest.eta <- as.numeric(head(test.spread_scaled, 1))   # seeds the recursion
test.spread_scaled <- test.spread_scaled[-1, ]
test.length <- nrow(test.spread_scaled); test.length

test.cov_matrix <- vector("list", test.length)
test.vol <- matrix(NA, nrow = test.length, ncol = pairs.no,
                   dimnames = list(gsub("CST", "America/Chicago", index(test.spread_scaled)),
                                   pairs.name))
Q.t1 <- (1 - theta1 - theta2) * Q_mean + theta2 * (latest.eta %*% t(latest.eta)) + theta1 * latest.Q
test.cov_matrix[[1]] <- as.matrix(Q.t1)
test.vol[1, ] <- diag(as.matrix(Q.t1))
for (row in 2:test.length) {
  if (row == 2) { Q.lag <- as.matrix(Q.t1) }
  eta.lag <- as.numeric(test.spread_scaled[row, ])
  Q.next <- (1 - theta1 - theta2) * Q_mean + theta2 * (eta.lag %*% t(eta.lag)) + theta1 * Q.lag
  test.cov_matrix[[row]] <- as.matrix(Q.next)
  test.vol[row, ] <- diag(test.cov_matrix[[row]])
  Q.lag <- Q.next
}

test.z_score <- test.spread_scaled / test.vol
colnames(test.z_score) <- pairs.name

write.zoo(as.zoo(test.spread_scaled), file = file.path(OUT, "test_spread_scaled.csv"), sep = ",")
write.zoo(as.zoo(as.xts(test.vol)),   file = file.path(OUT, "test_spread_vol.csv"),    sep = ",")
write.zoo(as.zoo(test.z_score),       file = file.path(OUT, "test_z_score.csv"),       sep = ",")

cat("wrote train and test spread_scaled, spread_vol and z_score to", OUT, "\n")
cat("theta1", theta1, " theta2", theta2, "\n")
