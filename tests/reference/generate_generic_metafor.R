# Regenerate generic_metafor.json from the repository root with:
# Rscript tests/reference/generate_generic_metafor.R [optional-output-path]

library(jsonlite)
library(metafor)

args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args) >= 1) args[[1]] else {
  "tests/reference/generic_metafor.json"
}
input <- read.csv("tests/reference/generic_input.csv")

pm_control <- list(tol = 1e-10, maxiter = 1000)
reml_control <- list(threshold = 1e-10, maxiter = 1000)

fit_summary <- function(fit) {
  list(
    estimate = unname(fit$b[1]),
    standard_error = unname(fit$se),
    ci = c(unname(fit$ci.lb), unname(fit$ci.ub)),
    tau2 = unname(fit$tau2),
    weights = unname(weights(fit) / 100)
  )
}

common <- rma.uni(
  yi = effect,
  vi = variance,
  data = input,
  method = "EE",
  test = "z"
)
dl <- rma.uni(effect, variance, data = input, method = "DL", test = "z")
pm <- rma.uni(
  effect,
  variance,
  data = input,
  method = "PM",
  test = "z",
  control = pm_control
)
reml <- rma.uni(
  effect,
  variance,
  data = input,
  method = "REML",
  test = "z",
  control = reml_control
)
reml_hk <- rma.uni(
  effect,
  variance,
  data = input,
  method = "REML",
  test = "knha",
  control = reml_control
)
reml_adhoc <- rma.uni(
  effect,
  variance,
  data = input,
  method = "REML",
  test = "adhoc",
  control = reml_control
)

q <- unname(common$QE)
q_df <- unname(common$k - common$p)
classic_variance <- 1 / sum(1 / (input$variance + reml$tau2))
prediction_se <- sqrt(reml$tau2 + classic_variance)
prediction_critical <- qt(0.975, df = nrow(input) - 2)
prediction_interval <- c(
  unname(reml$b[1]) - prediction_critical * prediction_se,
  unname(reml$b[1]) + prediction_critical * prediction_se
)

reference <- list(
  generated_by = "R metafor",
  r_version = R.version.string,
  metafor_version = as.character(packageVersion("metafor")),
  jsonlite_version = as.character(packageVersion("jsonlite")),
  iterative_control = list(tolerance = 1e-10, max_iterations = 1000),
  heterogeneity = list(
    q = q,
    df = q_df,
    pvalue = unname(common$QEp),
    i2 = max(0, (q - q_df) / q),
    h2 = q / q_df
  ),
  common = fit_summary(common),
  random = list(
    DL = fit_summary(dl),
    PM = fit_summary(pm),
    REML = fit_summary(reml)
  ),
  reml_hartung_knapp = fit_summary(reml_hk),
  reml_hartung_knapp_adhoc = fit_summary(reml_adhoc),
  reml_prediction_interval_hts = prediction_interval
)

write_json(
  reference,
  output,
  auto_unbox = TRUE,
  digits = 16,
  pretty = TRUE
)
