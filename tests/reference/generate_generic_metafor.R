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
two_study_hk_input <- input[c(1, 4), ]
reml_hk_k2 <- rma.uni(
  effect,
  variance,
  data = two_study_hk_input,
  method = "REML",
  test = "knha",
  control = reml_control
)

q <- unname(common$QE)
q_df <- unname(common$k - common$p)

riley_prediction_interval <- function(fit) {
  prediction <- predict(fit, predtype = "Riley")
  c(unname(prediction$pi.lb), unname(prediction$pi.ub))
}

q_profile <- confint(
  reml,
  level = 95,
  type = "QP",
  control = list(tol = 1e-10, maxiter = 1000)
)$random

q_profile_row <- function(name, divisor = 1) {
  list(
    estimate = unname(q_profile[name, "estimate"]) / divisor,
    ci_low = unname(q_profile[name, "ci.lb"]) / divisor,
    ci_high = unname(q_profile[name, "ci.ub"]) / divisor
  )
}

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
  reml_hartung_knapp_k2 = fit_summary(reml_hk_k2),
  reml_q_profile = list(
    method = "QP",
    confidence_level = 0.95,
    tau2 = q_profile_row("tau^2"),
    tau = q_profile_row("tau"),
    i2 = q_profile_row("I^2(%)", divisor = 100),
    h2 = q_profile_row("H^2")
  ),
  reml_prediction_interval_hts = riley_prediction_interval(reml),
  reml_prediction_interval_hartung_knapp = riley_prediction_interval(reml_hk),
  reml_prediction_interval_hartung_knapp_adhoc =
    riley_prediction_interval(reml_adhoc)
)

write_json(
  reference,
  output,
  auto_unbox = TRUE,
  digits = 16,
  pretty = TRUE
)
