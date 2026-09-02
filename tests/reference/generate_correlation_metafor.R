# Regenerate correlation_metafor.json from the repository root with:
# Rscript tests/reference/generate_correlation_metafor.R [optional-output-path]

library(jsonlite)
library(metafor)

args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args) >= 1) args[[1]] else {
  "tests/reference/correlation_metafor.json"
}
input <- read.csv("tests/reference/correlation_input.csv")

effects <- escalc(
  measure = "ZCOR",
  ri = correlation,
  ni = n,
  data = input
)

fit_payload <- function(fit) {
  list(
    estimate = unname(fit$b[1]),
    standard_error = unname(fit$se),
    ci = c(unname(fit$ci.lb), unname(fit$ci.ub)),
    display_estimate = unname(transf.ztor(fit$b[1])),
    display_ci = unname(transf.ztor(c(fit$ci.lb, fit$ci.ub))),
    tau2 = unname(fit$tau2),
    weights = unname(weights(fit) / sum(weights(fit)))
  )
}

common <- rma.uni(yi, vi, data = effects, method = "EE")
random <- rma.uni(
  yi,
  vi,
  data = effects,
  method = "REML",
  control = list(threshold = 1e-10, maxiter = 1000)
)

reference <- list(
  generated_by = "R metafor",
  r_version = R.version.string,
  metafor_version = as.character(packageVersion("metafor")),
  jsonlite_version = as.character(packageVersion("jsonlite")),
  measure = "ZCOR",
  effect = unname(effects$yi),
  variance = unname(effects$vi),
  common = fit_payload(common),
  random_reml = fit_payload(random)
)

write_json(
  reference,
  output,
  auto_unbox = TRUE,
  digits = 16,
  pretty = TRUE
)
