# Regenerate continuous_metafor.json from the repository root with:
# Rscript tests/reference/generate_continuous_metafor.R [optional-output-path]

library(jsonlite)
library(metafor)

args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args) >= 1) args[[1]] else {
  "tests/reference/continuous_metafor.json"
}
input <- read.csv("tests/reference/continuous_input.csv")

calculate <- function(measure) {
  effects <- escalc(
    measure = measure,
    m1i = mean_treat,
    sd1i = sd_treat,
    n1i = n_treat,
    m2i = mean_control,
    sd2i = sd_control,
    n2i = n_control,
    data = input,
    vtype = "LS",
    correct = TRUE
  )
  fit <- rma.uni(yi, vi, data = effects, method = "EE")

  list(
    effect = unname(effects$yi),
    variance = unname(effects$vi),
    estimate = unname(fit$b[1]),
    standard_error = unname(fit$se),
    ci = c(unname(fit$ci.lb), unname(fit$ci.ub))
  )
}

reference <- list(
  generated_by = "R metafor",
  r_version = R.version.string,
  metafor_version = as.character(packageVersion("metafor")),
  jsonlite_version = as.character(packageVersion("jsonlite")),
  md = calculate("MD"),
  smd = calculate("SMD")
)

write_json(
  reference,
  output,
  auto_unbox = TRUE,
  digits = 16,
  pretty = TRUE
)
