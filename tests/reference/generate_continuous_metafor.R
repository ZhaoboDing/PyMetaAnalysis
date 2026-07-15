# Regenerate continuous_metafor.json from the repository root with:
# Rscript tests/reference/generate_continuous_metafor.R

library(jsonlite)
library(metafor)

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
  fit <- rma.uni(yi, vi, data = effects, method = "FE")

  list(
    effect = unname(effects$yi),
    variance = unname(effects$vi),
    estimate = unname(fit$b[1]),
    standard_error = unname(fit$se),
    ci = c(unname(fit$ci.lb), unname(fit$ci.ub))
  )
}

reference <- list(
  generated_by = "metafor",
  metafor_version = as.character(packageVersion("metafor")),
  md = calculate("MD"),
  smd = calculate("SMD")
)

write_json(
  reference,
  "tests/reference/continuous_metafor.json",
  auto_unbox = TRUE,
  digits = 16,
  pretty = TRUE
)
