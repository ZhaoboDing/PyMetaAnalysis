# Regenerate sparse_binary_review_metafor.json from the repository root with:
# Rscript tests/reference/generate_sparse_binary_review_metafor.R [output-path]

library(jsonlite)
library(metafor)

required_r <- "4.6.1"
required_metafor <- "5.0.1"
required_jsonlite <- "2.0.0"
stopifnot(as.character(getRversion()) == required_r)
stopifnot(as.character(packageVersion("metafor")) == required_metafor)
stopifnot(as.character(packageVersion("jsonlite")) == required_jsonlite)

args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args) >= 1) args[[1]] else {
  "tests/reference/sparse_binary_review_metafor.json"
}

cases <- list(
  no_zero_balanced = data.frame(
    event_treat = c(1, 4, 12, 20),
    n_treat = c(50, 80, 100, 120),
    event_control = c(2, 6, 8, 18),
    n_control = c(50, 80, 100, 120)
  ),
  rare_single_zero = data.frame(
    event_treat = c(0, 2, 1, 4, 0),
    n_treat = c(40, 80, 30, 100, 60),
    event_control = c(1, 0, 3, 5, 2),
    n_control = c(42, 75, 35, 110, 65)
  ),
  double_boundary = data.frame(
    event_treat = c(0, 20, 0, 3, 5, 20),
    n_treat = c(30, 20, 25, 40, 50, 20),
    event_control = c(0, 25, 2, 0, 6, 24),
    n_control = c(35, 25, 30, 45, 55, 25)
  ),
  unequal_arms = data.frame(
    event_treat = c(1, 8, 2, 50),
    n_treat = c(10, 400, 25, 2000),
    event_control = c(15, 1, 25, 3),
    n_control = c(500, 20, 1000, 30)
  ),
  opposed_effects = data.frame(
    event_treat = c(1, 12, 2, 16, 5),
    n_treat = c(100, 100, 80, 120, 90),
    event_control = c(8, 2, 9, 3, 6),
    n_control = c(100, 100, 80, 120, 90)
  ),
  large_counts = data.frame(
    event_treat = c(1000, 2500000, 50000000, 125000),
    n_treat = c(1000000, 50000000, 100000000, 25000000),
    event_control = c(1500, 2000000, 48000000, 150000),
    n_control = c(1200000, 45000000, 100000000, 30000000)
  )
)

fit_summary <- function(fit) {
  list(
    estimate = unname(fit$b[1]),
    standard_error = unname(fit$se),
    ci = c(unname(fit$ci.lb), unname(fit$ci.ub)),
    weights = unname(weights(fit) / 100)
  )
}

heterogeneity_summary <- function(fit) {
  list(
    q = unname(fit$QE),
    df = unname(fit$k - 1),
    pvalue = unname(fit$QEp),
    i2 = unname(fit$I2 / 100),
    h2 = unname(fit$H2)
  )
}

study_summary <- function(data, measure) {
  effects <- escalc(
    measure = measure,
    ai = event_treat,
    bi = n_treat - event_treat,
    ci = event_control,
    di = n_control - event_control,
    data = data,
    add = 0.5,
    to = "only0",
    drop00 = measure != "RD"
  )
  list(
    effect = unname(effects$yi),
    variance = unname(effects$vi)
  )
}

mh_summary <- function(data, measure, pooling_correction) {
  fit <- rma.mh(
    measure = measure,
    ai = event_treat,
    bi = n_treat - event_treat,
    ci = event_control,
    di = n_control - event_control,
    data = data,
    add = c(0.5, pooling_correction),
    to = c("only0", if (pooling_correction > 0) "only0" else "none"),
    drop00 = c(measure != "RD", measure != "RD"),
    correct = FALSE
  )
  result <- fit_summary(fit)
  if (pooling_correction == 0) {
    result$heterogeneity <- heterogeneity_summary(fit)
  }
  result
}

peto_summary <- function(data) {
  effects <- escalc(
    measure = "PETO",
    ai = event_treat,
    bi = n_treat - event_treat,
    ci = event_control,
    di = n_control - event_control,
    data = data,
    add = 0.5,
    to = "only0",
    drop00 = TRUE
  )
  fit <- rma.peto(
    ai = event_treat,
    bi = n_treat - event_treat,
    ci = event_control,
    di = n_control - event_control,
    data = data,
    add = c(0.5, 0),
    to = c("only0", "none"),
    drop00 = c(TRUE, TRUE)
  )
  result <- fit_summary(fit)
  result$heterogeneity <- heterogeneity_summary(fit)
  list(
    effect = unname(effects$yi),
    variance = unname(effects$vi),
    fit = result
  )
}

case_summary <- function(data) {
  measures <- lapply(c("OR", "RR", "RD"), function(measure) {
    list(
      studies = study_summary(data, measure),
      mh_raw = mh_summary(data, measure, 0),
      mh_only_zero = mh_summary(data, measure, 0.5)
    )
  })
  names(measures) <- c("OR", "RR", "RD")
  list(
    input = lapply(data, unname),
    measures = measures,
    peto = peto_summary(data)
  )
}

reference <- list(
  generated_by = "R metafor",
  r_version = R.version.string,
  metafor_version = as.character(packageVersion("metafor")),
  jsonlite_version = as.character(packageVersion("jsonlite")),
  confidence_level = 0.95,
  study_correction = list(add = 0.5, scope = "only0"),
  mh_pooling_corrections = list(raw = 0, only_zero = 0.5),
  cases = lapply(cases, case_summary)
)

write_json(
  reference,
  output,
  auto_unbox = TRUE,
  digits = 16,
  pretty = TRUE,
  na = "null"
)
