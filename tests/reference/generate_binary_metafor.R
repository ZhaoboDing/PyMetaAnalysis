# Regenerate binary_metafor.json from the repository root with:
# Rscript tests/reference/generate_binary_metafor.R [optional-output-path]

library(jsonlite)
library(metafor)

args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args) >= 1) args[[1]] else {
  "tests/reference/binary_metafor.json"
}

clean_input <- read.csv("tests/reference/binary_input.csv")
sparse_input <- read.csv("tests/reference/binary_sparse_input.csv")

reml_control <- list(threshold = 1e-10, maxiter = 1000)

fit_summary <- function(fit, include_tau2 = TRUE) {
  result <- list(
    estimate = unname(fit$b[1]),
    standard_error = unname(fit$se),
    ci = c(unname(fit$ci.lb), unname(fit$ci.ub)),
    weights = unname(weights(fit) / 100)
  )
  if (include_tau2) {
    result$tau2 <- unname(fit$tau2)
  }
  result
}

calculate_effects <- function(data, measure) {
  escalc(
    measure = measure,
    ai = event_treat,
    bi = n_treat - event_treat,
    ci = event_control,
    di = n_control - event_control,
    data = data,
    add = 0.5,
    to = "only0",
    drop00 = TRUE
  )
}

calculate_clean <- function(measure) {
  effects <- calculate_effects(clean_input, measure)
  common <- rma.uni(yi, vi, data = effects, method = "EE", test = "z")
  random <- rma.uni(
    yi,
    vi,
    data = effects,
    method = "REML",
    test = "z",
    control = reml_control
  )
  result <- list(
    effect = unname(effects$yi),
    variance = unname(effects$vi),
    common_iv = fit_summary(common),
    random_reml_iv = fit_summary(random)
  )
  if (measure %in% c("OR", "RR")) {
    mh <- rma.mh(
      measure = measure,
      ai = event_treat,
      bi = n_treat - event_treat,
      ci = event_control,
      di = n_control - event_control,
      data = clean_input,
      add = c(0.5, 0),
      to = c("only0", "none"),
      drop00 = c(TRUE, TRUE),
      correct = FALSE
    )
    result$mantel_haenszel <- fit_summary(mh, include_tau2 = FALSE)
  }
  result
}

calculate_sparse <- function(measure) {
  effects <- calculate_effects(sparse_input, measure)
  common <- rma.uni(yi, vi, data = effects, method = "EE", test = "z")
  mh <- rma.mh(
    measure = measure,
    ai = event_treat,
    bi = n_treat - event_treat,
    ci = event_control,
    di = n_control - event_control,
    data = sparse_input,
    add = c(0.5, 0),
    to = c("only0", "none"),
    drop00 = c(TRUE, TRUE),
    correct = FALSE
  )
  list(
    effect = unname(effects$yi),
    variance = unname(effects$vi),
    common_iv = fit_summary(common),
    mantel_haenszel = fit_summary(mh, include_tau2 = FALSE)
  )
}

reference <- list(
  generated_by = "R metafor",
  r_version = R.version.string,
  metafor_version = as.character(packageVersion("metafor")),
  jsonlite_version = as.character(packageVersion("jsonlite")),
  iterative_control = list(tolerance = 1e-10, max_iterations = 1000),
  clean = list(
    OR = calculate_clean("OR"),
    RR = calculate_clean("RR"),
    RD = calculate_clean("RD")
  ),
  sparse = list(
    OR = calculate_sparse("OR"),
    RR = calculate_sparse("RR")
  )
)

write_json(
  reference,
  output,
  auto_unbox = TRUE,
  digits = 16,
  pretty = TRUE,
  na = "null"
)
