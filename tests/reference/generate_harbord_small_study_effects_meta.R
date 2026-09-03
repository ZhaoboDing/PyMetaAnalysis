# Regenerate harbord_small_study_effects_meta.json from the repository root with:
# Rscript tests/reference/generate_harbord_small_study_effects_meta.R [optional-output-path]

library(jsonlite)
library(meta)

args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args) >= 1) args[[1]] else {
  "tests/reference/harbord_small_study_effects_meta.json"
}
input <- read.csv("tests/reference/peters_small_study_effects_input.csv")

analysis <- metabin(
  event.e = event_treat,
  n.e = n_treat,
  event.c = event_control,
  n.c = n_control,
  studlab = study,
  data = input,
  sm = "OR",
  method = "Inverse",
  common = TRUE,
  random = FALSE,
  incr = 0.5,
  method.incr = "only0"
)
test <- metabias(
  analysis,
  method.bias = "Harbord",
  k.min = 3
)
critical <- qt(0.975, df = test$df)
intercept <- unname(test$estimate[["bias"]])
intercept_se <- unname(test$estimate[["se.bias"]])

reference <- list(
  generated_by = "R meta",
  r_version = R.version.string,
  meta_version = as.character(packageVersion("meta")),
  jsonlite_version = as.character(packageVersion("jsonlite")),
  method = "metabias(method.bias='Harbord')",
  model = "efficient-score regression with multiplicative dispersion",
  response = "standardized efficient score",
  predictor = "square root of efficient-score variance",
  weight = "efficient-score variance in the equivalent weighted form",
  uses_continuity_correction = FALSE,
  k = analysis$k,
  confidence_level = 0.95,
  intercept = intercept,
  intercept_standard_error = intercept_se,
  intercept_ci = c(
    intercept - critical * intercept_se,
    intercept + critical * intercept_se
  ),
  statistic = unname(test$statistic),
  df = unname(test$df),
  pvalue = unname(test$pval),
  limit_estimate = unname(test$intercept),
  limit_standard_error = unname(test$se.intercept),
  limit_ci = c(
    test$intercept - critical * test$se.intercept,
    test$intercept + critical * test$se.intercept
  ),
  residual_dispersion = unname(test$tau^2)
)

write_json(
  reference,
  output,
  auto_unbox = TRUE,
  digits = 16,
  pretty = TRUE
)
