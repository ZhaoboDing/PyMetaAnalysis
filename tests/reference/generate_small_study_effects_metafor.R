# Regenerate small_study_effects_metafor.json from the repository root with:
# Rscript tests/reference/generate_small_study_effects_metafor.R [optional-output-path]

library(jsonlite)
library(metafor)

args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args) >= 1) args[[1]] else {
  "tests/reference/small_study_effects_metafor.json"
}
input <- read.csv("tests/reference/small_study_effects_input.csv")

test <- regtest(
  input$effect,
  input$variance,
  model = "lm",
  predictor = "sei",
  ret.fit = TRUE,
  level = 95
)
coefficients <- coef(summary(test$fit))
critical <- qt(0.975, df = test$dfs)
intercept <- unname(coefficients[2, "Estimate"])
intercept_se <- unname(coefficients[2, "Std. Error"])

reference <- list(
  generated_by = "R metafor",
  r_version = R.version.string,
  metafor_version = as.character(packageVersion("metafor")),
  jsonlite_version = as.character(packageVersion("jsonlite")),
  method = "regtest(model='lm', predictor='sei')",
  model = "weighted regression with multiplicative dispersion",
  predictor = "standard error",
  k = nrow(input),
  confidence_level = 0.95,
  intercept = intercept,
  intercept_standard_error = intercept_se,
  intercept_ci = c(
    intercept - critical * intercept_se,
    intercept + critical * intercept_se
  ),
  statistic = unname(test$zval),
  df = unname(test$dfs),
  pvalue = unname(test$pval),
  limit_estimate = unname(test$est),
  limit_standard_error = unname(coefficients[1, "Std. Error"]),
  limit_ci = c(unname(test$ci.lb), unname(test$ci.ub))
)

write_json(
  reference,
  output,
  auto_unbox = TRUE,
  digits = 16,
  pretty = TRUE
)
