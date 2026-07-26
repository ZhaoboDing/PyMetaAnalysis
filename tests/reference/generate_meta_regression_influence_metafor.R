# Regenerate meta_regression_influence_metafor.json from the repository root:
# Rscript tests/reference/generate_meta_regression_influence_metafor.R

library(jsonlite)
library(metafor)

args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args) >= 1) args[[1]] else {
  "tests/reference/meta_regression_influence_metafor.json"
}
input <- read.csv("tests/reference/meta_regression_input.csv")
input$region <- factor(
  input$region,
  levels = c("North", "South", "East")
)

iterative_control <- list(
  threshold = 1e-10,
  tol = 1e-10,
  maxiter = 1000
)

numeric_array <- function(values) {
  as.list(unname(as.numeric(values)))
}

numeric_matrix <- function(values) {
  matrix_value <- unname(as.matrix(values))
  lapply(
    seq_len(nrow(matrix_value)),
    function(row) numeric_array(matrix_value[row, ])
  )
}

fit_model <- function(mods, method, test = "z") {
  rma.uni(
    yi = effect,
    vi = variance,
    mods = mods,
    data = input,
    method = method,
    test = test,
    control = iterative_control
  )
}

influence_summary <- function(fit) {
  diagnostics <- influence(fit)
  studentized <- rstudent(fit)
  dfbetas_table <- as.data.frame(diagnostics$dfbs)
  list(
    deleted_residual = numeric_array(studentized$resid),
    deleted_residual_se = numeric_array(studentized$se),
    externally_standardized_residual = numeric_array(studentized$z),
    cook_distance = numeric_array(diagnostics$inf$cook.d),
    dfbetas = numeric_matrix(dfbetas_table[, seq_len(fit$p), drop = FALSE])
  )
}

fits <- list(
  common = fit_model(~mean_age, "EE"),
  mixed_dl = fit_model(~mean_age, "DL"),
  mixed_pm = fit_model(~mean_age, "PM"),
  mixed_reml = fit_model(~mean_age, "REML"),
  mixed_reml_hartung_knapp = fit_model(~mean_age, "REML", "knha"),
  mixed_reml_hartung_knapp_adhoc = fit_model(
    ~mean_age,
    "REML",
    "adhoc"
  ),
  common_categorical = fit_model(~region, "EE"),
  mixed_multivariable_reml = fit_model(
    ~mean_age + dose + region,
    "REML"
  )
)

reference <- list(
  generated_by = "R metafor",
  r_version = R.version.string,
  metafor_version = as.character(packageVersion("metafor")),
  jsonlite_version = as.character(packageVersion("jsonlite")),
  iterative_control = list(tolerance = 1e-10, max_iterations = 1000),
  categorical_levels = list(region = c("North", "South", "East")),
  models = lapply(fits, influence_summary)
)

write_json(
  reference,
  output,
  auto_unbox = TRUE,
  digits = 16,
  pretty = TRUE,
  na = "null"
)
