# Regenerate meta_regression_contrasts_metafor.json from the repository root:
# Rscript tests/reference/generate_meta_regression_contrasts_metafor.R

library(jsonlite)
library(metafor)

args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args) >= 1) args[[1]] else {
  "tests/reference/meta_regression_contrasts_metafor.json"
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

contrast_matrix <- rbind(
  south_minus_east = c(0, 0, 0, 1, -1),
  age_plus_half_dose = c(0, 1, 0.5, 0, 0),
  east_profile = c(1, 50, 1.5, 0, 1)
)
rhs <- c(0, 0, 0)

numeric_array <- function(values) {
  as.list(unname(as.numeric(values)))
}

contrast_summary <- function(fit) {
  tests <- anova(fit, X = contrast_matrix, rhs = rhs)
  list(
    estimate = numeric_array(tests$Xb),
    standard_error = numeric_array(tests$se),
    statistic = numeric_array(tests$zval),
    pvalue = numeric_array(tests$pval),
    joint_statistic = unname(as.numeric(tests$QM)),
    joint_df = numeric_array(tests$QMdf),
    joint_pvalue = unname(as.numeric(tests$QMp))
  )
}

fit_model <- function(method, test = "z") {
  rma.uni(
    yi = effect,
    vi = variance,
    mods = ~mean_age + dose + region,
    data = input,
    method = method,
    test = test,
    control = iterative_control
  )
}

reference <- list(
  generated_by = "R metafor",
  r_version = R.version.string,
  metafor_version = as.character(packageVersion("metafor")),
  jsonlite_version = as.character(packageVersion("jsonlite")),
  categorical_levels = list(region = levels(input$region)),
  term_names = c(
    "intrcpt",
    "mean_age",
    "dose",
    "regionSouth",
    "regionEast"
  ),
  contrast_names = rownames(contrast_matrix),
  contrast_matrix = lapply(
    seq_len(nrow(contrast_matrix)),
    function(row) numeric_array(contrast_matrix[row, ])
  ),
  rhs = numeric_array(rhs),
  iterative_control = list(tolerance = 1e-10, max_iterations = 1000),
  models = list(
    common = contrast_summary(fit_model("EE")),
    mixed_reml = contrast_summary(fit_model("REML")),
    mixed_reml_hartung_knapp = contrast_summary(fit_model("REML", "knha")),
    mixed_reml_hartung_knapp_adhoc = contrast_summary(
      fit_model("REML", "adhoc")
    )
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
