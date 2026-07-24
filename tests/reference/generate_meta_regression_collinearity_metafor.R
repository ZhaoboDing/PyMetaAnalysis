# Regenerate meta_regression_collinearity_metafor.json from the repository root:
# Rscript tests/reference/generate_meta_regression_collinearity_metafor.R

library(jsonlite)
library(metafor)

args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args) >= 1) args[[1]] else {
  "tests/reference/meta_regression_collinearity_metafor.json"
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

vif_summary <- function(fit) {
  term_table <- as.data.frame(vif(fit))
  moderator_table <- as.data.frame(
    vif(
      fit,
      btt = list(
        mean_age = 2L,
        dose = 3L,
        region = c(4L, 5L)
      )
    )
  )
  list(
    term_vif = numeric_array(term_table$vif),
    term_sif = numeric_array(term_table$sif),
    moderator_gvif = numeric_array(moderator_table$vif),
    moderator_gsif = numeric_array(moderator_table$sif)
  )
}

fit_model <- function(method) {
  rma.uni(
    yi = effect,
    vi = variance,
    mods = ~mean_age + dose + region,
    data = input,
    method = method,
    test = "z",
    control = iterative_control
  )
}

reference <- list(
  generated_by = "R metafor",
  r_version = R.version.string,
  metafor_version = as.character(packageVersion("metafor")),
  jsonlite_version = as.character(packageVersion("jsonlite")),
  categorical_levels = list(region = levels(input$region)),
  term_names = c("mean_age", "dose", "regionSouth", "regionEast"),
  moderator_names = c("mean_age", "dose", "region"),
  iterative_control = list(tolerance = 1e-10, max_iterations = 1000),
  models = list(
    common = vif_summary(fit_model("EE")),
    mixed_reml = vif_summary(fit_model("REML"))
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
