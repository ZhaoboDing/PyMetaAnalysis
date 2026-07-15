# Regenerate workflow_metafor.json from the repository root with:
# Rscript tests/reference/generate_workflow_metafor.R [optional-output-path]

library(jsonlite)
library(metafor)

args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args) >= 1) args[[1]] else {
  "tests/reference/workflow_metafor.json"
}
input <- read.csv("tests/reference/workflow_input.csv")

reml_control <- list(threshold = 1e-10, maxiter = 1000)

fit_summary <- function(fit) {
  list(
    estimate = unname(fit$b[1]),
    standard_error = unname(fit$se),
    ci = c(unname(fit$ci.lb), unname(fit$ci.ub)),
    tau2 = unname(fit$tau2)
  )
}

series_summary <- function(series) {
  tau2 <- if (is.null(series$tau2)) rep(0, length(series$estimate)) else {
    series$tau2
  }
  list(
    study = sub("^-", "", as.character(series$slab)),
    estimate = unname(series$estimate),
    standard_error = unname(series$se),
    ci_low = unname(series$ci.lb),
    ci_high = unname(series$ci.ub),
    tau2 = unname(tau2),
    q = unname(series$Q),
    q_pvalue = unname(series$Qp),
    i2 = unname(series$I2 / 100),
    h2 = unname(series$H2)
  )
}

common <- rma.uni(
  effect,
  variance,
  slab = study,
  data = input,
  method = "EE",
  test = "z"
)
random <- rma.uni(
  effect,
  variance,
  slab = study,
  data = input,
  method = "REML",
  test = "z",
  control = reml_control
)

group_fits <- lapply(
  split(input, input$subgroup),
  function(group) rma.uni(
    effect,
    variance,
    slab = study,
    data = group,
    method = "EE",
    test = "z"
  )
)
moderator <- rma.uni(
  effect,
  variance,
  mods = ~ factor(subgroup),
  data = input,
  method = "EE",
  test = "z"
)

common_cumulative <- cumul(common, order = input$year)
random_cumulative <- cumul(random, order = input$year)

reference <- list(
  generated_by = "R metafor",
  r_version = R.version.string,
  metafor_version = as.character(packageVersion("metafor")),
  jsonlite_version = as.character(packageVersion("jsonlite")),
  iterative_control = list(tolerance = 1e-10, max_iterations = 1000),
  subgroup_common = list(
    overall = fit_summary(common),
    groups = lapply(group_fits, fit_summary),
    q_between = unname(moderator$QM),
    q_between_df = unname(moderator$QMdf[1]),
    q_between_pvalue = unname(moderator$QMp),
    i2_between = max(0, (moderator$QM - moderator$QMdf[1]) / moderator$QM)
  ),
  leave_one_out = list(
    common = series_summary(leave1out(common)),
    random_reml = series_summary(leave1out(random))
  ),
  cumulative = list(
    common = c(
      list(k = unname(common_cumulative$k), year = common_cumulative$order),
      series_summary(common_cumulative)
    ),
    random_reml = c(
      list(k = unname(random_cumulative$k), year = random_cumulative$order),
      series_summary(random_cumulative)
    )
  )
)

write_json(
  reference,
  output,
  auto_unbox = TRUE,
  digits = 16,
  pretty = TRUE
)
