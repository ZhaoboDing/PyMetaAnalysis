# Regenerate meta_regression_metafor.json from the repository root with:
# Rscript tests/reference/generate_meta_regression_metafor.R [optional-output-path]

library(jsonlite)
library(metafor)

args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args) >= 1) args[[1]] else {
  "tests/reference/meta_regression_metafor.json"
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

fit_summary <- function(fit, term_names, tau2_null = NULL) {
  residual_df <- fit$k - fit$p
  if (fit$method == "EE") {
    residual_i2 <- if (fit$QE <= 0) {
      0
    } else {
      max(0, (fit$QE - residual_df) / fit$QE)
    }
    residual_h2 <- fit$QE / residual_df
  } else {
    residual_i2 <- fit$I2 / 100
    residual_h2 <- fit$H2
  }
  pseudo_r2_raw <- if (is.null(tau2_null) || tau2_null <= 0) {
    NA_real_
  } else {
    1 - fit$tau2 / tau2_null
  }

  list(
    term_names = as.list(term_names),
    coefficients = list(
      estimate = numeric_array(fit$b),
      standard_error = numeric_array(fit$se),
      statistic = numeric_array(fit$zval),
      pvalue = numeric_array(fit$pval),
      ci_low = numeric_array(fit$ci.lb),
      ci_high = numeric_array(fit$ci.ub),
      covariance = numeric_matrix(vcov(fit))
    ),
    tau2 = unname(fit$tau2),
    tau2_null = if (is.null(tau2_null)) NA_real_ else unname(tau2_null),
    pseudo_r2_raw = pseudo_r2_raw,
    pseudo_r2 = if (is.na(pseudo_r2_raw)) {
      NA_real_
    } else {
      min(1, max(0, pseudo_r2_raw))
    },
    residual_heterogeneity = list(
      q = unname(fit$QE),
      df = unname(residual_df),
      pvalue = unname(fit$QEp),
      i2 = unname(residual_i2),
      h2 = unname(residual_h2)
    ),
    global_moderator_test = list(
      statistic = unname(fit$QM),
      df_num = unname(fit$QMdf[1]),
      df_denom = if (length(fit$QMdf) >= 2) {
        unname(fit$QMdf[2])
      } else {
        NA_real_
      },
      pvalue = unname(fit$QMp)
    ),
    residual_scale = unname(fit$s2w),
    normalized_weights = numeric_array(weights(fit) / 100),
    fitted_values = numeric_array(fitted(fit)),
    residuals = numeric_array(residuals(fit)),
    leverage = numeric_array(hatvalues(fit))
  )
}

prediction_summary <- function(fit, newmods) {
  prediction <- predict(fit, newmods = newmods)
  result <- list(
    estimate = numeric_array(prediction$pred),
    standard_error = numeric_array(prediction$se),
    ci_low = numeric_array(prediction$ci.lb),
    ci_high = numeric_array(prediction$ci.ub)
  )
  if (!is.null(prediction$pi.lb)) {
    result$pi_low <- numeric_array(prediction$pi.lb)
    result$pi_high <- numeric_array(prediction$pi.ub)
  }
  result
}

fit_model <- function(mods, method, test = "z", intercept = TRUE) {
  rma.uni(
    yi = effect,
    vi = variance,
    mods = mods,
    data = input,
    method = method,
    test = test,
    intercept = intercept,
    control = iterative_control
  )
}

numeric_terms <- c("intercept", "mean_age")
common_numeric <- fit_model(~mean_age, "EE")
mixed_numeric <- lapply(
  c("DL", "PM", "REML"),
  function(method) fit_model(~mean_age, method)
)
names(mixed_numeric) <- c("DL", "PM", "REML")
numeric_null_tau2 <- vapply(
  c("DL", "PM", "REML"),
  function(method) {
    rma.uni(
      effect,
      variance,
      data = input,
      method = method,
      test = "z",
      control = iterative_control
    )$tau2
  },
  numeric(1)
)

reml_hk <- fit_model(~mean_age, "REML", "knha")
reml_adhoc <- fit_model(~mean_age, "REML", "adhoc")
numeric_prediction_values <- c(45, 55, 65)
numeric_prediction_matrix <- matrix(numeric_prediction_values, ncol = 1)
colnames(numeric_prediction_matrix) <- "mean_age"

no_intercept <- fit_model(~0 + mean_age, "EE", intercept = FALSE)
common_categorical <- fit_model(~region, "EE")
multivariable <- fit_model(~mean_age + dose + region, "REML")
multivariable_null <- rma.uni(
  effect,
  variance,
  data = input,
  method = "REML",
  test = "z",
  control = iterative_control
)
region_test <- anova(multivariable, btt = 4:5)

prediction_rows <- data.frame(
  mean_age = c(46, 54, 62),
  dose = c(0.8, 1.7, 2.3),
  region = c("North", "South", "East")
)
multivariable_prediction_matrix <- cbind(
  mean_age = prediction_rows$mean_age,
  dose = prediction_rows$dose,
  regionSouth = as.numeric(prediction_rows$region == "South"),
  regionEast = as.numeric(prediction_rows$region == "East")
)

reference <- list(
  generated_by = "R metafor",
  r_version = R.version.string,
  metafor_version = as.character(packageVersion("metafor")),
  jsonlite_version = as.character(packageVersion("jsonlite")),
  iterative_control = list(tolerance = 1e-10, max_iterations = 1000),
  categorical_levels = list(region = c("North", "South", "East")),
  common_numeric = fit_summary(common_numeric, numeric_terms),
  mixed_numeric = list(
    DL = fit_summary(
      mixed_numeric$DL,
      numeric_terms,
      numeric_null_tau2[["DL"]]
    ),
    PM = fit_summary(
      mixed_numeric$PM,
      numeric_terms,
      numeric_null_tau2[["PM"]]
    ),
    REML = fit_summary(
      mixed_numeric$REML,
      numeric_terms,
      numeric_null_tau2[["REML"]]
    )
  ),
  mixed_numeric_inference = list(
    hartung_knapp = c(
      fit_summary(reml_hk, numeric_terms, numeric_null_tau2[["REML"]]),
      list(
        prediction_values = numeric_array(numeric_prediction_values),
        predictions = prediction_summary(reml_hk, numeric_prediction_matrix)
      )
    ),
    hartung_knapp_adhoc = fit_summary(
      reml_adhoc,
      numeric_terms,
      numeric_null_tau2[["REML"]]
    )
  ),
  common_no_intercept = fit_summary(no_intercept, c("mean_age")),
  common_categorical = fit_summary(
    common_categorical,
    c("intercept", "region[South]", "region[East]")
  ),
  mixed_multivariable_reml = c(
    fit_summary(
      multivariable,
      c(
        "intercept",
        "mean_age",
        "dose",
        "region[South]",
        "region[East]"
      ),
      multivariable_null$tau2
    ),
    list(
      region_test = list(
        statistic = unname(region_test$QM),
        df_num = unname(region_test$QMdf[1]),
        pvalue = unname(region_test$QMp)
      ),
      prediction_rows = list(
        mean_age = numeric_array(prediction_rows$mean_age),
        dose = numeric_array(prediction_rows$dose),
        region = as.list(prediction_rows$region)
      ),
      predictions = prediction_summary(
        multivariable,
        multivariable_prediction_matrix
      )
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
