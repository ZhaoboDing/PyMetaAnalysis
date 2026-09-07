# Deterministic review cases, not a coverage simulation.
suppressPackageStartupMessages(library(metafor))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1]] else "tests/reference/core_inference_metafor.json"
stopifnot(getRversion() == "4.6.1", packageVersion("metafor") == "5.0.1",
          packageVersion("jsonlite") == "2.0.0")
controls <- list(DL = list(), PM = list(tol = 1e-12, maxiter = 1000),
                 REML = list(threshold = 1e-12, maxiter = 1000))
qp_control <- list(tol = 1e-12, maxiter = 1000, tau2.max = 1e5)
tests <- c(normal = "z", hartung_knapp = "knha", hartung_knapp_adhoc = "adhoc")

summarize <- function(fit) {
  list(estimate = as.numeric(coef(fit)), standard_error = as.numeric(fit$se),
       ci = c(fit$ci.lb, fit$ci.ub), tau2 = fit$tau2,
       weights = as.numeric(weights(fit)) / 100,
       q = fit$QE, q_p_value = fit$QEp, i2 = fit$I2 / 100, h2 = fit$H2)
}

cases <- list()
for (k in c(2, 3, 5, 10)) {
  for (precision in c("balanced", "unbalanced")) {
    vi <- if (precision == "balanced") rep(0.2, k) else exp(seq(log(0.02), log(2), length.out = k))
    for (pattern in c("identical", "low_spread", "high_spread")) {
      yi <- switch(pattern, identical = rep(0.3, k),
                   low_spread = seq(-0.01, 0.01, length.out = k),
                   high_spread = seq(-2, 2, length.out = k))
      fits <- list()
      for (method in names(controls)) {
        for (ci_method in names(tests)) {
          fit <- rma.uni(yi, vi, method = method, test = tests[[ci_method]],
                         level = 95, control = controls[[method]])
          entry <- c(list(tau2_method = method, ci_method = ci_method), summarize(fit))
          if (k >= 3) {
            pred <- predict(fit, predtype = "Riley", level = 95)
            entry$prediction_interval <- c(pred$pi.lb, pred$pi.ub)
          } else {
            entry$prediction_interval <- NULL
          }
          fits[[length(fits) + 1]] <- entry
        }
      }
      common <- rma.uni(yi, vi, method = "EE", test = "z", level = 95)
      reml <- rma.uni(yi, vi, method = "REML", test = "z", level = 95,
                      control = controls$REML)
      # Preserve the default search's signs: a numeric 100 with ub.sign=">"
      # is a search limit, not the actual upper confidence bound.
      native_qp <- confint(reml, type = "QP", level = 95,
                           control = list(tol = 1e-12, maxiter = 1000))
      qp_warnings <- character()
      qp <- withCallingHandlers(
        confint(reml, type = "QP", level = 95, control = qp_control),
        warning = function(w) {
          qp_warnings <<- c(qp_warnings, conditionMessage(w))
          invokeRestart("muffleWarning")
        })
      stopifnot(qp$ub.sign != ">", all(is.finite(qp$random)))
      cases[[length(cases) + 1]] <- list(
        id = paste(k, precision, pattern, sep = "_"), k = k,
        precision = precision, pattern = pattern, effect = yi, variance = vi,
        common = summarize(common), fits = fits,
        q_profile = list(tau2_ci = unname(qp$random["tau^2", c("ci.lb", "ci.ub")]),
                         empty_by_q_at_zero = common$QE < qchisq(0.025, k - 1),
                         native_empty = qp$ci.null,
                         lower_sign = qp$lb.sign, upper_sign = qp$ub.sign,
                         warnings = qp_warnings,
                         default_search = list(
                           tau2_ci = unname(native_qp$random["tau^2", c("ci.lb", "ci.ub")]),
                           lower_sign = native_qp$lb.sign, upper_sign = native_qp$ub.sign)))
    }
  }
}
result <- list(
  r_version = as.character(getRversion()),
  metafor_version = as.character(packageVersion("metafor")),
  jsonlite_version = as.character(packageVersion("jsonlite")),
  confidence_level = 0.95,
  mean_t_critical = as.list(setNames(qt(0.975, c(1, 2, 4, 9)), c(1, 2, 4, 9))),
  controls = controls, q_profile_control = qp_control,
  notes = c("24 deterministic datasets; not a simulation of coverage.",
            "QP empty flag calculated from Q(0), not a native metafor flag.",
            "Riley predictions deliberately omitted at k=2 (Python API boundary)."),
  cases = cases)
write_json(result, output, digits = 16, pretty = TRUE, auto_unbox = TRUE, null = "null")
cat("Wrote", length(cases), "cases to", output, "\n")
