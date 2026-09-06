# Regenerate trimfill_metafor.json with R 4.6.1, metafor 5.0-1, jsonlite 2.0.0.
# From the repository root:
# Rscript tests/reference/generate_trimfill_metafor.R [optional-output-path]
library(jsonlite)
library(metafor)

args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args) >= 1) args[[1]] else {
  "tests/reference/trimfill_metafor.json"
}

yi <- c(.10, .15, .18, .22, .24, .29, .33, .37, .95, 1.15)
vi <- c(.04, .035, .03, .028, .025, .022, .02, .018, .012, .01)

reml_control <- list(threshold=1e-10, maxiter=1000)
fill_maxiter <- 100
summarize_fill <- function(filled) {
  # trimfill's final augmented fit may use native default convergence controls.
  # Retain that output, and independently refit its returned rows at our stated
  # controls so precision comparisons test the same equation and tolerance.
  refit <- rma(filled$yi, filled$vi, method=filled$method, control=reml_control)
  list(
        k0=unname(filled$k0),
        estimate=unname(refit$beta),
        tau2=unname(refit$tau2),
        k0_standard_error=if (is.finite(filled$se.k0)) unname(filled$se.k0) else NULL,
        k0_pvalue=if (is.na(filled$p.k0)) NULL else unname(filled$p.k0),
        side=filled$side,
        effect=unname(filled$yi), variance=unname(filled$vi), imputed=unname(filled$fill),
        standard_error=unname(refit$se), ci=c(refit$ci.lb, refit$ci.ub),
        q=unname(refit$QE), i2=unname(refit$I2 / 100), h2=unname(refit$H2),
        native_refit=list(estimate=unname(filled$beta), tau2=unname(filled$tau2))
  )
}
generate_cases <- function(yi, vi) {
  cases <- list()
  for (method in c("FE", "REML")) {
    fit <- rma(yi, vi, method=method, control=reml_control)
    for (estimator in c("L0", "R0")) {
      for (side in c("left", "right", "auto")) {
        filled <- if (side == "auto") {
          trimfill(fit, estimator=estimator, maxiter=fill_maxiter)
        } else {
          trimfill(fit, side=side, estimator=estimator, maxiter=fill_maxiter)
        }
        key <- paste(tolower(method), estimator, side, sep="_")
        cases[[key]] <- summarize_fill(filled)
      }
    }
  }
  cases
}
cases <- generate_cases(yi, vi)
# Preserve input order in this tied example for auditing first-rank conventions.
tied_yi <- c(.5, -.5, 0, -.5, .5, 0, 1.5)
tied_vi <- c(.05, .02, .04, .02, .05, .04, .1)
tied_cases <- generate_cases(tied_yi, tied_vi)
identical <- tryCatch({
  summarize_fill(trimfill(rma(rep(.3, 5), rep(.1, 5), method="FE"),
                         side="left", estimator="R0", maxiter=fill_maxiter))
}, error=function(e) list(error=conditionMessage(e)))

write_json(
  list(generated_by="R metafor", r_version=R.version.string,
       metafor_version=as.character(packageVersion("metafor")),
       jsonlite_version=as.character(packageVersion("jsonlite")),
       iterative_control=list(reml=reml_control, trimfill_maxiter=fill_maxiter),
       yi=yi, vi=vi, cases=cases, tied=list(yi=tied_yi, vi=tied_vi, cases=tied_cases),
       identical_R0=identical),
  output, auto_unbox=TRUE, pretty=TRUE, digits=16,
  null="null"
)
