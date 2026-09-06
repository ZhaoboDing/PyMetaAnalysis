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

cases <- list()
for (method in c("FE", "REML")) {
  fit <- rma(yi, vi, method=method)
  for (estimator in c("L0", "R0")) {
    for (side in c("left", "right")) {
      filled <- trimfill(fit, side=side, estimator=estimator)
      key <- paste(tolower(method), estimator, side, sep="_")
      cases[[key]] <- list(
        k0=unname(filled$k0),
        estimate=unname(filled$beta),
        tau2=unname(filled$tau2),
        k0_standard_error=unname(filled$se.k0),
        k0_pvalue=if (is.na(filled$p.k0)) NULL else unname(filled$p.k0)
      )
    }
  }
}

write_json(
  list(generated_by="R metafor", r_version=R.version.string,
       metafor_version=as.character(packageVersion("metafor")),
       jsonlite_version=as.character(packageVersion("jsonlite")),
       iterative_control=list(tolerance=1e-10, max_iterations=1000),
       yi=yi, vi=vi, cases=cases),
  output, auto_unbox=TRUE, pretty=TRUE, digits=16,
  null="null"
)
