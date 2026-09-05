# Regenerate begg_ranktest_metafor.json from the repository root with:
# Rscript tests/reference/generate_begg_ranktest_metafor.R [optional-output-path]

library(jsonlite)
library(metafor)

args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args) >= 1) args[[1]] else {
  "tests/reference/begg_ranktest_metafor.json"
}
input <- read.csv("tests/reference/small_study_effects_input.csv")

test <- ranktest(input$effect, input$variance)
weights <- 1 / input$variance
center <- weighted.mean(input$effect, weights)
pooled_variance <- 1 / sum(weights)
standardized_effect <- (input$effect - center) / sqrt(input$variance - pooled_variance)
kendall <- cor.test(
  standardized_effect,
  input$variance,
  method = "kendall",
  exact = TRUE
)

reference <- list(
  generated_by = "R metafor",
  r_version = R.version.string,
  metafor_version = as.character(packageVersion("metafor")),
  jsonlite_version = as.character(packageVersion("jsonlite")),
  method = "ranktest",
  correlation_method = "Kendall's tau-b",
  inference_method = "exact",
  k = nrow(input),
  tau = unname(test$tau),
  statistic = round(unname(test$tau) * nrow(input) * (nrow(input) - 1) / 2),
  pvalue = unname(test$pval)
)

stopifnot(isTRUE(all.equal(unname(test$tau), unname(kendall$estimate))))
stopifnot(isTRUE(all.equal(unname(test$pval), unname(kendall$p.value))))

tied_variance <- input$variance
tied_variance[[2]] <- tied_variance[[1]]
tied_test <- ranktest(input$effect, tied_variance)
tied_weights <- 1 / tied_variance
tied_center <- weighted.mean(input$effect, tied_weights)
tied_pooled_variance <- 1 / sum(tied_weights)
tied_response <- (
  (input$effect - tied_center) /
    sqrt(tied_variance - tied_pooled_variance)
)
tied_kendall <- suppressWarnings(cor.test(
  tied_response,
  tied_variance,
  method = "kendall",
  exact = FALSE
))
tied_corrected <- suppressWarnings(cor.test(
  tied_response,
  tied_variance,
  method = "kendall",
  exact = FALSE,
  continuity = TRUE
))
reference$tied_asymptotic <- list(
  input_change = "variance[2] = variance[1]",
  tau = unname(tied_test$tau),
  statistic = unname(qnorm(tied_kendall$p.value / 2, lower.tail = FALSE)) *
    sign(unname(tied_test$tau)),
  pvalue = unname(tied_test$pval)
)
reference$tied_continuity_corrected <- list(
  statistic = unname(qnorm(tied_corrected$p.value / 2, lower.tail = FALSE)) *
    sign(unname(tied_corrected$estimate)),
  pvalue = unname(tied_corrected$p.value)
)

stopifnot(isTRUE(all.equal(unname(tied_test$tau), unname(tied_kendall$estimate))))
stopifnot(isTRUE(all.equal(unname(tied_test$pval), unname(tied_kendall$p.value))))

write_json(
  reference,
  output,
  auto_unbox = TRUE,
  digits = 16,
  pretty = TRUE
)
