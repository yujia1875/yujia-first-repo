# 7. Design-based Supervised Learning (DSL)
# 7.1 Install and load dsl R package and Pan & Chen’s (2018) data

if (!requireNamespace("dsl", quietly = TRUE)) {
  if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools")
  devtools::install_github("naoki-egami/dsl", dependencies = TRUE)
}
library(dsl)
library(dplyr)
library(ggplot2)

install.packages("gam")
library(gam)
devtools::install_github("naoki-egami/dsl", dependencies = TRUE)

data("PanChen")
glimpse(PanChen[, c("SendOrNot", "countyWrong", "pred_countyWrong")])


# Step 1 — the naive regression (what a researcher who ignores annotation error would run):
m_naive <- glm(SendOrNot ~ pred_countyWrong + prefecWrong + connect2b +
                 prevalence + regionj + groupIssue,
               data = PanChen, family = binomial())
summary(m_naive)

# Step 2 — the DSL-corrected regression:
m_dsl <- dsl(
  model         = "logit",
  formula       = SendOrNot ~ countyWrong + prefecWrong + connect2b +
    prevalence + regionj + groupIssue,
  predicted_var = "countyWrong",         # the variable that needed annotation
  prediction    = "pred_countyWrong",    # the LLM's guess
  data          = PanChen,
  cross_fit     = 5,                     # 5-fold cross-fit for the ML gap model
  sample_split  = 10,                    # 10 sample splits, averaged, for stability
  seed          = 2025
)
summary(m_dsl)

# Compare the coefficient on wrongdoing side by side:
cmp <- tibble(
  method   = c("Naive (LLM label as truth)", "DSL (bias-corrected)"),
  estimate = c(coef(m_naive)["pred_countyWrong"],
               m_dsl$coefficients["countyWrong"]),
  se       = c(sqrt(diag(vcov(m_naive)))["pred_countyWrong"],
               m_dsl$standard_errors["countyWrong"])
)

ggplot(cmp, aes(method, estimate)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = estimate - 1.96 * se,
                    ymax = estimate + 1.96 * se), width = 0.15) +
  labs(x = NULL, y = "Effect of county wrongdoing (log-odds)",
       title = "Pan & Chen: DSL vs. Naive")

# 7.3 Power analysis: how many more expert labels do I need?
pwr <- power_dsl(dsl_out = m_dsl, labeled_size = seq(500, 1200, by = 100))
summary(pwr)
plot(pwr, coef_name = "countyWrong")


