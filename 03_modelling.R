# ============================================================
# ClassPass - Part 2: Modelling
# Machine Learning for Big Data, Group 5
#
# Two models compared:
#   (a) Logistic regression - interpretable baseline
#   (b) Random forest - flexible comparison
#
# The analytical question is which subscribers are at high risk
# of cancelling, and how much of that risk is driven by credit
# value frustration versus general disengagement.
# ============================================================

library(tidyverse)
library(broom)
library(pROC)          # for AUC and ROC curves
library(randomForest)  # for the comparison model


# ------------------------------------------------------------
# STEP 1: Load the prepared data
# ------------------------------------------------------------

train_df <- readRDS("train_df.rds")
test_df  <- readRDS("test_df.rds")

cat("Train:", nrow(train_df), "rows\n")
cat("Test: ", nrow(test_df),  "rows\n\n")


# ------------------------------------------------------------
# STEP 2: Logistic regression baseline
#
# Note: credits_rolled_over is EXCLUDED because it correlates
# -0.81 with credits_utilization (see Part 1). Including both
# would create unstable coefficients (multicollinearity).
# ------------------------------------------------------------

# The convert-churned-to-numeric-01 trick because glm() with
# family = binomial needs 0/1, but our factor is Retained/Churned
train_df <- train_df |>
  mutate(churned_num = as.integer(churned == "Churned"))

test_df <- test_df |>
  mutate(churned_num = as.integer(churned == "Churned"))

# Class weights to handle the 7% imbalance
# Weight for "Churned" = (1 / churn rate) / 2, so churn events
# get roughly 7x the weight of retention events
churn_weight    <- (1 / mean(train_df$churned_num)) / 2
retain_weight   <- (1 / (1 - mean(train_df$churned_num))) / 2

train_df <- train_df |>
  mutate(weight = if_else(churned_num == 1, churn_weight, retain_weight))


# The baseline model
logit_model <- glm(
  churned_num ~
    tenure_months +
    credits_utilization +
    avg_credit_cost_per_class +
    classes_booked +
    late_cancel_fees +
    support_tickets +
    price_increase_last_period +
    plan_tier +
    is_corporate,
  data = train_df,
  family = quasibinomial(link = "logit"),
  weights = weight
)


# ------------------------------------------------------------
# STEP 3: Interpret the coefficients
# ------------------------------------------------------------

cat("=== LOGISTIC REGRESSION COEFFICIENTS ===\n")
tidy(logit_model) |>
  mutate(
    odds_ratio = exp(estimate),
    signif = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01  ~ "**",
      p.value < 0.05  ~ "*",
      p.value < 0.1   ~ ".",
      TRUE            ~ ""
    )
  ) |>
  select(term, estimate, odds_ratio, std.error, p.value, signif) |>
  print(n = Inf)

cat("\n=== MODEL FIT ===\n")
glance(logit_model) |> print()


# ------------------------------------------------------------
# STEP 4: Confidence intervals for key coefficients
# ------------------------------------------------------------

cat("\n=== 95% CONFIDENCE INTERVALS ===\n")
confint.default(logit_model) |> round(3) |> print()


# ------------------------------------------------------------
# STEP 5: Predictions on the test set
# ------------------------------------------------------------

test_df <- test_df |>
  mutate(
    logit_prob = predict(logit_model, newdata = test_df, type = "response"),
    logit_pred = as.integer(logit_prob > 0.5)
  )

# Confusion matrix at 0.5 threshold
cat("\n=== LOGISTIC: CONFUSION MATRIX (threshold = 0.5) ===\n")
conf_logit <- table(
  Predicted = test_df$logit_pred,
  Actual    = test_df$churned_num
)
print(conf_logit)

# Metrics
tp <- conf_logit["1", "1"]
fp <- conf_logit["1", "0"]
fn <- conf_logit["0", "1"]
tn <- conf_logit["0", "0"]

precision <- tp / (tp + fp)
recall    <- tp / (tp + fn)
f1        <- 2 * precision * recall / (precision + recall)
accuracy  <- (tp + tn) / sum(conf_logit)

cat("\nLogistic model performance:\n")
cat("  Accuracy: ", round(accuracy, 3), "\n")
cat("  Precision:", round(precision, 3), "\n")
cat("  Recall:   ", round(recall, 3), "\n")
cat("  F1:       ", round(f1, 3), "\n")

# AUC
roc_logit <- roc(test_df$churned_num, test_df$logit_prob, quiet = TRUE)
cat("  AUC:      ", round(auc(roc_logit), 3), "  (benchmark 0.75-0.85)\n")


# ------------------------------------------------------------
# STEP 6: Random forest comparison
# ------------------------------------------------------------

set.seed(42)

rf_model <- randomForest(
  factor(churned_num) ~
    tenure_months +
    credits_utilization +
    avg_credit_cost_per_class +
    classes_booked +
    late_cancel_fees +
    support_tickets +
    price_increase_last_period +
    plan_tier +
    is_corporate,
  data = train_df,
  ntree = 500,
  mtry = 3,
  classwt = c("0" = 1, "1" = 7),   # weight churn class 7x
  importance = TRUE
)

cat("\n=== RANDOM FOREST ===\n")
print(rf_model)

# Test set predictions
test_df <- test_df |>
  mutate(
    rf_prob = predict(rf_model, newdata = test_df, type = "prob")[, "1"],
    rf_pred = as.integer(rf_prob > 0.5)
  )

# Confusion matrix
cat("\n=== RANDOM FOREST: CONFUSION MATRIX (threshold = 0.5) ===\n")
conf_rf <- table(
  Predicted = test_df$rf_pred,
  Actual    = test_df$churned_num
)
print(conf_rf)

tp_rf <- conf_rf["1", "1"]
fp_rf <- conf_rf["1", "0"]
fn_rf <- conf_rf["0", "1"]
tn_rf <- conf_rf["0", "0"]

precision_rf <- tp_rf / (tp_rf + fp_rf)
recall_rf    <- tp_rf / (tp_rf + fn_rf)
f1_rf        <- 2 * precision_rf * recall_rf / (precision_rf + recall_rf)
accuracy_rf  <- (tp_rf + tn_rf) / sum(conf_rf)

cat("\nRandom forest performance:\n")
cat("  Accuracy: ", round(accuracy_rf, 3), "\n")
cat("  Precision:", round(precision_rf, 3), "\n")
cat("  Recall:   ", round(recall_rf, 3), "\n")
cat("  F1:       ", round(f1_rf, 3), "\n")

roc_rf <- roc(test_df$churned_num, test_df$rf_prob, quiet = TRUE)
cat("  AUC:      ", round(auc(roc_rf), 3), "\n")

# Variable importance
cat("\n=== RANDOM FOREST: VARIABLE IMPORTANCE ===\n")
importance(rf_model) |>
  as.data.frame() |>
  rownames_to_column("variable") |>
  arrange(desc(MeanDecreaseGini)) |>
  print()


# ------------------------------------------------------------
# STEP 7: Compare the two models side-by-side
# ------------------------------------------------------------

comparison <- tibble(
  Metric    = c("Accuracy", "Precision", "Recall", "F1", "AUC"),
  Logistic  = c(accuracy, precision, recall, f1, as.numeric(auc(roc_logit))),
  RandomForest = c(accuracy_rf, precision_rf, recall_rf, f1_rf, as.numeric(auc(roc_rf)))
) |>
  mutate(across(where(is.numeric), \(x) round(x, 3)))

cat("\n=== MODEL COMPARISON ===\n")
print(comparison)


# ------------------------------------------------------------
# STEP 8: ROC curves - the headline evaluation plot
# ------------------------------------------------------------

roc_data <- bind_rows(
  tibble(
    fpr = 1 - roc_logit$specificities,
    tpr = roc_logit$sensitivities,
    model = "Logistic regression"
  ),
  tibble(
    fpr = 1 - roc_rf$specificities,
    tpr = roc_rf$sensitivities,
    model = "Random forest"
  )
)

ggplot(roc_data, aes(x = fpr, y = tpr, colour = model)) +
  geom_line(linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "grey60") +
  labs(
    title = "ROC curves - logistic regression vs. random forest",
    x = "False positive rate",
    y = "True positive rate",
    colour = "Model"
  ) +
  theme_minimal()


# ------------------------------------------------------------
# STEP 9: Subgroup analysis - do different plan tiers churn
# for different reasons?  (Session 4, Part D style)
# ------------------------------------------------------------

cat("\n=== SUBGROUP: Logistic regression BY PLAN TIER ===\n")

plan_tiers <- levels(train_df$plan_tier)
subgroup_results <- list()

for (tier in plan_tiers) {
  df_sub <- train_df |> filter(plan_tier == tier)
  
  # Skip if too few churn events for a stable model
  if (sum(df_sub$churned_num) < 30) {
    cat("\n---", tier, "--- SKIPPED (too few churn events) ---\n")
    next
  }
  
  model_sub <- glm(
    churned_num ~
      tenure_months +
      credits_utilization +
      avg_credit_cost_per_class +
      classes_booked +
      late_cancel_fees +
      is_corporate,
    data = df_sub,
    family = quasibinomial(link = "logit"),
    weights = weight
  )
  
  subgroup_results[[tier]] <- model_sub
  
  cat("\n---", tier, "(n =", nrow(df_sub),
      ", churn events =", sum(df_sub$churned_num), ") ---\n")
  tidy(model_sub) |>
    select(term, estimate, p.value) |>
    mutate(across(where(is.numeric), \(x) round(x, 3))) |>
    print(n = Inf)
}


# ------------------------------------------------------------
# STEP 10: Save models for Part 3 (evaluation & deployment)
# ------------------------------------------------------------

saveRDS(logit_model,       "logit_model.rds")
saveRDS(rf_model,          "rf_model.rds")
saveRDS(test_df,           "test_df_scored.rds")
saveRDS(subgroup_results,  "subgroup_models.rds")

cat("\nPart 2 complete. Models saved.\n")
