# ClassPass - Part 1: Data understanding & preparation
# Machine Learning for Big Data, Group 5


library(tidyverse)
library(broom)   # tidy() and glance() for later

# STEP 1: Load and inspect


df <- read_csv(
  "https://raw.githubusercontent.com/mrunmaimane0803-netizen/mlbd-classpass/main/classpass_subscribers.csv",
  show_col_types = FALSE
)

# Convert categorical variables to factors with explicit reference levels
# Reference = "Core" (largest plan tier) so coefficients are relative to it
df <- df |>
  mutate(
    plan_tier = factor(plan_tier,
                       levels = c("Core", "Starter", "Plus", "Premium")),
    is_corporate = factor(is_corporate,
                          levels = c(0, 1),
                          labels = c("Individual", "Corporate")),
    churned = factor(churned,
                     levels = c(0, 1),
                     labels = c("Retained", "Churned"))
  )

glimpse(df)



# STEP 2: Univariate EDA


# Summary statistics for all variables
summary(df)

# Numeric variable distributions
df |>
  select(tenure_months, credits_utilization, classes_booked,
         avg_credit_cost_per_class, late_cancel_fees, support_tickets) |>
  pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = "value"
  ) |>
  ggplot(aes(x = value)) +
  geom_histogram(bins = 30) +
  facet_wrap(~ variable, scales = "free") +
  labs(
    title = "Distributions of key numeric variables",
    x = NULL,
    y = "Count"
  )

# Categorical variables: frequencies
table(df$plan_tier)
table(df$is_corporate)
table(df$churned)      # class imbalance check



# STEP 3: Class imbalance


# What proportion of user-months are churn events?
prop.table(table(df$churned))

# Category A finding: churn is a minority class (~7%)
# We will handle this in modelling via class weights or resampling



# STEP 4: Multivariate EDA - churn by segment


# Churn rate by plan tier
df |>
  group_by(plan_tier) |>
  summarise(
    n = n(),
    churn_rate = mean(churned == "Churned"),
    .groups = "drop"
  )

# Churn rate by corporate status
df |>
  group_by(is_corporate) |>
  summarise(
    n = n(),
    churn_rate = mean(churned == "Churned"),
    .groups = "drop"
  )

# Churn rate by month (seasonality)
df |>
  group_by(month) |>
  summarise(
    churn_rate = mean(churned == "Churned"),
    .groups = "drop"
  ) |>
  ggplot(aes(x = month, y = churn_rate)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks = 1:12) +
  labs(
    title = "Monthly churn rate — seasonal pattern",
    x = "Month",
    y = "Churn probability"
  )



# STEP 5: The U-shape - our key mechanism plot


df <- df |>
  mutate(
    util_bucket = cut(credits_utilization,
                      breaks = c(-0.01, 0.3, 0.7, 1.01),
                      labels = c("Low (<30%)",
                                 "Mid (30–70%)",
                                 "High (>70%)"))
  )

df |>
  group_by(util_bucket) |>
  summarise(
    n = n(),
    churn_rate = mean(churned == "Churned"),
    .groups = "drop"
  ) |>
  ggplot(aes(x = util_bucket, y = churn_rate)) +
  geom_col(fill = c("tomato", "steelblue", "tomato")) +
  labs(
    title = "Churn rate by credit utilisation — the value-frustration U-shape",
    x = "Credit utilisation bucket",
    y = "Monthly churn probability"
  )



# STEP 6: Correlations among numeric predictors


num_vars <- df |>
  select(tenure_months, credits_utilization, classes_booked,
         avg_credit_cost_per_class, late_cancel_fees, support_tickets,
         credits_rolled_over)

cor(num_vars, use = "pairwise.complete.obs") |> round(2)



# STEP 7: Feature engineering


df <- df |>
  # Handle NA in avg_credit_cost_per_class (users who booked nothing)
  mutate(
    avg_credit_cost_per_class = replace_na(avg_credit_cost_per_class, 0),
    
    # Tenure buckets — early tenure is the highest-risk period
    tenure_bucket = case_when(
      tenure_months < 3  ~ "New (<3mo)",
      tenure_months < 12 ~ "Established (3-12mo)",
      TRUE               ~ "Loyal (>12mo)"
    ),
    tenure_bucket = factor(tenure_bucket,
                           levels = c("Established (3-12mo)",
                                      "New (<3mo)",
                                      "Loyal (>12mo)")),
    
    # A single "engagement" flag
    zero_booking_month = as.integer(classes_booked == 0)
  )



# STEP 8: Train/test split — BY USER, not by row
# (avoids data leakage across the panel)


set.seed(42)

user_ids <- unique(df$user_id)
train_users <- sample(user_ids, size = 0.7 * length(user_ids))

train_df <- df |> filter(user_id %in% train_users)
test_df  <- df |> filter(!user_id %in% train_users)

cat("Train:", nrow(train_df), "rows,", length(unique(train_df$user_id)), "users\n")
cat("Test: ", nrow(test_df),  "rows,", length(unique(test_df$user_id)),  "users\n")
cat("Train churn rate:", round(mean(train_df$churned == "Churned"), 3), "\n")
cat("Test churn rate: ", round(mean(test_df$churned  == "Churned"), 3), "\n")



# STEP 9: Save the prepared data for Part 2


saveRDS(train_df, "train_df.rds")
saveRDS(test_df,  "test_df.rds")

cat("\nPart 1 complete. Prepared data saved.\n")

