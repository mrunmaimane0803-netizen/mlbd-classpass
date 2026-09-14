# ============================================================
# ClassPass Synthetic Data Generation - v1
# Machine Learning for Big Data, Group 5
#
# Purpose: Generate a subscriber-month panel dataset for
# analysis of subscription churn, with focus on credit-value
# frustration vs. general disengagement as churn drivers.
#
# Reproducibility: set.seed(42) below guarantees identical
# output on any machine.
# ============================================================

set.seed(42)

# ------------------------------------------------------------
# PARAMETERS (each with source or explicit assumption)
# ------------------------------------------------------------

n_users  <- 3000    # subscriber population
n_months <- 12      # observation period (one year)

# Plan tiers - ClassPass US public pricing, August 2026
# Source: classpass.com pricing page (verified via getpulsesignal.com)
plan_names   <- c("Starter", "Core", "Plus", "Premium")
plan_prices  <- c(55, 99, 139, 199)      # USD per month
plan_credits <- c(25, 48, 68, 100)       # credits per month
plan_probs   <- c(0.25, 0.40, 0.25, 0.10)  # market share (assumption)

# Credit cost per class (dynamic pricing)
# Off-peak: ~6 credits typical; Peak: ~15 credits typical
# Source: ClassPass dynamic-pricing model, documented in Techcrunch
# and Harvard Digital Initiative case
credits_per_offpeak <- 6
credits_per_peak    <- 15

# Rollover cap - unused credits carry to next month up to a limit
# Source: ClassPass general policy
rollover_cap_frac <- 0.5   # can roll up to 50% of monthly allocation

# Monthly churn baseline: 8%
# Sources: Peloton App 10-K reports 7.0% monthly; Churnkey benchmark 7-10%
baseline_churn_logit <- -3.0    # implies ~5% before other effects

# Seasonality multipliers (Jan..Dec)
# Feb-Mar spike from New Year resolution drop-off: 2-3x normal
# Source: fitness app churn benchmark literature
seasonality <- c(1.0, 2.5, 2.2, 1.2, 1.0, 1.3, 1.4, 1.0, 0.9, 0.9, 1.0, 1.1)

# Share of corporate/employer-benefit plans (assumption)
corporate_share <- 0.15

# Probability of a plan price increase in any given month (assumption)
price_increase_prob <- 0.02


# ------------------------------------------------------------
# STEP 1: Generate user-level attributes (one row per user)
# ------------------------------------------------------------

users <- data.frame(
  user_id      = 1:n_users,
  plan_tier    = sample(plan_names, n_users, replace = TRUE, prob = plan_probs),
  signup_month = sample(-24:0, n_users, replace = TRUE),  # some tenured, some new
  is_corporate = rbinom(n_users, 1, corporate_share),
  
  # Latent traits - not observable in real data, but we plant them
  # so we can later verify what our models recover
  intrinsic_engagement = rnorm(n_users, 0, 1),
  peak_preference      = rbeta(n_users, 2, 4)  # skewed toward off-peak
)

# Attach plan-derived attributes
users$credits_allocated <- plan_credits[match(users$plan_tier, plan_names)]
users$monthly_price     <- plan_prices[match(users$plan_tier, plan_names)]


# ------------------------------------------------------------
# STEP 2: Expand to a user-month panel
# ------------------------------------------------------------

panel <- expand.grid(user_id = users$user_id, month = 1:n_months)
panel <- merge(panel, users, by = "user_id")
panel <- panel[order(panel$user_id, panel$month), ]
panel$tenure_months <- panel$month - panel$signup_month


# ------------------------------------------------------------
# STEP 3: Generate monthly booking behaviour
# ------------------------------------------------------------

# Log-mean number of classes attempted per month
log_mu_bookings <- 1.4 +
  0.6 * panel$intrinsic_engagement +          # engaged users book more
  -0.02 * pmax(0, panel$tenure_months - 6) +  # mild decay after 6 months
  0.2 * panel$is_corporate +                  # corporate users book steadily
  log(seasonality[panel$month]) * 0.3

classes_desired <- rpois(nrow(panel), exp(log_mu_bookings))

# Split into peak vs off-peak based on user's preference
panel$peak_share <- pmin(0.9, pmax(0,
                                   panel$peak_preference + rnorm(nrow(panel), 0, 0.1)))
panel$classes_peak    <- rbinom(nrow(panel), classes_desired, panel$peak_share)
panel$classes_offpeak <- classes_desired - panel$classes_peak


# ------------------------------------------------------------
# STEP 4: Credits, utilization, rollover
# ------------------------------------------------------------

credits_needed <- panel$classes_offpeak * credits_per_offpeak +
  panel$classes_peak    * credits_per_peak

# Users can't spend more credits than they have available
panel$credits_used <- pmin(panel$credits_allocated, credits_needed)
panel$classes_booked <- panel$classes_offpeak + panel$classes_peak

# Adjust class counts if credits ran out (proportional scale-down)
insufficient <- credits_needed > panel$credits_allocated
if (any(insufficient)) {
  scale <- panel$credits_used[insufficient] / credits_needed[insufficient]
  panel$classes_offpeak[insufficient] <- floor(panel$classes_offpeak[insufficient] * scale)
  panel$classes_peak[insufficient]    <- floor(panel$classes_peak[insufficient]    * scale)
  panel$classes_booked[insufficient]  <- panel$classes_offpeak[insufficient] +
    panel$classes_peak[insufficient]
}

panel$credits_utilization <- panel$credits_used / panel$credits_allocated

# Credits rolled over to next month (capped)
unused <- panel$credits_allocated - panel$credits_used
panel$credits_rolled_over <- pmin(unused,
                                  panel$credits_allocated * rollover_cap_frac)


# ------------------------------------------------------------
# STEP 5: Value frustration proxy
#
# avg_credit_cost_per_class captures "am I paying more credits
# per class than I feel is fair?" - higher = more frustration
# ------------------------------------------------------------

panel$avg_credit_cost_per_class <- ifelse(
  panel$classes_booked > 0,
  panel$credits_used / panel$classes_booked,
  NA
)


# ------------------------------------------------------------
# STEP 6: Late cancellation fees and support tickets
# (documented ClassPass friction points)
# ------------------------------------------------------------

panel$late_cancel_fees <- rpois(nrow(panel), 0.15 * panel$classes_booked)

# Support tickets correlate with friction (fees, high credit cost)
support_rate <- 0.05 +
  0.02 * panel$late_cancel_fees +
  0.01 * pmax(0, panel$avg_credit_cost_per_class - 8)
support_rate[is.na(support_rate)] <- 0.05
panel$support_tickets <- rpois(nrow(panel), pmax(0, support_rate))


# ------------------------------------------------------------
# STEP 7: Price increase events
# ------------------------------------------------------------

panel$price_increase_last_period <- rbinom(nrow(panel), 1, price_increase_prob)


# ------------------------------------------------------------
# STEP 8: Churn (the target variable)
#
# Two mechanisms combine:
#   (a) VALUE FRUSTRATION - high credit cost per class, low
#       utilization, price increases, late fees
#   (b) GENERAL DISENGAGEMENT - low booking activity, short
#       tenure, low intrinsic engagement
#
# The analytical question is which mechanism dominates.
# ------------------------------------------------------------

# (a) Value frustration component
frustration <-
  0.20 * pmax(0, panel$avg_credit_cost_per_class - 8) +   # paying too much per class
  0.80 * as.numeric(panel$credits_utilization < 0.3) +    # paying for unused credits
  0.05 * panel$late_cancel_fees +
  0.70 * panel$price_increase_last_period

# (b) General disengagement component
disengagement <-
  -0.5 * panel$intrinsic_engagement +
  0.4 * as.numeric(panel$classes_booked <= 1) +
  0.3 * as.numeric(panel$tenure_months < 3) +
  -0.02 * pmin(panel$tenure_months, 24)   # tenure protective, capped at 2 years

# Structural protections
protection <- -1.2 * panel$is_corporate +
  log(seasonality[panel$month]) * 0.5

# Handle NA in frustration (users with 0 classes have NA avg_credit_cost)
frustration[is.na(frustration)] <- 0.5   # no bookings is itself a signal

churn_logit <- baseline_churn_logit + frustration + disengagement + protection
churn_prob  <- 1 / (1 + exp(-churn_logit))
panel$churned <- rbinom(nrow(panel), 1, churn_prob)

# Once a user churns, remove their subsequent months (they've left)
churn_running <- ave(panel$churned, panel$user_id, FUN = cumsum)
panel <- panel[churn_running <= 1, ]


# ------------------------------------------------------------
# STEP 9: Select final columns (drop latent traits)
# ------------------------------------------------------------

final <- panel[, c(
  "user_id", "month", "tenure_months", "plan_tier",
  "is_corporate", "monthly_price",
  "credits_allocated", "credits_used", "credits_utilization",
  "credits_rolled_over", "avg_credit_cost_per_class",
  "classes_booked", "classes_peak", "classes_offpeak",
  "late_cancel_fees", "support_tickets", "price_increase_last_period",
  "churned"
)]


# ------------------------------------------------------------
# STEP 10: Sanity checks against public benchmarks
# ------------------------------------------------------------

cat("=== Dataset validation ===\n")
cat("Rows:", nrow(final), " | Unique users:", length(unique(final$user_id)), "\n\n")

cat("Overall monthly churn rate:", round(mean(final$churned), 3),
    "  (benchmark 7-10%)\n")
cat("February churn rate:", round(mean(final$churned[final$month == 2]), 3),
    "  (benchmark ~2-3x baseline)\n")
cat("Corporate churn rate:", round(mean(final$churned[final$is_corporate == 1]), 3),
    "  (should be much lower than baseline)\n\n")

cat("Median classes/month:", median(final$classes_booked),
    "  (benchmark ~6 for break-even)\n")
cat("Mean credit utilization:", round(mean(final$credits_utilization), 3), "\n")
cat("Share of user-months with <30% utilization:",
    round(mean(final$credits_utilization < 0.3), 3),
    "  (a key frustration signal)\n\n")

cat("Churn rate | low utilization (<30%):",
    round(mean(final$churned[final$credits_utilization < 0.3]), 3), "\n")
cat("Churn rate | high utilization (>70%):",
    round(mean(final$churned[final$credits_utilization > 0.7]), 3), "\n")
cat("Churn rate | price increase this period:",
    round(mean(final$churned[final$price_increase_last_period == 1]), 3), "\n")


# ------------------------------------------------------------
# STEP 11: Write to disk
# ------------------------------------------------------------

write.csv(final, "classpass_subscribers.csv", row.names = FALSE)
cat("\nDataset written to:", file.path(getwd(), "classpass_subscribers.csv"), "\n")
