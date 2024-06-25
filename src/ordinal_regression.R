# Load necessary libraries
library(readxl)
library(MASS)
library(here)  
library(openxlsx)

# Define working directory and path to the data
work_dir <- here::here()
path_data <- file.path(work_dir, 'projects', 'rls_regression', "reduced_dataset.xlsx")

# Read the data
df_data <- read_excel(path_data)

# Define features and target variable
features <- c('dem_age',
              'dem_sex',
              'dem_race',
              'bmi',
              'rls_duration',
              'rls_severity',
              'rls_bilateral',
              'rls_pregnancy',
              'rls_treated',
              'rls_med_responsive',
              'rls_med_frequency',
              'secondary_conditions',
              'plm_presence',
              'ipaq_vig',
              'ipaq_mod',
              'ipaq_walk',
              'ipaq_6a',
              'ipaq_total',
              'ipaq_sit_hrs')
target <- 'response_class'

# Ensure the target variable is treated as an ordered factor
df_data[[target]] <- factor(df_data[[target]], ordered = TRUE)

# Formula for the model
formula <- as.formula(paste(target, "~", paste(features, collapse = "+")))

# Run ordinal regression
model <- polr(formula, 
              data = df_data, 
              Hess = TRUE, 
              method=c('logistic'))

# Summary of the model
summary(model)

model_summary <- summary(model)
coefficients <- model_summary$coefficients[, "Value"]
std_errors <- model_summary$coefficients[, "Std. Error"]
z_values <-  model_summary$coefficients[, "t value"] 
p_values <- 2 * pnorm(abs(z_values), lower.tail = FALSE)
# Lower bound of the 95% CI
lower_bound <- coefficients - (1.96 * std_errors)

# Upper bound of the 95% CI
upper_bound <- coefficients + (1.96 * std_errors)
results_df <- data.frame(coefficients, std_errors, lower_bound, upper_bound, p_values)
print(results_df)


# convert to odds ratios
odds_ratios <- results_df  # Copy the original dataframe to retain structure
odds_ratios$odds_ratios <- exp(results_df$coefficients)
odds_ratios$lower_bound <- exp(results_df$lower_bound)
odds_ratios$upper_bound <- exp(results_df$upper_bound)
odds_ratios <- odds_ratios[, c("odds_ratios", "lower_bound", "upper_bound", 'p_value')]

odds_ratios$feature <- rownames(odds_ratios)  # or use an existing column

# Dropping rows with index names "0|1" and "1|2"
odds_ratios <- odds_ratios[!rownames(odds_ratios) %in% c("0|1", "1|2"), ]



### plot odds ratio 
ggplot(odds_ratios, aes(x = odds_ratios, y = feature)) +
  geom_vline(xintercept = 1, size = 0.25, linetype = "dashed", color = "blue") +  # Line at odds ratio of 1
  geom_errorbarh(aes(xmin = lower_bound, xmax = upper_bound), height = 0.2, color = "gray50") +  # CI
  geom_point(size = 3.5, color = "orange") +  # Odds ratio points
  scale_x_continuous(name = "Odds Ratio", 
                     breaks = seq(0, max(odds_ratios$upper_bound, na.rm = TRUE) + 0.5, by = 0.5), 
                     limits = c(0, max(odds_ratios$upper_bound, na.rm = TRUE) + 1)) +  # Adjusted x-axis
  coord_cartesian(xlim = c(0, max(odds_ratios$upper_bound, na.rm = TRUE) + 1)) +  # To avoid negative odds showing
  theme_minimal() +
  labs(y = "", x = "Odds Ratio") +
  ggtitle("Odds Ratios and Confidence Intervals for Features")


### plot odds ratio in the log scale
odds_ratios$log_odds <- log(odds_ratios$odds_ratios)
odds_ratios$log_lower_bound <- log(odds_ratios$lower_bound)
odds_ratios$log_upper_bound <- log(odds_ratios$upper_bound)

max_log_upper_bound <- max(odds_ratios$log_upper_bound, na.rm = TRUE)
min_log_lower_bound <- min(odds_ratios$log_lower_bound, na.rm = TRUE)

ggplot(odds_ratios, aes(x = log_odds, y = feature)) +
  geom_vline(xintercept = 0, size = 0.25, linetype = "dashed") +
  geom_errorbarh(aes(xmin = log_lower_bound, xmax = log_upper_bound), height = 0.2, color = "gray50") +
  geom_point(size = 3.5, color = "orange") +
  scale_x_continuous(name = "Log Odds Ratio",
                     breaks = seq(min_log_lower_bound, max_log_upper_bound, length.out = 10),  # dynamic breaks
                     labels = round(seq(min_log_lower_bound, max_log_upper_bound, length.out = 10), 2)) +
  theme_bw() +
  theme(panel.grid.minor = element_blank()) +
  labs(y = "", x = "Log Odds Ratio") +
  ggtitle("Odds Ratios and Confidence Intervals for Features")



# save the excel 
write.xlsx(odds_ratios, file = 'odds_ratios.xlsx')
