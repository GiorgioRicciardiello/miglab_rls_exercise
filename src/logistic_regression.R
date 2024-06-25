# Load necessary libraries
library(readxl)
library(openxlsx)
library(ggplot2)

# Define working directory and path to the data
work_dir <- here::here()

datasets <- c("dataset_non_vs_all.xlsx", "dataset_pos_vs_neg.xlsx")

for (dataset_ in datasets) {
  # dataset_ = "dataset_pos_vs_neg.xlsx"
  name_dataset = unlist(strsplit(dataset_, "\\."))[1]
  path_data <- file.path(work_dir, 'projects', 'rls_regression', dataset_)
  out_name <- paste0('odds_ratios_', dataset_)
  path_output_odds <- file.path(work_dir, 'projects', 'rls_regression', out_name)
  path_output_summary <- file.path(work_dir, 'projects', 'rls_regression', paste(name_dataset, "model_summary.txt", sep = ""))
  path_output_odds_plot <- file.path(work_dir, 'projects', 'rls_regression',  name_dataset)
  
  # Read the data
  df_data <- read_excel(path_data)
  
  # Define features and target variable
  features <-c(
    'dem_age_zscores',
    'dem_sex',
    # 'dem_race',
    'rls_bilateral',
    'secondary_conditions',
    'plm_presence',
    'bmi_zscores',
    'rls_duration_zscores',
    'rls_severity_zscores',
    'ipaq_total',
    'ipaq_sit_hrs_zscores'
  )
  target <- 'response_class'
  
  # Ensure the target variable is treated as a binary factor
  df_data[[target]] <- factor(df_data[[target]])
  
  # Formula for the model
  formula <- as.formula(paste(target, "~", paste(features, collapse = "+")))
  
  # Run logistic regression
  model <- glm(formula, data = df_data, family = binomial)
  
  # Summary of the model
  summary(model)
  
  # save model
  model_summary <- capture.output(summary(model))
  writeLines(model_summary, path_output_summary)
  
  
  # Coefficients
  coefficients <- coef(model)
  std_errors <- summary(model)$coefficients[, "Std. Error"]
  z_values <-  summary(model)$coefficients[, "z value"] 
  p_values <- summary(model)$coefficients[, "Pr(>|z|)"]
  
  # Lower bound of the 95% CI
  lower_bound <- coefficients - (1.96 * std_errors)
  
  # Upper bound of the 95% CI
  upper_bound <- coefficients + (1.96 * std_errors)
  
  results_df <- data.frame(coefficients, std_errors, lower_bound, upper_bound, p_values)
  print(results_df)
  
  
  # Convert coefficients to odds ratios
  odds_ratios <- results_df  # Copy the original dataframe to retain structure
  odds_ratios$odds_ratios <- exp(results_df$coefficients)
  odds_ratios$lower_bound <- exp(results_df$lower_bound)
  odds_ratios$upper_bound <- exp(results_df$upper_bound)
  odds_ratios <- odds_ratios[, c("odds_ratios", "lower_bound", "upper_bound", 'p_values')]

  odds_ratios$feature <- names(coefficients) 
 
  # Plotting odds ratios
  ### plot odds ratio 
  odds_plot <- ggplot(odds_ratios, aes(x = odds_ratios, y = feature)) +
    geom_vline(xintercept = 1, size = 0.25, linetype = "dashed", color = "blue") +  
    geom_errorbarh(aes(xmin = lower_bound, xmax = upper_bound, y = feature), height = 0.2, color = "gray50") +  
    geom_point(aes(y = feature), size = 3.5, color = "orange") +  
    scale_x_continuous(name = "Odds Ratio", 
                       breaks = seq(0, max(odds_ratios$upper_bound, na.rm = TRUE) + 0.5, by = 0.5), 
                       limits = c(0, max(odds_ratios$upper_bound, na.rm = TRUE) + 1)) +  
    coord_cartesian(xlim = c(0, max(odds_ratios$upper_bound, na.rm = TRUE) + 1)) +  
    theme_minimal() +
    labs(y = "", x = "Odds Ratio") +
    ggtitle("Odds Ratios and Confidence Intervals for Features")
  
  
  ggsave(paste0(path_output_odds_plot, ".png"), plot = odds_plot)
  # Save the results to Excel
  write.xlsx(odds_ratios, file = paste0(path_output_odds))
}
