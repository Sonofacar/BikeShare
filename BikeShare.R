# Libraries
library(tidyverse)
library(tidymodels)
library(vroom)

# Read in the data
train_df <- vroom("train.csv")
test_df <- vroom("test.csv")

# Deal with small number of observations for weather=4
train_df[train_df$weather == 4, "weather"] <- 3
test_df[test_df$weather == 4, "weather"] <- 3

# Change the season and weather variables to a categorical variable
train_df["season"] <- train_df$season %>%
  as_factor()
train_df["weather"] <- train_df$weather %>%
  as_factor()
test_df["season"] <- test_df$season %>%
  as_factor()
test_df["weather"] <- test_df$weather %>%
  as_factor()

# Encode date variable into usable forms
train_df["hour"] <- train_df$datetime %>%
  hour()
train_df["hour_category"] <- 4
train_df[(10 >= train_df$hour) & (train_df$hour > 5), "hour_category"] <- 1
train_df[(15 >= train_df$hour) & (train_df$hour > 10), "hour_category"] <- 2
train_df[(19 >= train_df$hour) & (train_df$hour > 15), "hour_category"] <- 3
train_df["hour_category"] <- train_df$hour_category %>%
  as_factor()
test_df["hour"] <- test_df$datetime %>%
  hour()
train_df["weekday"] <- train_df$datetime %>%
  weekdays() %>%
  as_factor()
test_df["weekday"] <- test_df$datetime %>%
  weekdays() %>%
  as_factor()
test_df["hour_category"] <- 4
test_df[(10 >= test_df$hour) & (test_df$hour > 5), "hour_category"] <- 1
test_df[(15 >= test_df$hour) & (test_df$hour > 10), "hour_category"] <- 2
test_df[(19 >= test_df$hour) & (test_df$hour > 15), "hour_category"] <- 3
test_df["hour_category"] <- test_df$hour_category %>%
  as_factor()

#####################
# linear regression #
#####################

# Variable selection
tmp_df <- train_df[c(-1, -10, -11)]
full_model <- lm(log(count) ~ ., data = tmp_df)
base_model <- lm(log(count) ~ 1, data = tmp_df)
selected <- stats::step(base_model,
                        direction = "both",
                        scope = list(upper = full_model, lower = base_model))

# split the data into two models
casual_df <- train_df[c(-1, -6, -11, -12)]
registered_df <- train_df[c(-1, -6, -10, -12)]

# Various models
casual_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression") %>%
  fit(formula = sqrt(casual) ~ ., data = casual_df)
registered_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression") %>%
  fit(formula = sqrt(registered) ~ ., data = registered_df)
overall_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression") %>%
  fit(formula = log(count) ~ ., data = train_df[c(-1, -6, -10, -11)])

# Making predictions
casual_predictions <- predict(casual_model, new_data = test_df)
registered_predictions <- predict(registered_model, new_data = test_df)
combination <- casual_predictions$.pred^2 + registered_predictions$.pred^2
overall_predictions <- predict(overall_model, new_data = test_df)$.pred %>%
  exp()

# Output Kaggle submissions
combination_output <- tibble(datetime = as.character(format(test_df$datetime)),
                             count = combination)
overall_output <- tibble(datetime = as.character(format(test_df$datetime)),
                         count = overall_predictions)
vroom_write(combination_output, "separate_linear_regressions.csv", delim = ",")
vroom_write(overall_output, "linear_regression.csv", delim = ",")

