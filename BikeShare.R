# Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)

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
train_df["weekday"] <- train_df$datetime %>%
  weekdays() %>%
  as_factor()
train_df["rush_hour"] <- train_df$hour %in% c(7, 8, 17, 18)
test_df["hour"] <- test_df$datetime %>%
  hour()
test_df["hour_category"] <- 4
test_df[(10 >= test_df$hour) & (test_df$hour > 5), "hour_category"] <- 1
test_df[(15 >= test_df$hour) & (test_df$hour > 10), "hour_category"] <- 2
test_df[(19 >= test_df$hour) & (test_df$hour > 15), "hour_category"] <- 3
test_df["hour_category"] <- test_df$hour_category %>%
  as_factor()
test_df["weekday"] <- test_df$datetime %>%
  weekdays() %>%
  as_factor()
test_df["rush_hour"] <- test_df$hour %in% c(7, 8, 17, 18)

#####################
# linear regression #
#####################

# Variable selection
tmp_df <- train_df[c(-1, -10, -11, -13)]
full_model <- lm(log(count) ~ . +
                   rush_hour * holiday +
                   weekday * hour_category,
                 data = tmp_df)
base_model <- lm(log(count) ~ 1, data = tmp_df)
selected <- stats::step(full_model,
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
  fit(formula = log(count) ~ . +
        rush_hour * holiday +
        weekday * hour_category, data = train_df[c(-1, -6, -10, -11)])

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

######################
# Poisson regression #
######################

# Variable selection
tmp_df <- train_df[c(-1, -11, -12, -13)]
full_casual_model <- glm(casual ~ . +
                           rush_hour * holiday +
                           weekday * hour_category,
                         data = tmp_df,
                         family = poisson())
base_casual_model <- glm(casual ~ 1,
                         data = tmp_df,
                         family = poisson())
scope <- list(upper = full_casual_model, lower = base_casual_model)
casual_poisson_selected <- stats::step(full_casual_model,
                                       direction = "both",
                                       scope = scope)
tmp_df <- train_df[c(-1, -10, -12, -13)]
full_registered_model <- glm(registered ~ . +
                               rush_hour * holiday +
                               weekday * hour_category,
                             data = tmp_df,
                             family = poisson())
base_registered_model <- glm(registered ~ 1,
                             data = tmp_df,
                             family = poisson())
scope <- list(upper = full_registered_model, lower = base_registered_model)
registered_poisson_selected <- stats::step(full_registered_model,
                                           direction = "both",
                                           scope = scope)
tmp_df <- train_df[c(-1, -10, -11, -13)]
full_model <- glm(count ~ . +
                    rush_hour * holiday +
                    weekday * hour_category,
                  data = tmp_df,
                  family = poisson())
base_model <- glm(count ~ 1,
                  data = tmp_df,
                  family = poisson())
scope <- list(upper = full_model, lower = base_model)
poisson_selected <- stats::step(full_model,
                                direction = "both",
                                scope = scope)

# Make models
casual_model <- poisson_reg() %>%
  set_engine("glm") %>%
  set_mode("regression") %>%
  fit(formula = casual_poisson_selected$formula, data = train_df)
registered_model <- poisson_reg() %>%
  set_engine("glm") %>%
  set_mode("regression") %>%
  fit(formula = registered_poisson_selected$formula, data = train_df)
overall_model <- poisson_reg() %>%
  set_engine("glm") %>%
  set_mode("regression") %>%
  fit(formula = poisson_selected$formula, data = train_df)

# Make predictions
casual_predictions <- predict(casual_model, new_data = test_df)$.pred
registered_predictions <- predict(registered_model, new_data = test_df)$.pred
combination <- casual_predictions + registered_predictions
overall_predictions <- predict(overall_model, new_data = test_df)$.pred
no_select_poisson_predictions <- predict(full_model, new_data = test_df)

# Organize and write output
combination_output <- tibble(datetime = as.character(format(test_df$datetime)),
                             count = combination)
overall_output <- tibble(datetime = as.character(format(test_df$datetime)),
                         count = overall_predictions)
no_select_poisson_output <- tibble(datetime = test_df$datetime %>%
                                     format() %>%
                                     as.character(),
                                   count = overall_predictions)
vroom_write(combination_output, "separate_poisson_regression.csv", delim = ",")
vroom_write(overall_output, "poisson_regression.csv", delim = ",")
vroom_write(no_select_poisson_output,
            "no_select_poisson_regression.csv",
            delim = ",")

