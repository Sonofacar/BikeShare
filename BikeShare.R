# Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)

# Read in the data
train_df_dirty <- vroom("train.csv") %>%
  select(-casual, -registered)
test_df_dirty <- vroom("test.csv")

# Data cleaning recipe
recipe <- recipe(count ~ ., data = train_df_dirty) %>%
  step_mutate(weather = weather %>%
                replace(weather == 4, 3) %>%
                as_factor()) %>%
  step_mutate(hour = hour(datetime)) %>%
  step_mutate(season = as_factor(season)) %>%
  step_mutate(rush_hour = hour %in% c(7, 8, 17, 18)) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.5)
poisson_recipe <- recipe
linear_recipe <- recipe %>%
  step_mutate(count = log(count))

# Models
linear_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")
poisson_model <- poisson_reg() %>%
  set_engine("glm") %>%
  set_mode("regression")

# Create workflows
linear_workflow <- workflow() %>%
  add_recipe(linear_recipe) %>%
  add_model(linear_model)
poisson_workflow <- workflow() %>%
  add_recipe(poisson_recipe) %>%
  add_model(poisson_model)

# Fit models
linear_fit <- fit(linear_workflow, data = train_df_dirty)
poisson_fit <- fit(poisson_workflow, data = train_df_dirty)

# Make predictions
linear_predictions <- predict(linear_fit, new_data = test_df_dirty)$.pred %>%
  exp()
poisson_predictions <- predict(poisson_fit, new_data = test_df_dirty)$.pred

# Write output
linear_output <- tibble(datetime = test_df_dirty$datetime %>%
                          format() %>%
                          as.character(),
                        count = linear_predictions)
poisson_output <- tibble(datetime = test_df_dirty$datetime %>%
                           format() %>%
                           as.character(),
                         count = poisson_predictions)
vroom_write(linear_output, "linear_regression.csv", delim = ",")
vroom_write(poisson_output, "poisson_regression.csv", delim = ",")

