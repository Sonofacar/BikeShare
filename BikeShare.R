# Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)

# Read in the data
train_df_dirty <- vroom("train.csv") %>%
  select(-casual, -registered)
log_train_df_dirty <- train_df_dirty %>%
  mutate(count = log(count))
test_df_dirty <- vroom("test.csv")

# Data cleaning recipe
recipe <- recipe(count ~ ., train_df_dirty) %>%
  step_mutate(weather = weather %>%
                replace(weather == 4, 3) %>%
                as_factor()) %>%
  step_time(datetime, features = c("hour")) %>%
  step_rename(hour = datetime_hour) %>%
  step_mutate(rush_hour = as.integer(hour %in% c(7, 8, 17, 18))) %>%
  step_poly(hour, degree = 3) %>%
  step_mutate(season = as_factor(season)) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.5)
penalized_recipe <- recipe %>%
  step_normalize(all_numeric_predictors())
prepped_recipe <- prep(recipe)
clean_data <- bake(prepped_recipe, new_data = train_df_dirty)

# Models
linear_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")
poisson_model <- poisson_reg() %>%
  set_engine("glm") %>%
  set_mode("regression")
penalized_model <- linear_reg(penalty = 0.1, mixture = 0.5) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

# Create workflows
linear_workflow <- workflow() %>%
  add_model(linear_model) %>%
  add_recipe(recipe)
poisson_workflow <- workflow() %>%
  add_model(poisson_model) %>%
  add_recipe(recipe)
penalized_workflow <- workflow() %>%
  add_model(penalized_model) %>%
  add_recipe(penalized_recipe)

# Fit models
linear_fit <- fit(linear_workflow, data = log_train_df_dirty)
poisson_fit <- fit(poisson_workflow, data = train_df_dirty)
penalized_fit <- fit(penalized_workflow, data = log_train_df_dirty)

# Make predictions
linear_predictions <- predict(linear_fit, new_data = test_df_dirty)$.pred %>%
  exp()
poisson_predictions <- predict(poisson_fit, new_data = test_df_dirty)$.pred
penalized_predictions <- predict(penalized_fit, new_data = test_df_dirty)$.pred


# Write output
linear_output <- tibble(datetime = test_df_dirty$datetime %>%
                          format() %>%
                          as.character(),
                        count = linear_predictions)
poisson_output <- tibble(datetime = test_df_dirty$datetime %>%
                           format() %>%
                           as.character(),
                         count = poisson_predictions)
penalized_output <- tibble(datetime = test_df_dirty$datetime %>%
                             format() %>%
                             as.character(),
                           count = penalized_predictions)
vroom_write(linear_output, "linear_regression.csv", delim = ",")
vroom_write(poisson_output, "poisson_regression.csv", delim = ",")
vroom_write(penalized_output, "penalized_regression.csv", delim = ",")

