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

# Find the seasonality of hours

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

#####################
# Linear regression #
#####################

# Create the model
linear_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

# Create the workflow
linear_workflow <- workflow() %>%
  add_model(linear_model) %>%
  add_recipe(recipe)

# Fit model and make predictions
linear_fit <- fit(linear_workflow, data = log_train_df_dirty)
linear_predictions <- predict(linear_fit, new_data = test_df_dirty)$.pred %>%
  exp()

# Write output
linear_output <- tibble(datetime = test_df_dirty$datetime %>%
                          format() %>%
                          as.character(),
                        count = linear_predictions)
vroom_write(linear_output, "linear_regression.csv", delim = ",")

######################
# Poisson regression #
######################

# Create the model
poisson_model <- poisson_reg() %>%
  set_engine("glm") %>%
  set_mode("regression")

# Create the workflow
poisson_workflow <- workflow() %>%
  add_model(poisson_model) %>%
  add_recipe(recipe)

# Fit model and make predictions
poisson_fit <- fit(poisson_workflow, data = train_df_dirty)
poisson_predictions <- predict(poisson_fit, new_data = test_df_dirty)$.pred

# Write output
poisson_output <- tibble(datetime = test_df_dirty$datetime %>%
                           format() %>%
                           as.character(),
                         count = poisson_predictions)
vroom_write(poisson_output, "poisson_regression.csv", delim = ",")

########################
# Penalized regression #
########################

# Create the model
penalized_model <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

# Create the workflow
penalized_workflow <- workflow() %>%
  add_model(penalized_model) %>%
  add_recipe(penalized_recipe)

# Cross validate
param_grid <- grid_regular(penalty(),
                           mixture(),
                           levels = 10)
folds <- vfold_cv(log_train_df_dirty, v = 10, repeats = 1)
cv_results <- penalized_workflow %>%
  tune_grid(resamples = folds,
            grid = param_grid,
            metrics = metric_set(rmse))
best_tune <- cv_results %>%
  select_best(metric = "rmse")

# Fit model and make predictions
penalized_fit <- penalized_workflow %>%
  finalize_workflow(best_tune) %>%
  fit(data = log_train_df_dirty)
penalized_predictions <- predict(penalized_fit,
                                 new_data = test_df_dirty)$.pred %>%
  exp()

# Write output
penalized_output <- tibble(datetime = test_df_dirty$datetime %>%
                             format() %>%
                             as.character(),
                           count = penalized_predictions)
vroom_write(penalized_output, "penalized_regression.csv", delim = ",")

