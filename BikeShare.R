# Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)
library(stacks)

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
  step_mutate(season = as_factor(season)) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.5)
log_recipe <- recipe(count ~ ., log_train_df_dirty) %>%
  step_mutate(weather = weather %>%
                replace(weather == 4, 3) %>%
                as_factor()) %>%
  step_time(datetime, features = c("hour")) %>%
  step_rename(hour = datetime_hour) %>%
  step_mutate(rush_hour = as.integer(hour %in% c(7, 8, 17, 18))) %>%
  step_mutate(season = as_factor(season)) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.5)
linear_recipe <- recipe %>%
  step_poly(hour, degree = 10)
prepped_recipe <- prep(recipe)
clean_data <- bake(prepped_recipe, new_data = train_df_dirty)

# Set up folds
folds <- vfold_cv(log_train_df_dirty, v = 10, repeats = 1)
poisson_folds <- vfold_cv(train_df_dirty, v = 10, repeats = 1)

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
  add_recipe(linear_recipe)

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
  add_recipe(linear_recipe)

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

# Model-specific recipe
penalized_recipe <- linear_recipe %>%
  step_normalize(all_numeric_predictors())

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
penalized_cv <- penalized_workflow %>%
  tune_grid(resamples = folds,
            grid = param_grid,
            metrics = metric_set(rmse),
            control = control_stack_grid())
best_tune <- penalized_cv %>%
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

####################
# Regression Trees #
####################

# Create the model
tree_model <- decision_tree(cost_complexity = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")

# Create the workflow
tree_workflow <- workflow() %>%
  add_model(tree_model) %>%
  add_recipe(recipe)

# Cross validate
param_grid <- grid_regular(cost_complexity(),
                           levels = 30)
tree_cv <- tree_workflow %>%
  tune_grid(resamples = folds,
            grid = param_grid,
            metrics = metric_set(rmse),
            control = control_stack_grid())
best_tune <- tree_cv %>%
  select_best(metric = "rmse")

# Fit model and make predicitons
tree_fit <- tree_workflow %>%
  finalize_workflow(best_tune) %>%
  fit(data = log_train_df_dirty)
tree_predictions <- predict(tree_fit, new_data = test_df_dirty)$.pred %>%
  exp()

# Write output
tree_output <- tibble(datetime = test_df_dirty$datetime %>%
                        format() %>%
                        as.character(),
                      count = tree_predictions)
vroom_write(tree_output, "tree_regression.csv", delim = ",")

##################
# Random Forrest #
##################

# Create the model
forest_model <- rand_forest(mtry = tune(),
                            min_n = tune(),
                            trees = 200) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# Create the workflow
forest_workflow <- workflow() %>%
  add_model(forest_model) %>%
  add_recipe(recipe)

# Cross validate
param_grid <- grid_regular(mtry(range = c(1, 11)),
                           min_n(),
                           levels = 10)
forest_cv <- forest_workflow %>%
  tune_grid(resamples = folds,
            grid = param_grid,
            metrics = metric_set(rmse),
            control = control_stack_grid())
best_tune <- forest_cv %>%
  select_best(metric = "rmse")

# Fit model and make predicitons
forest_fit <- forest_workflow %>%
  finalize_workflow(best_tune) %>%
  fit(data = log_train_df_dirty)
forest_predictions <- predict(forest_fit, new_data = test_df_dirty)$.pred %>%
  exp()

# Write output
forest_output <- tibble(datetime = test_df_dirty$datetime %>%
                          format() %>%
                          as.character(),
                        count = forest_predictions)
vroom_write(forest_output, "forest_regression.csv", delim = ",")

#################
# Stacked model #
#################

# Fit linear and poisson models as resampling objects
linear_resample_fit <- fit_resamples(linear_workflow,
                                     resamples = folds,
                                     metrics = metric_set(rmse),
                                     control = control_stack_resamples())
poisson_resample_fit <- fit_resamples(poisson_workflow,
                                      resamples = poisson_folds,
                                      metrics = metric_set(rmse),
                                      control = control_stack_resamples())

# Specify the stack
stack <- stacks() %>%
  #add_candidates(linear_resample_fit) %>%
  #add_candidates(poisson_resample_fit) %>%
  add_candidates(penalized_cv) %>%
  add_candidates(tree_cv) %>%
  add_candidates(forest_cv)

# Create the stack model
stack_model <- stack %>%
  blend_predictions() %>%
  fit_members()

# Make predicitons
stack_predictions <- predict(stack_model, new_data = test_df_dirty)$.pred

# Write output
stack_output <- tibble(datetime = test_df_dirty$datetime %>%
                         format() %>%
                         as.character(),
                       count = forest_predictions)
vroom_write(stack_output, "stacked_models.csv", delim = ",")

#################################
# Support Vector Machines (RBF) #
#################################

# Create the model
svm_model <- svm_rbf(cost = tune()) %>%
  set_engine("kernlab") %>%
  set_mode("regression")

# Create the workflow
svm_workflow <- workflow() %>%
  add_model(svm_model) %>%
  add_recipe(log_recipe)

# Cross validate
param_grid <- grid_regular(cost(), levels = 10)
svm_cv <- svm_workflow %>%
  tune_grid(resamples = folds,
            grid = param_grid,
            metrics = metric_set(rmse),
            control = control_stack_grid())
best_tune <- svm_cv %>%
  select_best(metric = "rmse")

# Fit model and make predictions
svm_fit <- svm_workflow %>%
  finalize_workflow(best_tune) %>%
  fit(data = log_train_df_dirty)
svm_predictions <- predict(svm_fit, new_data = test_df_dirty)$.pred %>%
  exp()

# Write output
svm_output <- tibble(datetime = test_df_dirty$datetime %>%
                       format() %>%
                       as.character(),
                     count = svm_predictions)
vroom_write(svm_output, "svm_rbf.csv", delim = ",")

########
# Bart #
########

# Create the model
bart_model <- parsnip::bart(trees = 500) %>%
  set_engine("dbarts") %>%
  set_mode("regression")

# Create the workflow
bart_workflow <- workflow() %>%
  add_model(bart_model) %>%
  add_recipe(log_recipe)

# Fit model and make predictions
bart_fit <- bart_workflow %>%
  fit(data = log_train_df_dirty)
bart_predictions <- predict(bart_fit, new_data = test_df_dirty)$.pred %>%
  exp()

# Write output
bart_output <- tibble(datetime = test_df_dirty$datetime %>%
                       format() %>%
                       as.character(),
                     count = bart_predictions)
vroom_write(bart_output, "bart_regression.csv", delim = ",")

#######
# GAM #
#######

# Create the model
# Create the workflow
# Cross validate
# Fit model and make predictions
# Write output

########
# MARS #
########

# Create the model
# Create the workflow
# Cross validate
# Fit model and make predictions
# Write output

#######
# MLP #
#######

# Create the model
# Create the workflow
# Cross validate
# Fit model and make predictions
# Write output

#################
# Boosted Trees #
#################

# Create the model
# Create the workflow
# Cross validate
# Fit model and make predictions
# Write output

######################
# K-nearest neighbor #
######################

# Create the model
# Create the workflow
# Cross validate
# Fit model and make predictions
# Write output

