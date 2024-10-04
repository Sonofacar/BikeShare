# Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)
library(stacks)

# Read in the data
offset <- 0.5
raw_train <- vroom("train.csv")
train_df_dirty <- raw_train %>%
  select(-casual, -registered)
log_train_df_dirty <- train_df_dirty %>%
  mutate(count = log(count))
test_df_dirty <- vroom("test.csv")
casual_df_dirty <- raw_train %>%
  select(-count, -registered)
registered_df_dirty <- raw_train %>%
  select(-count, -casual)
log_casual_df_dirty <- casual_df_dirty %>%
  mutate(casual = log(casual + offset))
log_registered_df_dirty <- registered_df_dirty %>%
  mutate(registered = log(registered + offset))

# Data cleaning recipe
# nolint start
cleaner <- function(r) {
  r %>%
    step_rm(temp) %>%
    step_normalize(atemp, humidity, windspeed) %>%
    step_mutate(weather = weather %>%
                  replace(weather == 4, 3) %>%
                  as_factor()) %>%
    step_time(datetime, features = c("hour")) %>%
    step_date(datetime, features = c("dow", "month", "year")) %>%
    step_rename(hour = datetime_hour,
                dow = datetime_dow,
                month = datetime_month,
                year = datetime_year) %>%
    step_mutate(season = as_factor(season)) %>%
    step_mutate(holiday = as_factor(holiday)) %>%
    step_mutate(workingday = as_factor(workingday)) %>%
    step_mutate(hour = as_factor(hour)) %>% # this is *very* important
    step_mutate(dow = as_factor(dow)) %>%
    step_mutate(month = as_factor(month)) %>%
    step_mutate(year = as_factor(year)) %>%
    step_mutate(rush_hour = as_factor(hour %in% c(7, 8, 17, 18))) %>%
    step_mutate(night = as_factor(hour %in% c(0, 1, 2, 3, 4, 22, 23))) %>%
    step_interact(terms = ~ hour:workingday) %>%
    step_interact(terms = ~ hour:weather) %>%
    step_interact(terms = ~ rush_hour:workingday) %>%
    step_interact(terms = ~ atemp:hour) %>%
    step_interact(terms = ~ weather:windspeed) %>%
    step_interact(terms = ~ weather:atemp) %>%
    step_interact(terms = ~ night:dow) %>%
    step_spline_natural(atemp, deg_free = 4) %>%
    step_dummy(all_factor_predictors()) %>%
    step_rm(datetime) %>%
    return()
}
# nolint end
recipe <- recipe(count ~ ., train_df_dirty) %>%
  cleaner()
casual_recipe <- recipe(casual ~ ., casual_df_dirty) %>%
  cleaner()
registered_recipe <- recipe(registered ~ ., registered_df_dirty) %>%
  cleaner()
log_recipe <- recipe(count ~ ., log_train_df_dirty) %>%
  cleaner()
log_casual_recipe <- recipe(casual ~ ., log_casual_df_dirty) %>%
  cleaner()
log_registered_recipe <- recipe(registered ~ ., log_registered_df_dirty) %>%
  cleaner()
prepped_recipe <- prep(recipe)
clean_data <- bake(prepped_recipe, new_data = train_df_dirty)

# Set up folds
folds <- vfold_cv(log_train_df_dirty, v = 10, repeats = 1)
poisson_folds <- vfold_cv(train_df_dirty, v = 10, repeats = 1)
casual_folds <- vfold_cv(log_casual_df_dirty, v = 10, repeats = 1)
registered_folds <- vfold_cv(log_registered_df_dirty, v = 10, repeats = 1)

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
  add_recipe(log_recipe)

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

# Now as two separate models
# Create the workflows
casual_linear_workflow <- workflow() %>%
  add_model(linear_model) %>%
  add_recipe(log_casual_recipe)
registered_linear_workflow <- workflow() %>%
  add_model(linear_model) %>%
  add_recipe(log_registered_recipe)

# Fit models and make predictions
casual_linear_fit <- fit(casual_linear_workflow, data = log_casual_df_dirty)
casual_linear_predictions <- predict(casual_linear_fit,
                                     new_data = test_df_dirty)$.pred %>%
  exp() %>%
  `-`(offset)
registered_linear_fit <- fit(registered_linear_workflow,
                             data = log_registered_df_dirty)
registered_linear_predictions <- predict(registered_linear_fit,
                                         new_data = test_df_dirty)$.pred %>%
  exp() %>%
  `-`(offset)

# Write output
split_linear_output <- tibble(datetime = test_df_dirty$datetime %>%
                                format() %>%
                                as.character(),
                              count = casual_linear_predictions +
                                registered_linear_predictions)
split_linear_output[split_linear_output$count < 0, "count"] <- 0
vroom_write(split_linear_output, "split_linear_regression.csv", delim = ",")

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

# Now do the same thing for casual and registered users
# Zero inflate the data
zinf_casual_df_dirty <- casual_df_dirty %>%
  mutate(casual = casual + 1)
zinf_registered_df_dirty <- registered_df_dirty %>%
  mutate(registered = registered + 1)

# Create the workflow
casual_poisson_workflow <- workflow() %>%
  add_model(poisson_model) %>%
  add_recipe(casual_recipe)
registered_poisson_workflow <- workflow() %>%
  add_model(poisson_model) %>%
  add_recipe(registered_recipe)

# Fit model and make predictions
casual_poisson_fit <- fit(casual_poisson_workflow, data = zinf_casual_df_dirty)
casual_poisson_predictions <- predict(casual_poisson_fit,
                                      new_data = test_df_dirty)$.pred %>%
  `-`(1)
registered_poisson_fit <- fit(registered_poisson_workflow,
                              data = zinf_registered_df_dirty)
registered_poisson_predictions <- predict(registered_poisson_fit,
                                          new_data = test_df_dirty)$.pred %>%
  `-`(1)

# Write output
split_linear_output <- tibble(datetime = test_df_dirty$datetime %>%
                                format() %>%
                                as.character(),
                              count = casual_linear_predictions +
                                registered_linear_predictions)
split_linear_output[split_linear_output$count < 0, "count"] <- 0
vroom_write(split_linear_output, "split_linear_regression.csv", delim = ",")

########################
# Penalized regression #
########################

# Model-specific recipe
penalized_recipe <- log_recipe %>%
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
                           levels = 20)
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

# Now do the same thing but for casual and registered users
# Model-specific recipes
casual_penalized_recipe <- log_casual_recipe %>%
  step_normalize(all_numeric_predictors())
registered_penalized_recipe <- log_registered_recipe %>%
  step_normalize(all_numeric_predictors())

# Create the workflows
casual_penalized_workflow <- workflow() %>%
  add_model(penalized_model) %>%
  add_recipe(casual_penalized_recipe)
registered_penalized_workflow <- workflow() %>%
  add_model(penalized_model) %>%
  add_recipe(registered_penalized_recipe)

# Cross validate
param_grid <- grid_regular(penalty(),
                           mixture(),
                           levels = 20)
casual_penalized_cv <- casual_penalized_workflow %>%
  tune_grid(resamples = casual_folds,
            grid = param_grid,
            metrics = metric_set(rmse),
            control = control_stack_grid())
casual_best_tune <- casual_penalized_cv %>%
  select_best(metric = "rmse")
registered_penalized_cv <- registered_penalized_workflow %>%
  tune_grid(resamples = registered_folds,
            grid = param_grid,
            metrics = metric_set(rmse),
            control = control_stack_grid())
registered_best_tune <- registered_penalized_cv %>%
  select_best(metric = "rmse")

# Fit models and make predictions
casual_penalized_fit <- casual_penalized_workflow %>%
  finalize_workflow(casual_best_tune) %>%
  fit(data = log_casual_df_dirty)
casual_penalized_predictions <- predict(casual_penalized_fit,
                                        new_data = test_df_dirty)$.pred %>%
  exp() %>%
  `-`(offset)
registered_penalized_fit <- registered_penalized_workflow %>%
  finalize_workflow(registered_best_tune) %>%
  fit(data = log_registered_df_dirty)
registered_penalized_predictions <- predict(registered_penalized_fit,
                                            new_data = test_df_dirty)$.pred %>%
  exp() %>%
  `-`(offset)

# Write output
split_penalized_output <- tibble(datetime = test_df_dirty$datetime %>%
                                   format() %>%
                                   as.character(),
                                 count = casual_penalized_predictions +
                                   registered_penalized_predictions)
split_penalized_output[split_penalized_output$count < 0, "count"] <- 0
vroom_write(split_penalized_output, "split_penalized_regression.csv", delim = ",")

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
  add_recipe(log_recipe)

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
                            trees = 250) %>%
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

# Now the same thing but for casual and registered users
# Create the workflow
casual_bart_workflow <- workflow() %>%
  add_model(bart_model) %>%
  add_recipe(log_casual_recipe)
registered_bart_workflow <- workflow() %>%
  add_model(bart_model) %>%
  add_recipe(log_registered_recipe)

# Fit model and make predictions
casual_bart_fit <- casual_bart_workflow %>%
  fit(data = log_casual_df_dirty)
casual_bart_predictions <- predict(casual_bart_fit, new_data = test_df_dirty)$.pred %>%
  exp()
registered_bart_fit <- registered_bart_workflow %>%
  fit(data = log_registered_df_dirty)
registered_bart_predictions <- predict(registered_bart_fit, new_data = test_df_dirty)$.pred %>%
  exp()

# Write output
split_bart_output <- tibble(datetime = test_df_dirty$datetime %>%
				   format() %>%
				   as.character(),
				 count = casual_bart_predictions +
				   registered_bart_predictions)
split_bart_output[split_bart_output$count < 0, "count"] <- 0
vroom_write(split_bart_output, "split_bart_regression.csv", delim = ",")

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
mlp_model <- mlp(hidden_units = tune(),
                 epochs = tune()) %>%
  set_engine("nnet") %>%
  set_mode("regression")

# Create the workflow
mlp_workflow <- workflow() %>%
  add_model(mlp_model) %>%
  add_recipe(log_recipe)

# Cross validate
param_grid <- grid_regular(hidden_units(),
                           epochs(),
                           levels = 10)
mlp_cv <- mlp_workflow %>%
  tune_grid(resamples = folds,
            grid = param_grid,
            metrics = metric_set(rmse),
            control = control_stack_grid())
best_tune <- mlp_cv %>%
  select_best(metric = "rmse")

# Fit model and make predictions
mlp_fit <- mlp_workflow %>%
  finalize_workflow(best_tune) %>%
  fit(data = log_train_df_dirty)
mlp_predictions <- predict(mlp_fit, new_data = test_df_dirty)$.pred %>%
  exp()

# Write output
mlp_output <- tibble(datetime = test_df_dirty$datetime %>%
                       format() %>%
                       as.character(),
                     count = mlp_predictions)
vroom_write(mlp_output, "mlp_model.csv", delim = ",")

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
bart_resample_fit <- fit_resamples(bart_workflow,
                                   resamples = folds,
                                   metrics = metric_set(rmse),
                                   control = control_stack_resamples())

# Specify the stack
stack <- stacks() %>%
  #add_candidates(linear_resample_fit) %>%
  #add_candidates(poisson_resample_fit) %>%
  add_candidates(penalized_cv) %>%
  add_candidates(tree_cv) %>%
  add_candidates(svm_cv) %>%
  add_candidates(mlp_cv) %>%
  add_candidates(bart_resample_fit) %>%
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

