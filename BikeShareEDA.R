# Simple EDA of the Dataset. Outputs a plot
# containing some potentially useful information

library(tidyverse)
library(vroom)
library(patchwork)
library(ggcorrplot)
library(car)

data <- vroom("train.csv")
data["weather"] <- data$weather |>
  as_factor()
data["season"] <- data$season |>
  as_factor()

# Leverage the datetime data
data["hour"] <- data$datetime %>%
  hour()
data["hour_category"] <- 4
data[(10 >= data$hour) & (data$hour > 5), "hour_category"] <- 1
data[(15 >= data$hour) & (data$hour > 10), "hour_category"] <- 2
data[(19 >= data$hour) & (data$hour > 15), "hour_category"] <- 3
data["hour_category"] <- data$hour_category %>%
  as_factor()
data["weekday"] <- data$datetime %>%
  weekdays() %>%
  as_factor()

# Problem with weather
sink("/dev/null")
data |>
  count(weather)
sink()
# there is only one observation of the '4' category

# weather bar plot
weather_plot <- ggplot(data) +
  geom_bar(aes(weather), fill = "darkgreen") +
  ggtitle("Distribution of Weather Types") +
  xlab("Weather Category") +
  ylab("# of Days") +
  theme_classic()

# correlation plot with weather-related variables
correlations <- data[c("temp", "atemp", "humidity", "windspeed", "count")] |>
  cor() |>
  ggcorrplot(type = "upper", lab = TRUE) +
  ggtitle("Correlations of Numeric Variables")

# taking a look at weather relationships with each season
sink("/dev/null")
data[data$season == 1, c(6, 7, 8, 9)] |>
  summary()
data[data$season == 2, c(6, 7, 8, 9)] |>
  summary()
data[data$season == 3, c(6, 7, 8, 9)] |>
  summary()
data[data$season == 4, c(6, 7, 8, 9)] |>
  summary()
sink()
# there seems to be some strong relationships
# between temperature and season (duh)
# means multicollinearity

# Season-weather boxplots
seasons <- ggplot(data) +
  geom_boxplot(aes(x = season, y = temp, fill = season)) +
  ggtitle("Season-Temperature Boxplots") +
  xlab("Season") +
  ylab("Temperature") +
  theme_classic()

# This could also be shown as a plot with the
# datetime variable
seasonality <- ggplot(data) +
  geom_smooth(aes(x = datetime, y = temp), color = "red", se = FALSE) +
  ggtitle("Date-Temperature Chart") +
  xlab("Date") +
  ylab("Average Temperature") +
  theme_classic()

# Taking a look at categorical variables
categoricals <- data |>
  pivot_longer(c(holiday, workingday),
               names_to = "variable",
               values_to = "value") |>
  mutate(value = ifelse(value == 1, "True", "False")) |>
  group_by(variable) |>
  ggplot() +
  geom_bar(aes(x = variable, fill = value), position = "dodge") +
  ggtitle("Distribution of Holidays and Workdays") +
  xlab("Type of Day") +
  ylab("# of Days") +
  theme_classic()

# Final plot
final_plot <- (weather_plot | correlations) /
  (seasonality | categoricals)

ggsave("EDA_plot.png", final_plot)

# Looking into rush hour
casual_hour_dist <- ggplot(data) +
  geom_point(aes(x = hour, y = casual))
registered_hour_dist <- ggplot(data) +
  geom_point(aes(x = hour, y = registered))
(casual_hour_dist | registered_hour_dist)

# Now by day for casual
casual_hour_dist_sunday <- ggplot(data[data$weekday == "Sunday", ]) +
  geom_point(aes(x = hour, y = casual))
casual_hour_dist_monday <- ggplot(data[data$weekday == "Monday", ]) +
  geom_point(aes(x = hour, y = casual))
casual_hour_dist_tuesday <- ggplot(data[data$weekday == "Tuesday", ]) +
  geom_point(aes(x = hour, y = casual))
casual_hour_dist_wednesday <- ggplot(data[data$weekday == "Sunday", ]) +
  geom_point(aes(x = hour, y = casual))
casual_hour_dist_thursday <- ggplot(data[data$weekday == "Thursday", ]) +
  geom_point(aes(x = hour, y = casual))
casual_hour_dist_friday <- ggplot(data[data$weekday == "Friday", ]) +
  geom_point(aes(x = hour, y = casual))
casual_hour_dist_saturday <- ggplot(data[data$weekday == "Saturday", ]) +
  geom_point(aes(x = hour, y = casual))
(casual_hour_dist_sunday | casual_hour_dist_monday | casual_hour_dist_tuesday) /
  (casual_hour_dist_wednesday |
     casual_hour_dist_thursday |
     casual_hour_dist_friday |
     casual_hour_dist_saturday)

# Now by day for registered
registered_hour_dist_sunday <- ggplot(data[data$weekday == "Sunday", ]) +
  geom_point(aes(x = hour, y = registered))
registered_hour_dist_monday <- ggplot(data[data$weekday == "Monday", ]) +
  geom_point(aes(x = hour, y = registered))
registered_hour_dist_tuesday <- ggplot(data[data$weekday == "Tuesday", ]) +
  geom_point(aes(x = hour, y = registered))
registered_hour_dist_wednesday <- ggplot(data[data$weekday == "Sunday", ]) +
  geom_point(aes(x = hour, y = registered))
registered_hour_dist_thursday <- ggplot(data[data$weekday == "Thursday", ]) +
  geom_point(aes(x = hour, y = registered))
registered_hour_dist_friday <- ggplot(data[data$weekday == "Friday", ]) +
  geom_point(aes(x = hour, y = registered))
registered_hour_dist_saturday <- ggplot(data[data$weekday == "Saturday", ]) +
  geom_point(aes(x = hour, y = registered))
(registered_hour_dist_sunday |
   registered_hour_dist_monday |
   registered_hour_dist_tuesday) /
  (registered_hour_dist_wednesday |
     registered_hour_dist_thursday |
     registered_hour_dist_friday |
     registered_hour_dist_saturday)

# this makes it seem like there are some interactions here:
# first is between registered and casual users. Casual users
# experience no rush hour while registered users do. In
# addition, registered users only see that on certain days.

# Looking at the linear relationship
formula <- count ~ hour_category +
  atemp +
  season +
  humidity +
  weather +
  weekday +
  temp +
  holiday +
  windspeed
model <- glm(formula, data = data, family = poisson())
data["residuals"] <- model$residuals

resid_vs_hour_category <- ggplot(data) +
  geom_point(aes(x = hour_category, y = residuals)) +
  theme(aspect.ratio = 1)
resid_vs_atemp <- ggplot(data) +
  geom_point(aes(x = atemp, y = residuals)) +
  theme(aspect.ratio = 1)
resid_vs_season <- ggplot(data) +
  geom_point(aes(x = season, y = residuals)) +
  theme(aspect.ratio = 1)
resid_vs_humidity <- ggplot(data) +
  geom_point(aes(x = humidity, y = residuals)) +
  theme(aspect.ratio = 1)
resid_vs_weather <- ggplot(data) +
  geom_point(aes(x = weather, y = residuals)) +
  theme(aspect.ratio = 1)
resid_vs_weekday <- ggplot(data) +
  geom_point(aes(x = weekday, y = residuals)) +
  theme(aspect.ratio = 1)
resid_vs_temp <- ggplot(data) +
  geom_point(aes(x = temp, y = residuals)) +
  theme(aspect.ratio = 1)
resid_vs_holiday <- ggplot(data) +
  geom_point(aes(x = holiday, y = residuals)) +
  theme(aspect.ratio = 1)
resid_vs_windspeed <- ggplot(data) +
  geom_point(aes(x = windspeed, y = residuals)) +
  theme(aspect.ratio = 1)

(resid_vs_hour_category | resid_vs_atemp | resid_vs_season) /
  (resid_vs_humidity | resid_vs_weather | resid_vs_weekday) /
  (resid_vs_temp | resid_vs_holiday | resid_vs_windspeed)

