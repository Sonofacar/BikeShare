# Simple EDA of the Dataset. Outputs a plot
# containing some potentially useful information

library(tidyverse)
library(vroom)
library(patchwork)
library(ggcorrplot)

data <- vroom("train.csv")
data["weather"] <- data$weather |>
  as_factor()
data["season"] <- data$season |>
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

