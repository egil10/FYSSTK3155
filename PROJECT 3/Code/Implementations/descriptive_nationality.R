############################################################
# Transfermarkt – Nationality Signal (Refined EDA)
#
# Outputs:
#   1) Boxplot: Top 10 vs Bottom 5 countries by median MV
#   2) Facet grid: Median MV lifecycle by nationality
#   3) Facet grid: Max MV lifecycle by nationality
#
# Output size: 12 x 6
############################################################

# ----------------------------
# Libraries
# ----------------------------
library(tidyverse)
library(lubridate)
library(scales)

# ----------------------------
# Paths
# ----------------------------
DATA_DIR <- "../Data"
PLOT_DIR <- "../Plots"

# PLOT_DIR  <- "../Plots"
# DATA_PATH <- "../Data_Processed/cumlag_nn_tabular_dataset.csv"

# ----------------------------
# Load data
# ----------------------------
players <- read_csv(file.path(DATA_DIR, "players.csv"), show_col_types = FALSE)
values  <- read_csv(file.path(DATA_DIR, "player_valuations.csv"), show_col_types = FALSE)

# ----------------------------
# Merge & prepare data
# ----------------------------
df <- values %>%
  inner_join(players, by = "player_id", suffix = c("_val", "_pl")) %>%
  filter(!is.na(position)) %>%
  mutate(
    log_mv = log10(market_value_in_eur_val + 1),
    age = as.numeric(date - as.Date(date_of_birth)) / 365.25
  ) %>%
  filter(
    is.finite(log_mv),
    age >= 16,
    age <= 42
  )

############################################################
# Plot 01: Top 10 vs Bottom 5 countries (Boxplot)
############################################################
country_rank <- df %>%
  group_by(country_of_citizenship) %>%
  summarise(
    median_mv = median(log_mv, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  filter(n >= 300) %>%     # stability
  arrange(desc(median_mv))

top10 <- country_rank %>% slice_head(n = 10)
bot5  <- country_rank %>% slice_tail(n = 5)

countries_01 <- c(
  top10$country_of_citizenship,
  bot5$country_of_citizenship
)

df_01 <- df %>%
  filter(country_of_citizenship %in% countries_01) %>%
  mutate(country = fct_reorder(country_of_citizenship, log_mv, median))

############################################################
# Countries for lifecycle plots (World Cup + peers)
############################################################
countries_grid <- c(
  "Brazil", "Germany", "Italy", "Argentina",
  "France", "Uruguay", "England", "Spain",
  "Netherlands", "Belgium", "Portugal",
  "Croatia", "Denmark", "Turkey", "Ukraine"
)

df_grid <- df %>%
  filter(country_of_citizenship %in% countries_grid) %>%
  mutate(country = fct_reorder(country_of_citizenship, log_mv, median))

# ----------------------------
# Global theme
# ----------------------------
theme_set(
  theme_minimal(base_family = "sans", base_size = 12) +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0),
      plot.subtitle = element_text(size = 11, hjust = 0),
      axis.title = element_text(size = 11),
      axis.text = element_text(size = 9),
      strip.text = element_text(size = 9, face = "bold"),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      legend.position = "none",
      plot.margin = margin(12, 16, 12, 16)
    )
)

############################################################
# Plot 01 – Boxplot
############################################################
p1 <- ggplot(df_01, aes(country, log_mv, fill = country)) +
  geom_boxplot(outlier.alpha = 0.1, alpha = 0.75) +
  coord_flip() +
  labs(
    title = "Market Value by Country of Citizenship",
    subtitle = "Top 10 vs Bottom 5 countries by median market value",
    x = NULL,
    y = "log10(Market value in EUR + 1)"
  )

ggsave(
  file.path(PLOT_DIR, "01_country_market_value_boxplot.pdf"),
  p1, width = 12, height = 6, device = "pdf"
)

############################################################
# Plot 02 – Median lifecycle
############################################################
p2_data <- df_grid %>%
  mutate(age_int = round(age)) %>%
  group_by(country, age_int) %>%
  summarise(
    value = median(log_mv),
    n_bin = n(),
    .groups = "drop"
  ) %>%
  filter(n_bin >= 25)

p2 <- ggplot(p2_data, aes(age_int, value)) +
  geom_line(linewidth = 0.9) +
  geom_smooth(method = "loess", se = FALSE, span = 0.45,
              linetype = "dashed", linewidth = 0.7) +
  facet_wrap(~country, ncol = 5, scales = "free_y") +
  labs(
    title = "Age–Market Value Lifecycle by Nationality",
    subtitle = "Median log(MV) by age (dashed = loess)",
    x = "Age",
    y = "Median log10(Market value + 1)"
  )

ggsave(
  file.path(PLOT_DIR, "02_country_age_median_grid.pdf"),
  p2, width = 12, height = 6, device = "pdf"
)

############################################################
# Plot 03 – Max lifecycle (Upside envelope)
############################################################
p3_data <- df_grid %>%
  mutate(age_int = round(age)) %>%
  group_by(country, age_int) %>%
  summarise(
    value = max(log_mv),
    n_bin = n(),
    .groups = "drop"
  ) %>%
  filter(n_bin >= 10)   # lower threshold OK for max

p3 <- ggplot(p3_data, aes(age_int, value)) +
  geom_line(linewidth = 0.9, color = "firebrick") +
  geom_smooth(method = "loess", se = FALSE, span = 0.5,
              linetype = "dashed", linewidth = 0.7, color = "grey40") +
  facet_wrap(~country, ncol = 5, scales = "free_y") +
  labs(
    title = "Age–Market Value Upside by Nationality",
    subtitle = "Maximum log(MV) by age (superstar envelope)",
    x = "Age",
    y = "Max log10(Market value + 1)"
  )

ggsave(
  file.path(PLOT_DIR, "03_country_age_max_grid.pdf"),
  p3, width = 12, height = 6, device = "pdf"
)

############################################################
# END
############################################################
cat("Saved 3 nationality-focused plots to:", PLOT_DIR, "\n")
