############################################################
# Extended Descriptive EDA – Covering All Features (Updated)
# Comprehensive summary statistics, distributions, and relationships
# Adapted for the full cumulative + lagged dataset (278,558 rows)
# Updates: Improved plot 06_lag10_distributions
#   - Included zero values to show full distribution (many events are rare in 10 matches)
#   - Assigned unique colors to each variable for better distinction
#   - Removed overall fill/color; now per variable
#   - Set fixed bandwidth (bw=0.5) for density to reduce waviness from discreteness
#   - Kept histogram with density overlay; scales="free" for varying ranges
#   - Arranged facets in 2 rows for better layout
############################################################

# ----------------------------
# Libraries
# ----------------------------
library(tidyverse)
library(scales)
library(patchwork)
library(corrplot)       # for correlation plot

# ----------------------------
# Paths
# ----------------------------
PLOT_DIR  <- "../Plots"
DATA_PATH <- "../Data_Processed/cumlag_nn_tabular_dataset.csv"

# ----------------------------
# Safety checks
# ----------------------------
dir.create(PLOT_DIR, showWarnings = FALSE, recursive = TRUE)
stopifnot(file.exists(DATA_PATH))

# ----------------------------
# Load & clean data
# ----------------------------
df <- read_csv(DATA_PATH, show_col_types = FALSE) %>%
  filter(
    is.finite(y_log),
    is.finite(age_years),
    is.finite(height_in_cm),
    y_raw > 0
  )

# Add derived columns for plotting
df <- df %>%
  mutate(
    position = case_when(
      pos_ATT ~ "ATT",
      pos_MID ~ "MID",
      pos_DEF ~ "DEF",
      pos_GK ~ "GK",
      TRUE ~ "MISSING"
    ),
    foot = case_when(
      foot_R ~ "Right",
      foot_L ~ "Left",
      foot_B ~ "Both",
      foot_UNK ~ "Unknown",
      TRUE ~ "Unknown"
    ),
    year = year(valuation_date)
  )

# ----------------------------
# Colors
# ----------------------------
COL_BLUE <- "#1f4fd8"
COL_RED <- "#c9332c"
COL_YEL <- "#f2b705"
COL_GRAY <- "grey70"
COL_GREEN <- "#2ca02c"

# ----------------------------
# Global theme
# ----------------------------
theme_set(
  theme_minimal(base_family = "sans", base_size = 12) +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0),
      plot.subtitle = element_text(size = 11, hjust = 0),
      axis.title = element_text(size = 11),
      axis.text = element_text(size = 10),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      legend.position = "bottom",
      legend.title = element_blank(),
      plot.margin = margin(12, 16, 12, 16)
    )
)

# ----------------------------
# PDF writer
# ----------------------------
save_pdf <- function(p, name, w = 12, h = 6) {
  ggsave(
    filename = file.path(PLOT_DIR, paste0(name, ".pdf")),
    plot = p,
    width = w,
    height = h,
    device = "pdf"
  )
}

############################################################
# 1. Overall dataset summary
############################################################
# Numerical summary (view in console)
summary(df)

# Unique players & time span
cat("Unique players:", n_distinct(df$player_id), "\n")
cat("Date range:", min(df$valuation_date), "to", max(df$valuation_date), "\n")

############################################################
# 2. Target: Market value distributions (log & raw)
############################################################
p1a <- df %>%
  filter(between(y_log, quantile(y_log, 0.005), quantile(y_log, 0.995))) %>%
  ggplot(aes(y_log)) +
  geom_histogram(aes(y = after_stat(density)), bins = 140, fill = COL_BLUE, alpha = 0.4) +
  geom_density(color = COL_RED, linewidth = 1.2) +
  labs(title = "Distribution of log(Market value)", x = "log(Market value)", y = "Density")

p1b <- df %>%
  filter(y_raw <= quantile(y_raw, 0.99)) %>%
  ggplot(aes(y_raw / 1e6)) +
  geom_histogram(bins = 100, fill = COL_BLUE, alpha = 0.4) +
  scale_x_log10(labels = label_number(prefix = "€", suffix = "M")) +
  labs(title = "Distribution of Market value (raw, log-x)", x = "Market value (€ millions)", y = "Count")

p1 <- p1a / p1b
save_pdf(p1, "01_market_value_distributions", h = 10)

############################################################
# 3. Age & Height
############################################################
p3a <- df %>%
  ggplot(aes(age_years)) +
  geom_density(fill = COL_BLUE, alpha = 0.6, adjust = 1.5) +
  labs(title = "Age distribution", x = "Age (years)", y = "Density")

p3b <- df %>%
  filter(height_in_cm >= 150) %>%
  ggplot(aes(height_in_cm)) +
  geom_density(fill = COL_RED, alpha = 0.6, adjust = 1.5) +
  labs(title = "Height distribution", x = "Height (cm)", y = "Density")

p3 <- p3a + p3b
save_pdf(p3, "02_age_height_dist")

p4 <- df %>%
  mutate(age_bin = cut(age_years, breaks = seq(15, 45, by = 1))) %>%
  group_by(age_bin) %>%
  summarise(age = mean(age_years, na.rm = TRUE),
            y_mean = mean(y_log, na.rm = TRUE),
            h_mean = mean(height_in_cm, na.rm = TRUE),
            .groups = "drop") %>%
  pivot_longer(cols = c(y_mean, h_mean), names_to = "metric") %>%
  ggplot(aes(age, value)) +
  geom_line(color = COL_BLUE, linewidth = 1.1) +
  facet_wrap(~metric, scales = "free_y", 
             labeller = labeller(metric = c(y_mean = "Mean log(MV)", h_mean = "Mean Height (cm)"))) +
  labs(title = "Age trends: Market value & Height")

save_pdf(p4, "03_age_trends")

############################################################
# 4. Categorical features: Position, Foot, Big-5
############################################################
p5_pos <- ggplot(df, aes(position, y_log, fill = position)) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.1) +
  scale_fill_manual(values = c("ATT" = COL_RED, "MID" = COL_BLUE, "DEF" = COL_GRAY, "GK" = "black", "MISSING" = "grey85")) +
  labs(title = "Market value by Position", x = NULL, y = "log(Market value)")

p5_foot <- ggplot(df, aes(foot, y_log, fill = foot)) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.1) +
  scale_fill_manual(values = c("Right" = COL_BLUE, "Left" = COL_RED, "Both" = COL_GRAY, "Unknown" = "grey85")) +
  labs(title = "Market value by Preferred Foot", x = NULL, y = "log(Market value)")

p5_big5 <- ggplot(df, aes(factor(is_big5_league, labels = c("Non Big-5", "Big-5")), y_log)) +
  geom_boxplot(fill = COL_GREEN, alpha = 0.75) +
  labs(title = "Big-5 League Premium", x = NULL, y = "log(Market value)")

p5 <- (p5_pos | p5_foot) / p5_big5
save_pdf(p5, "04_categorical_mv", h = 10)

############################################################
# 5. Cumulative career stats distributions
############################################################
cum_vars <- c("cumulative_goals", "cumulative_assists", 
              "cumulative_yellow_cards", "cumulative_red_cards",
              "cumulative_sub_in", "cumulative_sub_out")

p_cum <- df %>%
  select(all_of(cum_vars)) %>%
  pivot_longer(everything()) %>%
  filter(value > 0) %>%
  ggplot(aes(log1p(value), fill = name)) +
  geom_density(alpha = 0.4, adjust = 2) +
  facet_wrap(~name, scales = "free_y") +
  labs(title = "Cumulative career stats (log1p scale)", x = "log(1 + count)", y = "Density")

save_pdf(p_cum, "05_cumulative_distributions", w = 14, h = 10)

############################################################
# 6. Lag-10 recent form distributions (Improved)
############################################################
lag_vars <- c("lag_10_goals", "lag_10_assists", 
              "lag_10_yellow_cards", "lag_10_red_cards",
              "lag_10_sub_in", "lag_10_sub_out")

colors <- c(
  "lag_10_goals" = COL_RED,
  "lag_10_assists" = COL_BLUE,
  "lag_10_yellow_cards" = COL_YEL,
  "lag_10_red_cards" = "#8B0000",  # darkred
  "lag_10_sub_in" = COL_GREEN,
  "lag_10_sub_out" = COL_GRAY
)

p_lag <- df %>%
  select(all_of(lag_vars)) %>%
  pivot_longer(everything()) %>%
  # Removed filter(value > 0) to include zeros
  ggplot(aes(value, fill = name, color = name)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 1, alpha = 0.4, position = "identity") +
  geom_density(linewidth = 1.2, bw = 0.5) +  # Fixed bw to reduce waviness
  scale_fill_manual(values = colors) +
  scale_color_manual(values = colors) +
  facet_wrap(~name, scales = "free", nrow = 2) +  # 2 rows for better layout
  labs(title = "Recent form (lag-10 matches, original scale)", x = "Count", y = "Density")

save_pdf(p_lag, "06_lag10_distributions", w = 14, h = 10)

############################################################
# 7. Correlations with log(Market value)
############################################################
cor_df <- df %>%
  select(y_log, age_years, height_in_cm, is_big5_league,
         starts_with("cumulative_"), starts_with("lag_10_")) %>%
  select(where(is.numeric))

cor_mat <- cor(cor_df, use = "pairwise.complete.obs")

pdf(file.path(PLOT_DIR, "07_correlation_matrix.pdf"), width = 14, height = 12)
corrplot(cor_mat, method = "color", type = "upper", 
         tl.cex = 0.8, tl.col = "black",
         order = "hclust", addCoef.col = "black", number.cex = 0.7)
dev.off()

############################################################
# 8. Additional: Market value over time
############################################################
p8 <- df %>%
  group_by(year) %>%
  summarise(mean_y = mean(y_log), .groups = "drop") %>%
  ggplot(aes(year, mean_y)) +
  geom_line(color = COL_BLUE, linewidth = 1.2) +
  geom_point(color = COL_RED) +
  geom_smooth(method = "loess", se = FALSE, color = COL_GRAY, linetype = "dashed") +
  labs(title = "Average log(Market value) over time", x = "Year", y = "Mean log(MV)")

save_pdf(p8, "08_mv_over_time")

############################################################
# END – All major features now covered visually & statistically
############################################################
cat("All plots saved to", PLOT_DIR, "\n")