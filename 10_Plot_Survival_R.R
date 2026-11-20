library(survival)
library(survminer)
library(readr)
library(ggplot2)

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("--file=", args, value = TRUE)
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(sub("--file=", "", file_arg))) else getwd()
data_dir <- normalizePath(file.path(script_dir, "..", "data"), mustWork = FALSE)
fig_dir <- normalizePath(file.path(script_dir, "..", "figures"), mustWork = FALSE)
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

plot_km <- function(data_file, out_pdf, title_text, xlab_text, ylab_text) {
  df <- read_csv(file.path(data_dir, data_file), show_col_types = FALSE)
  df$Group <- factor(df$Group, levels = c("High", "Low"))
  fit <- survfit(Surv(Time, Event) ~ Group, data = df)

  splots <- ggsurvplot(
    fit, data = df, size = 1.4,
    palette = c("#E63946", "#1D3557"),
    pval = TRUE, pval.coord = c(4, 0.12),
    risk.table = TRUE, risk.table.col = "strata", risk.table.height = 0.27,
    legend.title = "Lineage Signature",
    legend.labs = c("High (>= Median)", "Low (< Median)"),
    xlab = xlab_text, ylab = ylab_text,
    title = title_text,
    ggtheme = theme_classic(base_family = "sans", base_size = 12) +
      theme(plot.title = element_text(size = 12.5, face = "bold", hjust = 0.5),
            plot.margin = margin(10, 20, 10, 15))
  )
  splots$table <- splots$table + labs(y = NULL) + theme(plot.margin = margin(0, 20, 10, 15))

  cairo_pdf(file.path(fig_dir, out_pdf), width = 8.2, height = 6.2)
  print(splots)
  dev.off()
}

plot_km("Figure_7B_Survival_Data.csv", "Figure_7B_TCGA_OS.pdf",
        "Overall Survival in Resected EGFR-Mutant LUAD (TCGA)",
        "Overall Survival (Months)", "Overall Survival Probability")

plot_km("Figure_9_Survival_Data.csv", "Figure_9_TCGA_PFS.pdf",
        "Progression-Free Survival in Resected EGFR-Mutant LUAD (TCGA)",
        "Progression-Free Survival (Months)", "Progression-Free Probability")

print("--- SCRIPT 10 COMPLETE ---")
