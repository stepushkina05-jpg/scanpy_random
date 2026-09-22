

library(readr)
library(ggplot2)
library(dplyr)
library(pheatmap)



# --------------------------------------------------
# 1. LOAD MASTER RESULTS
# --------------------------------------------------

res <- read_csv("scanpy_master_runs.csv")
res <- res %>%
  filter(n_iter %in% c(2, 3, 4, 5))

with(res, table(n_iter, n_oversamples))
with(res, table(n_iter, dataset))

res$n_oversamples <- as.factor(res$n_oversamples)

dim(res)

# remove extreme runtime outliers
res <- res %>% filter(runtime_s < 700)

dim(res)


# --------------------------------------------------
# 2. RUNTIME
# --------------------------------------------------

ggplot(
  res,
  aes(
    n_iter,
    runtime_s,
    colour = n_oversamples
  )
) +
  geom_point(position = position_dodge(width = 0.5))


# --------------------------------------------------
# 3. APPROXIMATION QUALITY
# --------------------------------------------------

ggplot(
  res,
  aes(
    n_iter,
    approximation_quality,
    colour = n_oversamples
  )
) +
  geom_jitter(position = position_dodge(width = 0.5))


# --------------------------------------------------
# 4. MEAN PC CORRELATION
# --------------------------------------------------

ggplot(
  res,
  aes(
    n_iter,
    mean_abs_pc_corr,
    colour = n_oversamples
  )
) +
  geom_jitter(position = position_dodge(width = 0.5))


# --------------------------------------------------
# 5. APPROXIMATION QUALITY VS PC RECOVERY
# --------------------------------------------------

res$n_iter <- as.factor(res$n_iter)

ggplot(
  res,
  aes(
    approximation_quality,
    mean_abs_pc_corr,
    colour = n_iter
  )
) +
  geom_point(size = 2) +
  facet_grid(dataset ~ n_oversamples)


# --------------------------------------------------
# 6. LOAD INDIVIDUAL PC CORRELATIONS
# --------------------------------------------------

pcs_all <- read_csv("scanpy_pc_correlations.csv") %>%
  as.data.frame()


# --------------------------------------------------
# 7. SORT RUNS EXPLICITLY
# dataset -> n_iter -> n_oversamples -> seed
# --------------------------------------------------

pcs_all <- pcs_all %>%
  arrange(
    dataset,
    n_iter,
    n_oversamples,
    seed
  )


# --------------------------------------------------
# 8. SPLIT METADATA AND CORRELATIONS
# --------------------------------------------------

cor_cols <- grep("^PC", colnames(pcs_all))

meta <- pcs_all[, -cor_cols]
pcs <- pcs_all[, cor_cols]


# give heatmap rows unique IDs
rownames(meta) <- rownames(pcs) <- paste0(
  "scenario",
  seq_len(nrow(pcs))
)


# --------------------------------------------------
# 9. SORTED PC CORRELATION HEATMAP
# --------------------------------------------------

pheatmap(
  pcs,
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  show_rownames = FALSE,
  annotation_row = meta %>% select(dataset, n_iter, n_oversamples),
  filename = "sorted_pc_correlations_heatmap.png",
  width = 10,
  height = 12
)