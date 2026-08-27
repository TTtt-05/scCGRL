# Exact audited multi-method R runner, patched through 2026-08-21.
args <- commandArgs(trailingOnly = TRUE)
if (exists("fixed_method", inherits = FALSE)) {
  args <- c(fixed_method, args)
}
if (length(args) != 7L) {
  stop("Usage: runner.R METHOD INPUT_DIR OUTPUT_CSV LINEAGE_CSV SEED ROOT_CELL_ID R_LIBRARY")
}
method <- args[[1L]]
input_dir <- normalizePath(args[[2L]], winslash = "/", mustWork = TRUE)
output_csv <- args[[3L]]
lineage_csv <- args[[4L]]
seed <- as.integer(args[[5L]])
root_cell_id <- args[[6L]]
r_library <- normalizePath(args[[7L]], winslash = "/", mustWork = TRUE)
.libPaths(c(r_library, .libPaths()))
set.seed(seed)
suppressPackageStartupMessages(library(Matrix))

# Monocle 2.40.0 still calls dplyr's pre-1.0 underscore API from its
# dispersion-estimation path. Keep that historical API isolated from every
# other benchmark method; no trajectory parameter is changed.
if (method == "Monocle2") {
  legacy_dplyr_library <- Sys.getenv("SCCGRL_MONOCLE2_DPLYR_LIBRARY")
  if (!dir.exists(legacy_dplyr_library)) stop("Monocle2 legacy dplyr library is missing")
  .libPaths(c(legacy_dplyr_library, r_library, .libPaths()))
}

# Monocle3 is installed in a dedicated R 4.6 library, while its official
# dependencies are supplied by the established benchmark library. Monocle2
# runs in a self-contained R 4.4 conda environment and does not enter here.
if (method == "Monocle3") {
  shared_library <- Sys.getenv("SCCGRL_SHARED_R_LIBRARY")
  if (dir.exists(shared_library)) .libPaths(c(r_library, shared_library, .libPaths()))
}

meta <- read.csv(file.path(input_dir, "cell_metadata.csv"), stringsAsFactors = FALSE, check.names = FALSE)
features <- read.csv(file.path(input_dir, "feature_metadata.csv"), stringsAsFactors = FALSE, check.names = FALSE)
cell_ids <- as.character(meta$cell_id)
gene_ids <- make.unique(as.character(features$gene_id))
root_index <- match(root_cell_id, cell_ids)
if (is.na(root_index)) stop("Shared root cell is absent from baseline input")
expression <- methods::as(Matrix::readMM(file.path(input_dir, "expression.mtx")), "CsparseMatrix")
rownames(expression) <- cell_ids
colnames(expression) <- gene_ids
expression_is_counts <- identical(trimws(readLines(file.path(input_dir, "expression_is_counts.txt"))), "true")

normalize01 <- function(x) {
  x <- as.numeric(x)
  finite <- is.finite(x)
  if (!any(finite)) stop("No finite pseudotime")
  x[!finite] <- median(x[finite])
  span <- max(x) - min(x)
  if (span <= 0) return(rep(0, length(x)))
  (x - min(x)) / span
}
orient_root <- function(x) {
  x <- normalize01(x)
  if (x[[root_index]] > 0.5) x <- 1 - x
  normalize01(x)
}
write_standard <- function(pt, branch, details, lineage = NULL,
                           preprocessing_seconds = NA_real_, inference_seconds = NA_real_) {
  pt <- orient_root(pt)
  branch <- as.character(branch)
  if (length(branch) != length(pt)) branch <- rep("trajectory", length(pt))
  out <- data.frame(
    cell_id = cell_ids, pseudotime = pt, branch = branch,
    method_details = details,
    method_preprocessing_seconds = preprocessing_seconds,
    method_inference_seconds = inference_seconds,
    stringsAsFactors = FALSE
  )
  write.csv(out, output_csv, row.names = FALSE, fileEncoding = "UTF-8")
  if (!is.null(lineage)) {
    lineage <- as.data.frame(lineage)
    lineage <- cbind(cell_id = cell_ids, lineage)
    write.csv(lineage, lineage_csv, row.names = FALSE, fileEncoding = "UTF-8")
  }
}
log_normalized <- function() {
  matrix <- expression
  if (expression_is_counts) {
    totals <- Matrix::rowSums(matrix)
    totals[totals <= 0] <- 1
    matrix <- Matrix::Diagonal(x = 1e4 / totals) %*% matrix
    if (length(matrix@x)) matrix@x <- log1p(matrix@x)
  }
  matrix
}
top_variable <- function(matrix, n = 2000L) {
  means <- Matrix::colMeans(matrix)
  squared <- Matrix::colMeans(matrix ^ 2)
  variances <- pmax(0, squared - means ^ 2)
  order(variances, decreasing = TRUE)[seq_len(min(n, length(variances)))]
}
cluster_mclust <- function(pca) {
  suppressPackageStartupMessages(library(mclust))
  fit <- mclust::Mclust(pca)
  as.character(fit$classification)
}

if (method == "Monocle3") {
  preprocessing_started <- proc.time()[["elapsed"]]
  suppressPackageStartupMessages(library(monocle3))
  counts <- expression
  if (!expression_is_counts) {
    # Monocle3 requires a nonnegative expression matrix. No inverse transform
    # is invented; the source is recorded as already normalized expression.
    if (length(counts@x) && min(counts@x) < 0) stop("Monocle3 input is negative")
  }
  gene_meta <- data.frame(gene_short_name = gene_ids, row.names = gene_ids)
  cell_meta <- data.frame(row.names = cell_ids)
  cds <- monocle3::new_cell_data_set(Matrix::t(counts), cell_metadata = cell_meta, gene_metadata = gene_meta)
  cds <- monocle3::preprocess_cds(cds, num_dim = min(50L, nrow(cds) - 1L))
  cds <- monocle3::reduce_dimension(cds)
  preprocessing_seconds <- proc.time()[["elapsed"]] - preprocessing_started
  inference_started <- proc.time()[["elapsed"]]
  # Monocle3 1.0.0 can produce NA partition significance links with current
  # igraph; the documented partition-free mode keeps the official Leiden
  # clusters and avoids inventing a patched graph implementation.
  # Use the current official default clustering/partitioning behavior.  The
  # previous forced partition_qval=1 was inherited from the obsolete 1.0.0
  # workaround and can create invalid multi-component principal graphs.
  cds <- monocle3::cluster_cells(cds)
  # Learn one graph over the full unsupervised cell manifold.  This is the
  # documented `use_partition = FALSE` option and avoids the current Windows
  # `multi_component_RGE/connect_tips` failure for small disconnected Leiden
  # partitions without supplying labels, endpoints, or branch counts.
  cds <- monocle3::learn_graph(cds, use_partition = FALSE, close_loop = FALSE)
  cds <- monocle3::order_cells(cds, root_cells = root_cell_id)
  pt <- monocle3::pseudotime(cds)
  closest <- cds@principal_graph_aux[["UMAP"]]$pr_graph_cell_proj_closest_vertex
  closest <- as.character(closest[cell_ids, 1])
  graph <- monocle3::principal_graph(cds)[["UMAP"]]
  graph_names <- igraph::V(graph)$name
  if (!all(closest %in% graph_names) && all(paste0("Y_", closest) %in% graph_names)) closest <- paste0("Y_", closest)
  degrees <- igraph::degree(graph)
  leaves <- names(degrees)[degrees == 1]
  weights <- igraph::edge_attr(graph, "weight")
  use_weights <- if (is.null(weights) || any(!is.finite(weights) | weights < 0)) NA else weights
  distances <- igraph::distances(graph, v = unique(closest), to = leaves, weights = use_weights)
  assignment <- leaves[max.col(-as.matrix(distances), ties.method = "first")]
  names(assignment) <- rownames(distances)
  write_standard(pt, paste0("leaf_", assignment[closest]),
                 "official:preprocess_cds->reduce_dimension->cluster_cells->learn_graph(use_partition=FALSE,close_loop=FALSE)->order_cells;terminal_prior=false",
                 preprocessing_seconds = preprocessing_seconds,
                 inference_seconds = proc.time()[["elapsed"]] - inference_started)
}

if (method == "Monocle2") {
  preprocessing_started <- proc.time()[["elapsed"]]
  suppressPackageStartupMessages(library(monocle))
  suppressPackageStartupMessages(library(Biobase))
  matrix <- expression
  pd <- new("AnnotatedDataFrame", data = data.frame(row.names = cell_ids))
  fd <- new("AnnotatedDataFrame", data = data.frame(gene_short_name = gene_ids, row.names = gene_ids))
  if (expression_is_counts) {
    cds <- monocle::newCellDataSet(Matrix::t(matrix), phenoData = pd, featureData = fd,
                                   expressionFamily = VGAM::negbinomial.size())
    cds <- estimateSizeFactors(cds)
    cds <- estimateDispersions(cds)
  } else {
    cds <- monocle::newCellDataSet(Matrix::t(matrix), phenoData = pd, featureData = fd,
                                   expressionFamily = VGAM::uninormal())
  }
  cds <- monocle::detectGenes(cds, min_expr = 0.1)
  expressed <- row.names(subset(Biobase::fData(cds), num_cells_expressed >= max(10, 0.01 * length(cell_ids))))
  if (length(expressed) < 10L) expressed <- gene_ids[top_variable(matrix)]
  cds <- monocle::setOrderingFilter(cds, expressed)
  preprocessing_seconds <- proc.time()[["elapsed"]] - preprocessing_started
  inference_started <- proc.time()[["elapsed"]]
  reduction_arguments <- list(cds = cds, reduction_method = "DDRTree")
  if (!expression_is_counts) reduction_arguments$norm_method <- "none"
  cds <- do.call(monocle::reduceDimension, reduction_arguments)
  cds <- monocle::orderCells(cds)
  root_state <- as.character(Biobase::pData(cds)$State[[root_index]])
  cds <- monocle::orderCells(cds, root_state = root_state)
  write_standard(Biobase::pData(cds)$Pseudotime, paste0("state_", Biobase::pData(cds)$State),
                 paste0("official:CellDataSet->ordering_genes->DDRTree->orderCells;root_state=", root_state),
                 preprocessing_seconds = preprocessing_seconds,
                 inference_seconds = proc.time()[["elapsed"]] - inference_started)
}

if (method == "Slingshot") {
  preprocessing_started <- proc.time()[["elapsed"]]
  suppressPackageStartupMessages(library(slingshot))
  matrix <- log_normalized()
  selected <- top_variable(matrix)
  # The official Slingshot vignette performs GMM clustering on the first two
  # PCA coordinates (rd1 <- pca$x[, 1:2]); retain that recommended input.
  pca_full <- prcomp(as.matrix(matrix[, selected, drop = FALSE]),
                     rank. = min(50L, length(selected), length(cell_ids) - 1L))$x
  pca <- pca_full[, seq_len(min(2L, ncol(pca_full))), drop = FALSE]
  clusters <- cluster_mclust(pca)
  start_cluster <- clusters[[root_index]]
  preprocessing_seconds <- proc.time()[["elapsed"]] - preprocessing_started
  inference_started <- proc.time()[["elapsed"]]
  fit <- slingshot::slingshot(pca, clusterLabels = clusters, start.clus = start_cluster)
  lineage <- slingshot::slingPseudotime(fit, na = TRUE)
  weights <- slingshot::slingCurveWeights(fit, as.probs = TRUE)
  if (is.null(dim(lineage))) lineage <- matrix(lineage, ncol = 1L)
  if (is.null(dim(weights))) weights <- matrix(weights, ncol = 1L)
  chosen <- max.col(weights, ties.method = "first")
  pt <- vapply(seq_len(nrow(lineage)), function(i) lineage[i, chosen[[i]]], numeric(1))
  missing <- !is.finite(pt)
  if (any(missing)) pt[missing] <- apply(lineage[missing, , drop = FALSE], 1L, function(x) mean(x[is.finite(x)]))
  write_standard(pt, paste0("lineage_", chosen),
                 paste0("official:PCA+mclust+slingshot;start.clus=", start_cluster, ";end.clus=NULL"), lineage,
                 preprocessing_seconds = preprocessing_seconds,
                 inference_seconds = proc.time()[["elapsed"]] - inference_started)
}

if (method == "TSCAN") {
  suppressPackageStartupMessages(library(TSCAN))
  preprocessing_started <- proc.time()[["elapsed"]]
  # Reproduce TSCAN::preprocess() defaults on a sparse gene-by-cell matrix.
  # The package implementation first densifies the full matrix and then
  # filters genes, which is unnecessarily expensive for bone marrow.  These
  # operations are algebraically identical to its defaults:
  #   log2(count + 1), expression fraction > 0.5 at value > 1, and CV > 1.
  # Only the retained genes are converted to a dense matrix for exprmclust.
  transformed <- methods::as(Matrix::t(expression), "CsparseMatrix")
  if (expression_is_counts && length(transformed@x)) {
    transformed@x <- log2(transformed@x + 1)
  }
  cell_count <- ncol(transformed)
  gene_means <- Matrix::rowMeans(transformed)
  gene_second_moments <- Matrix::rowMeans(transformed ^ 2)
  gene_variances <- pmax(0, gene_second_moments - gene_means ^ 2)
  if (cell_count > 1L) gene_variances <- gene_variances * cell_count / (cell_count - 1L)
  gene_cv <- sqrt(gene_variances) / gene_means
  expressed_fraction <- Matrix::rowMeans(transformed > 1)
  official_keep <- expressed_fraction > 0.5 & is.finite(gene_cv) & gene_cv > 1
  tscan_preprocess_parameters <- "minexpr_value=1;minexpr_percent=0.5;cvcutoff=1"
  if (sum(official_keep) < 2L) {
    # TSCAN exposes cvcutoff as an official preprocessing parameter.  Its
    # historical default of 1 can remove every gene from sparse UMI data after
    # log2(count+1).  Apply one fixed, label-free relaxation while preserving
    # the expression threshold and required cell fraction.
    official_keep <- expressed_fraction > 0.5 & is.finite(gene_cv) & gene_cv > 0.5
    tscan_preprocess_parameters <- paste0(
      "minexpr_value=1;minexpr_percent=0.5;cvcutoff=0.5;",
      "fallback_reason=default_retained_fewer_than_two_genes"
    )
  }
  if (sum(official_keep) < 2L) stop("TSCAN preprocessing retained fewer than two genes after fixed CV fallback")
  processed <- as.matrix(transformed[official_keep, , drop = FALSE])
  tscan_filter <- paste0(
    "official_preprocess_sparse_equivalent;", tscan_preprocess_parameters,
    ";retained_genes=", nrow(processed)
  )
  # TSCAN's PCA uses scale=TRUE and its MST requires finite cluster centres.
  # Removing non-finite and constant genes is numerical input validation, not
  # biological feature tuning, and uses no labels/reference pseudotime.
  finite_rows <- apply(processed, 1L, function(x) all(is.finite(x)))
  variances <- apply(processed, 1L, stats::var)
  valid_rows <- finite_rows & is.finite(variances) & variances > 0
  processed <- processed[valid_rows, , drop = FALSE]
  tscan_filter <- paste0(tscan_filter, ";finite_nonconstant_genes")
  if (nrow(processed) < 2L) stop("TSCAN retained fewer than two nonconstant genes")
  valid_tscan_model <- function(candidate) {
    if (is.null(candidate) || is.null(candidate$clusterid) || is.null(candidate$pcareduceres)) return(FALSE)
    ids <- as.integer(candidate$clusterid)
    centres <- candidate$clucenter
    length(ids) == ncol(processed) && all(is.finite(ids)) &&
      length(unique(ids)) >= 2L && !is.null(centres) && all(is.finite(centres))
  }
  # VVV estimates a separate full covariance matrix for every component.  On
  # large cell collections it is both prohibitively slow and prone to empty
  # hard-assignment components, which make TSCAN's cluster-distance matrix NA.
  # Use the fixed numerical-recovery branch below before VVV for large inputs;
  # this threshold depends only on input size and never on labels or scores.
  use_regularized_first <- ncol(processed) >= 5000L
  default_error <- if (use_regularized_first) {
    "VVV skipped by fixed n_cells>=5000 numerical guard"
  } else {
    NULL
  }
  model <- NULL
  if (!use_regularized_first) {
    model <- tryCatch(
      TSCAN::exprmclust(processed),
      error = function(e) {
        default_error <<- conditionMessage(e)
        NULL
      }
    )
  }
  if (!use_regularized_first && !valid_tscan_model(model)) {
    # exprmclust officially exposes clusternum=2:9.  If its default model
    # search returns a degenerate/empty component, retry those supported
    # cluster counts in a fixed order and take the first numerically valid
    # model.  This recovery never consults labels, endpoints or evaluation.
    model <- NULL
    for (candidate_g in 2:9) {
      candidate <- tryCatch(
        TSCAN::exprmclust(processed, clusternum = candidate_g),
        error = function(e) NULL
      )
      if (valid_tscan_model(candidate)) {
        model <- candidate
        tscan_filter <- paste0(
          tscan_filter,
          ";degenerate_default_fallback_clusternum=", candidate_g
        )
        break
      }
    }
  }
  if (!valid_tscan_model(model)) {
    # TSCAN 1.x hard-codes mclust's unconstrained VVV covariance model inside
    # exprmclust(), even though exprmclust exposes a modelNames argument.  VVV
    # can be singular on low-rank simulated trajectories.  Preserve TSCAN's
    # standardized PCA -> model clustering -> cluster-centre MST workflow, but
    # use the equal-covariance EEE family.  For inputs covered by the fixed
    # large-data guard, use TSCAN's supported clusternum=2 setting so the
    # recovery is bounded; smaller failed inputs retain BIC selection over
    # G=2:9.  Neither choice uses labels, endpoints, or benchmark scores.
    # reference pseudotime, endpoints, or benchmark scores.
    suppressPackageStartupMessages(library(mclust))
    suppressPackageStartupMessages(library(igraph))
    build_regularized_tscan_model <- function(data) {
      # Match exprmclust's standardized PCA and elbow rule while avoiding its
      # duplicate full PCA decomposition.
      standardized <- t(apply(data, 1L, scale))
      standardized[!is.finite(standardized)] <- 0
      pca_fit <- stats::prcomp(t(standardized), scale. = TRUE)
      available <- min(20L, length(pca_fit$sdev))
      if (available < 2L) return(NULL)
      x <- seq_len(available)
      bend_candidates <- 2:min(10L, available)
      bend_loss <- vapply(bend_candidates, function(i) {
        x2 <- pmax(0, x - i)
        sum(stats::lm(pca_fit$sdev[seq_len(available)] ~ x + x2)$residuals^2)
      }, numeric(1))
      pcadim <- min(bend_candidates[[which.min(bend_loss)]] + 1L, available)
      pcadim <- min(pcadim, ncol(pca_fit$rotation))
      pca_reduced <- t(standardized) %*% pca_fit$rotation[, seq_len(pcadim), drop = FALSE]
      rownames(pca_reduced) <- colnames(data)
      set.seed(12345)
      recovery_g <- if (ncol(data) >= 5000L) 2L else 2:9
      fit <- tryCatch(
        mclust::Mclust(pca_reduced, G = recovery_g, modelNames = "EEE"),
        error = function(e) NULL
      )
      if (is.null(fit) || is.null(fit$classification)) return(NULL)
      cluster_id <- as.integer(fit$classification)
      names(cluster_id) <- rownames(pca_reduced)
      observed <- sort(unique(cluster_id))
      if (length(observed) < 2L) return(NULL)
      centres <- do.call(rbind, lapply(observed, function(cid) {
        colMeans(pca_reduced[cluster_id == cid, , drop = FALSE])
      }))
      rownames(centres) <- as.character(observed)
      if (any(!is.finite(centres))) return(NULL)
      distances <- as.matrix(stats::dist(centres))
      graph <- igraph::graph_from_adjacency_matrix(
        distances, mode = "undirected", weighted = TRUE, diag = FALSE
      )
      mst <- igraph::mst(graph, weights = igraph::E(graph)$weight)
      # TSCAN 1.50 mapCellsToEdges() expects the same vertex-coordinate
      # attribute produced by TrajectoryUtils::createClusterMST().  Legacy
      # exprmclust-style MST objects omit it, which silently maps every cell to
      # distance zero and yields constant pseudotime.
      coordinate_list <- lapply(seq_len(nrow(centres)), function(i) centres[i, ])
      names(coordinate_list) <- rownames(centres)
      igraph::V(mst)$coordinates <- coordinate_list[names(igraph::V(mst))]
      list(
        pcareduceres = pca_reduced,
        MSTtree = mst,
        clusterid = cluster_id,
        clucenter = centres,
        fallback_model_name = as.character(fit$modelName),
        fallback_cluster_count = length(observed),
        fallback_requested_g = paste(recovery_g, collapse = "-")
      )
    }
    model <- build_regularized_tscan_model(processed)
    if (valid_tscan_model(model)) {
      tscan_filter <- paste0(
        tscan_filter,
        ";regularized_mclust_BIC_model=", model$fallback_model_name,
        ";G=", model$fallback_cluster_count,
        ";requested_G=", model$fallback_requested_g
      )
    }
  }
  if (!valid_tscan_model(model)) {
    stop(paste0(
      "TSCAN clustering failed after VVV and fixed regularized-mclust recovery; default_error=",
      default_error
    ))
  }
  clusters <- as.character(model$clusterid[cell_ids])
  start_cluster <- as.character(clusters[[root_index]])
  pca <- model$pcareduceres[cell_ids, , drop = FALSE]
  mst <- model$MSTtree
  mapping <- TSCAN::mapCellsToEdges(pca, mst = mst, clusters = clusters)
  preprocessing_seconds <- proc.time()[["elapsed"]] - preprocessing_started
  inference_started <- proc.time()[["elapsed"]]
  ordering <- TSCAN::orderCells(mapping, mst, start = start_cluster)
  lineage <- as.matrix(SummarizedExperiment::assay(ordering))
  pt <- apply(lineage, 1L, function(x) {
    values <- x[is.finite(x)]
    if (length(values)) mean(values) else NA_real_
  })
  chosen <- apply(lineage, 1L, function(x) {
    candidates <- which(is.finite(x))
    if (length(candidates)) candidates[[which.max(x[candidates])]] else 1L
  })
  write_standard(pt, paste0("lineage_", chosen),
                 paste0("official:preprocess(defaults,conditional_log)->exprmclust(PCA+mclust+MST)->mapCellsToEdges->orderCells;filter=", tscan_filter, ";start=", start_cluster),
                 lineage,
                 preprocessing_seconds = preprocessing_seconds,
                 inference_seconds = proc.time()[["elapsed"]] - inference_started)
}

if (method == "SLICER") {
  preprocessing_started <- proc.time()[["elapsed"]]
  suppressPackageStartupMessages(library(SLICER))
  suppressPackageStartupMessages(library(lle))
  matrix <- as.matrix(log_normalized())
  genes <- SLICER::select_genes(matrix)
  message("SLICER stage: select_genes complete; selected_genes=", length(genes))
  if (length(genes) < 2L) genes <- top_variable(Matrix::Matrix(matrix, sparse = TRUE), 500L)
  selected <- matrix[, genes, drop = FALSE]
  k <- SLICER::select_k(selected)
  message("SLICER stage: select_k complete; k=", k)
  embedding <- lle::lle(selected, m = 2L, k = k)$Y
  message("SLICER stage: final LLE complete")
  graph <- SLICER::conn_knn_graph(embedding, 5L)
  message("SLICER stage: connected KNN graph complete")
  preprocessing_seconds <- proc.time()[["elapsed"]] - preprocessing_started
  inference_started <- proc.time()[["elapsed"]]
  order <- SLICER::cell_order(graph, root_index)
  pt <- rep(NA_real_, length(cell_ids)); pt[order] <- seq_along(order)
  branch <- tryCatch(SLICER::assign_branches(graph, root_index), error = function(e) rep("trajectory", length(pt)))
  write_standard(pt, branch, paste0("official:select_genes->select_k->LLE->KNN;LLE_k=", k, ";graph_k=5"),
                 preprocessing_seconds = preprocessing_seconds,
                 inference_seconds = proc.time()[["elapsed"]] - inference_started)
  message("SLICER stage: result written")
  quit(save = "no", status = 0L)
}

if (method == "SCORPIUS") {
  preprocessing_started <- proc.time()[["elapsed"]]
  suppressPackageStartupMessages(library(SCORPIUS))
  matrix <- as.matrix(log_normalized())
  space <- SCORPIUS::reduce_dimensionality(matrix)
  preprocessing_seconds <- proc.time()[["elapsed"]] - preprocessing_started
  inference_started <- proc.time()[["elapsed"]]
  fit <- SCORPIUS::infer_trajectory(space)
  write_standard(fit$time, rep("linear_trajectory", length(cell_ids)),
                 "official:reduce_dimensionality(defaults)->infer_trajectory(defaults);root_used_for_direction_only",
                 preprocessing_seconds = preprocessing_seconds,
                 inference_seconds = proc.time()[["elapsed"]] - inference_started)
}
