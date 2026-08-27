# Exact audited historical Monocle1 runner, patched through 2026-08-21.
# Historical Monocle 1.0.0 runner. This script deliberately uses only the
# original package API. It must be executed in the R 3.1 / Bioconductor 3.0
# environment described in environments/monocle1_r.yml.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6L) stop("Usage: runner.R INPUT_DIR OUTPUT_CSV LINEAGE_CSV SEED ROOT_CELL_ID R_LIBRARY")
input_dir <- args[[1L]]; output_csv <- args[[2L]]; seed <- as.integer(args[[4L]])
root_cell_id <- args[[5L]]; r_library <- args[[6L]]
.libPaths(c(r_library, .libPaths())); set.seed(seed)
# Large modern datasets can create a much deeper recursive P-Q decomposition
# than the small datasets for which R 3.1's default expression limit was set.
# A strict subgraph-size guard below ensures that raising this limit cannot
# conceal a non-shrinking recursion cycle.
options(expressions = 500000L)
options(error = function() {
  traceback(50L)
  quit(save = "no", status = 1L, runLast = FALSE)
})
suppressPackageStartupMessages(library(monocle))
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(Biobase))
if (as.character(packageVersion("monocle")) != "1.0.0") {
  stop(paste0("Historical Monocle1 environment required; found monocle ", packageVersion("monocle")))
}

# Monocle 1.0.0 asks irlba for min(n_cells, n_genes) singular vectors inside
# ica_helper(), even though reduceDimension() only requests max_components=2.
# Modern benchmark matrices commonly have at least as many cells as retained
# genes, so that historical call requests the complete rank of a square
# covariance matrix.  irlba 1.0.3 correctly rejects this because a truncated
# SVD rank must be strictly smaller than the matrix dimension.  Falling back
# to Monocle's full La.svd path requires a dense genes-by-genes decomposition
# and failed with memory/recursion errors on these data.
#
# The compatibility helper below is otherwise the Monocle 1.0.0 implementation
# verbatim; the only numerical correction is to request n.comp singular
# vectors, which are the only vectors consumed by the subsequent ICA step.
# This preserves ICA, the cell-level MST, orderCells(), and all method defaults.
monocle1_ica_helper_compat <- function(
    X, n.comp, alg.typ = c("parallel", "deflation"),
    fun = c("logcosh", "exp"), alpha = 1, row.norm = TRUE,
    maxit = 200, tol = 1e-04, verbose = FALSE, w.init = NULL,
    use_irlba = TRUE) {
  dd <- dim(X)
  d <- dd[dd != 1L]
  if (length(d) != 2L) stop("data must be matrix-conformal")
  X <- if (length(d) != length(dd)) matrix(X, d[1L], d[2L]) else as.matrix(X)
  if (alpha < 1 || alpha > 2) stop("alpha must be in range [1,2]")
  alg.typ <- match.arg(alg.typ)
  fun <- match.arg(fun)
  n <- nrow(X)
  p <- ncol(X)
  if (n.comp > min(n, p)) {
    message("'n.comp' is too large: reset to ", min(n, p))
    n.comp <- min(n, p)
  }
  if (n.comp >= min(n, p)) {
    stop("Monocle1 ICA requires at least one dimension beyond max_components")
  }
  if (is.null(w.init)) {
    w.init <- matrix(rnorm(n.comp^2), n.comp, n.comp)
  } else if (!is.matrix(w.init) || length(w.init) != n.comp^2) {
    stop("w.init is not a matrix or is the wrong size")
  }
  if (verbose) message("Centering")
  X <- scale(X, scale = FALSE)
  X <- if (row.norm) t(scale(X, scale = row.norm)) else t(X)
  if (verbose) message("Whitening")
  V <- X %*% t(X) / n
  if (verbose) message("Finding truncated SVD")
  if (use_irlba) {
    s <- irlba::irlba(V, nu = n.comp, nv = n.comp)
    svs <- s$d
  } else {
    s <- La.svd(V, nu = n.comp, nv = n.comp)
    svs <- s$d
    s$u <- s$u[, seq_len(n.comp), drop = FALSE]
  }
  D <- diag(c(1 / sqrt(s$d[seq_len(n.comp)])))
  K <- D %*% t(s$u[, seq_len(n.comp), drop = FALSE])
  K <- matrix(K[seq_len(n.comp), , drop = FALSE], n.comp, p)
  X1 <- K %*% X
  if (verbose) message("Running ICA")
  if (alg.typ == "deflation") {
    a <- fastICA:::ica.R.def(X1, n.comp, tol = tol, fun = fun,
                             alpha = alpha, maxit = maxit,
                             verbose = verbose, w.init = w.init)
  } else {
    a <- fastICA:::ica.R.par(X1, n.comp, tol = tol, fun = fun,
                             alpha = alpha, maxit = maxit,
                             verbose = verbose, w.init = w.init)
  }
  w <- a %*% K
  S <- w %*% X
  A <- t(w) %*% solve(w %*% t(w))
  list(X = t(X), K = t(K), W = t(a), A = t(A), S = t(S), svs = svs)
}

# Monocle 1.0.0's pq_helper() assembles its recursive P-Q tree with the old
# `graph + vertex()` / `graph + edge(name, name)` igraph DSL.  On real,
# highly-branched cell MSTs, recursive subtree merging can expose duplicate
# vertex names to that DSL; subsequent name-based edge insertion then fails
# with "Invalid vertex names".  The historical algorithm itself is retained
# below.  Only graph assembly is made explicit: vertices are unique by name
# and edges are inserted using validated numeric vertex indices.
monocle1_add_vertex_compat <- function(g, vertex_name, type, color,
                                       diam_path_len = NULL) {
  vertex_name <- as.character(vertex_name)[1L]
  if (is.na(vertex_name) || !nzchar(vertex_name)) {
    stop("Monocle1 P-Q tree attempted to add an unnamed vertex")
  }
  current_names <- V(g)$name
  if (!(vertex_name %in% current_names)) {
    attrs <- list(name = vertex_name, type = type, color = color)
    if (!is.null(diam_path_len)) attrs$diam_path_len <- diam_path_len
    g <- add.vertices(g, 1L, attr = attrs)
  }
  g
}

monocle1_add_edge_compat <- function(g, from_name, to_name) {
  endpoint_names <- as.character(c(from_name, to_name))
  vertex_names <- V(g)$name
  endpoint_idx <- match(endpoint_names, vertex_names)
  if (any(is.na(endpoint_idx))) {
    stop(paste0(
      "Monocle1 P-Q tree edge endpoint missing: ",
      paste(endpoint_names[is.na(endpoint_idx)], collapse = ", ")
    ))
  }
  add.edges(g, endpoint_idx)
}

monocle1_pq_helper_compat <- function(mst, use_weights = TRUE,
                                      root_node = NULL,
                                      parent_vertex_count = Inf) {
  new_subtree <- graph.empty()
  jobs <- list(list(
    mst = mst,
    use_weights = use_weights,
    root_node = root_node,
    attach_parent = NULL,
    parent_vertex_count = parent_vertex_count
  ))
  top_root_id <- NULL

  while (length(jobs) > 0L) {
    job_idx <- length(jobs)
    job <- jobs[[job_idx]]
    jobs[[job_idx]] <- NULL
    current_mst <- job$mst

    if (vcount(current_mst) >= job$parent_vertex_count) {
      stop(paste0(
        "Monocle1 P-Q decomposition did not shrink: child=",
        vcount(current_mst), ", parent=", job$parent_vertex_count
      ))
    }

    root_node_id <- paste("Q_", monocle:::get_next_node_id(), sep = "")
    if (is.null(top_root_id)) top_root_id <- root_node_id
    new_subtree <- monocle1_add_vertex_compat(
      new_subtree, root_node_id, type = "Q", color = "black"
    )
    if (!is.null(job$attach_parent)) {
      new_subtree <- monocle1_add_edge_compat(
        new_subtree, job$attach_parent, root_node_id
      )
    }

    if (!is.null(job$root_node)) {
      sp <- get.all.shortest.paths(
        current_mst, from = V(current_mst)[job$root_node]
      )
      sp_lengths <- sapply(sp$res, length)
      target_node_idx <- which(sp_lengths == max(sp_lengths))[1L]
      diam <- V(current_mst)[unlist(sp$res[target_node_idx])]
    } else if (job$use_weights) {
      diam <- V(current_mst)[get.diameter(current_mst)]
    } else {
      diam <- V(current_mst)[get.diameter(current_mst, weights = NA)]
    }

    V(new_subtree)[root_node_id]$diam_path_len <- length(diam)
    diam_decisiveness <- igraph::degree(current_mst, v = diam) > 2
    ind_nodes <- diam_decisiveness[diam_decisiveness == TRUE]
    first_diam_path_node_idx <- head(as.vector(diam), n = 1L)
    last_diam_path_node_idx <- tail(as.vector(diam), n = 1L)

    if (sum(ind_nodes) == 0 ||
        (igraph::degree(current_mst, first_diam_path_node_idx) == 1 &&
         igraph::degree(current_mst, last_diam_path_node_idx) == 1)) {
      ind_backbone <- diam
    } else {
      last_bb_point <- names(tail(ind_nodes, n = 1L))[[1L]]
      first_bb_point <- names(head(ind_nodes, n = 1L))[[1L]]
      diam_path_names <- V(current_mst)[as.vector(diam)]$name
      last_bb_point_idx <- which(diam_path_names == last_bb_point)[1L]
      first_bb_point_idx <- which(diam_path_names == first_bb_point)[1L]
      ind_backbone_idxs <- as.vector(diam)[
        first_bb_point_idx:last_bb_point_idx
      ]
      ind_backbone <- V(current_mst)[ind_backbone_idxs]
    }

    mst_no_backbone <- current_mst - ind_backbone
    child_jobs <- list()
    for (backbone_n in ind_backbone) {
      backbone_name <- V(current_mst)[backbone_n]$name
      if (igraph::degree(current_mst, v = backbone_n) > 2) {
        new_p_id <- paste("P_", monocle:::get_next_node_id(), sep = "")
        new_subtree <- monocle1_add_vertex_compat(
          new_subtree, new_p_id, type = "P", color = "grey"
        )
        new_subtree <- monocle1_add_vertex_compat(
          new_subtree, backbone_name, type = "leaf", color = "white"
        )
        new_subtree <- monocle1_add_edge_compat(
          new_subtree, new_p_id, backbone_name
        )
        new_subtree <- monocle1_add_edge_compat(
          new_subtree, root_node_id, new_p_id
        )

        nb <- graph.neighborhood(current_mst, 1, nodes = backbone_n)[[1L]]
        for (n_i in V(nb)) {
          n <- V(nb)[n_i]$name
          if (n %in% V(mst_no_backbone)$name) {
            sc <- subcomponent(mst_no_backbone, n)
            sg <- induced.subgraph(
              mst_no_backbone, sc, impl = "copy_and_delete"
            )
            if (ecount(sg) > 0) {
              if (vcount(sg) >= vcount(current_mst)) {
                stop(paste0(
                  "Monocle1 P-Q decomposition produced a non-shrinking ",
                  "subgraph: child=", vcount(sg),
                  ", parent=", vcount(current_mst)
                ))
              }
              child_jobs[[length(child_jobs) + 1L]] <- list(
                mst = sg,
                use_weights = job$use_weights,
                root_node = NULL,
                attach_parent = new_p_id,
                parent_vertex_count = vcount(current_mst)
              )
            } else {
              new_subtree <- monocle1_add_vertex_compat(
                new_subtree, n, type = "leaf", color = "white"
              )
              new_subtree <- monocle1_add_edge_compat(
                new_subtree, new_p_id, n
              )
            }
          }
        }
      } else {
        new_subtree <- monocle1_add_vertex_compat(
          new_subtree, backbone_name, type = "leaf", color = "white"
        )
        new_subtree <- monocle1_add_edge_compat(
          new_subtree, root_node_id, backbone_name
        )
      }
    }

    # LIFO with reversed insertion preserves the original depth-first order.
    if (length(child_jobs) > 0L) {
      for (child_idx in rev(seq_along(child_jobs))) {
        jobs[[length(jobs) + 1L]] <- child_jobs[[child_idx]]
      }
    }
  }
  list(root = top_root_id, subtree = new_subtree)
}

# The final Monocle1 pseudotime assignment recursively walks every cell in a
# directed ordering tree.  A long real-data path can exceed the R 3.1 protect
# stack even after the P-Q decomposition itself is iterative.  Replace only
# that nested walker with an explicit LIFO stack; edge-distance accumulation
# and the resulting pseudotime values are identical to the historical code.
monocle1_extract_good_branched_ordering_compat <-
  monocle:::extract_good_branched_ordering
extract_body <- as.list(body(monocle1_extract_good_branched_ordering_compat))
branch_loop_idx <- which(vapply(
  extract_body,
  function(expr) {
    is.call(expr) && identical(expr[[1L]], as.name("for")) &&
      grepl("num_branches", paste(deparse(expr), collapse = " "))
  },
  logical(1L)
))
if (length(branch_loop_idx) != 1L) {
  stop("Unable to locate Monocle1 branch-tree construction loop")
}
extract_body[[branch_loop_idx]] <- quote(
  for (i in seq_len(num_branches)) {
    branch_point_roots[[length(branch_point_roots) + 1L]] <-
      names(branch_node_counts)[i]
    branch_id <- names(branch_node_counts)[i]
    if (!(branch_id %in% V(branch_tree)$name)) {
      branch_tree <- add.vertices(
        branch_tree, 1L, attr = list(name = branch_id)
      )
    }
    parents <- V(pq_tree)[nei(branch_id, mode = "in")]
    if (length(parents) > 0L && parents$type[[1L]] == "P") {
      p_node_parent <- parents[[1L]]
      parent_branch_id <- V(pq_tree)[nei(p_node_parent, mode = "in")]$name
      # The historical code tried to add this edge even when the parent Q
      # branch was not among the selected `num_branches`, causing igraph's
      # "Invalid vertex names" error.  An unselected parent is intentionally
      # absent from the reduced branch tree, so the selected branch is a root.
      if (length(parent_branch_id) > 0L &&
          parent_branch_id[[1L]] %in% V(branch_tree)$name) {
        edge_idx <- match(
          c(parent_branch_id[[1L]], branch_id), V(branch_tree)$name
        )
        branch_tree <- add.edges(branch_tree, edge_idx)
      }
    }
    incoming <- V(pq_tree)[nei(branch_id, mode = "in")]
    if (length(incoming) > 0L) {
      pq_tree[incoming, branch_id] <- FALSE
    }
  }
)
curr_branch_idx <- which(vapply(
  extract_body,
  function(expr) {
    is.call(expr) && identical(expr[[1L]], as.name("<-")) &&
      identical(as.character(expr[[2L]]), "curr_branch")
  },
  logical(1L)
))
if (length(curr_branch_idx) != 1L) {
  stop("Unable to locate Monocle1 current branch initialization")
}
extract_body[[curr_branch_idx]] <- quote(
  curr_branch <- branch_point_roots[[1L]]
)
state_helper_idx <- which(vapply(
  extract_body,
  function(expr) {
    is.call(expr) && identical(expr[[1L]], as.name("<-")) &&
      identical(as.character(expr[[2L]]), "assign_cell_state_helper")
  },
  logical(1L)
))
if (length(state_helper_idx) != 1L) {
  stop("Unable to locate Monocle1 assign_cell_state_helper")
}
extract_body[[state_helper_idx]] <- quote(
  assign_cell_state_helper <- function(ordering_tree_res, curr_cell) {
    cell_tree <- ordering_tree_res$subtree
    # Each frame records the same information held by one historical
    # recursive invocation.  Siblings are processed sequentially so the
    # outer `curr_state` counter changes in exactly the original DFS order.
    stack <- list(list(
      cell = as.character(curr_cell),
      entered = FALSE,
      children = character(0L),
      next_child = 0L
    ))

    while (length(stack) > 0L) {
      frame_idx <- length(stack)
      frame <- stack[[frame_idx]]
      if (!frame$entered) {
        V(cell_tree)[frame$cell]$cell_state <- curr_state
        children <- V(cell_tree)[nei(frame$cell, mode = "out")]
        child_names <- if (length(children) > 0L) {
          V(cell_tree)[children]$name
        } else {
          character(0L)
        }
        frame$entered <- TRUE
        frame$children <- child_names
        frame$next_child <- 0L
        stack[[frame_idx]] <- frame

        if (length(child_names) == 0L) {
          stack[[frame_idx]] <- NULL
        } else {
          frame$next_child <- 1L
          stack[[frame_idx]] <- frame
          if (length(child_names) > 1L) curr_state <<- curr_state + 1L
          stack[[length(stack) + 1L]] <- list(
            cell = child_names[[1L]],
            entered = FALSE,
            children = character(0L),
            next_child = 0L
          )
        }
      } else if (frame$next_child >= length(frame$children)) {
        stack[[frame_idx]] <- NULL
      } else {
        frame$next_child <- frame$next_child + 1L
        stack[[frame_idx]] <- frame
        # This branch is reached only for another sibling.  The historical
        # recursive implementation increments before visiting every sibling.
        curr_state <<- curr_state + 1L
        stack[[length(stack) + 1L]] <- list(
          cell = frame$children[[frame$next_child]],
          entered = FALSE,
          children = character(0L),
          next_child = 0L
        )
      }
    }
    ordering_tree_res$subtree <- cell_tree
    ordering_tree_res
  }
)
pseudotime_helper_idx <- which(vapply(
  extract_body,
  function(expr) {
    is.call(expr) && identical(expr[[1L]], as.name("<-")) &&
      identical(as.character(expr[[2L]]), "assign_pseudotime_helper")
  },
  logical(1L)
))
if (length(pseudotime_helper_idx) != 1L) {
  stop("Unable to locate Monocle1 assign_pseudotime_helper")
}
extract_body[[pseudotime_helper_idx]] <- quote(
  assign_pseudotime_helper <- function(ordering_tree_res, dist_matrix,
                                        last_pseudotime, curr_cell) {
    cell_tree <- ordering_tree_res$subtree
    pending_cells <- as.character(curr_cell)
    pending_times <- as.numeric(last_pseudotime)
    visited <- rep(FALSE, vcount(cell_tree))
    names(visited) <- V(cell_tree)$name

    while (length(pending_cells) > 0L) {
      stack_idx <- length(pending_cells)
      current_cell <- pending_cells[[stack_idx]]
      current_time <- pending_times[[stack_idx]]
      pending_cells <- pending_cells[-stack_idx]
      pending_times <- pending_times[-stack_idx]
      if (isTRUE(visited[[current_cell]])) next
      visited[[current_cell]] <- TRUE
      V(cell_tree)[current_cell]$pseudotime <- current_time

      children <- V(cell_tree)[nei(current_cell, mode = "out")]
      if (length(children) > 0L) {
        child_names <- V(cell_tree)[children]$name
        child_times <- vapply(
          child_names,
          function(next_node) {
            current_time + dist_matrix[current_cell, next_node]
          },
          numeric(1L)
        )
        pending_cells <- c(pending_cells, child_names)
        pending_times <- c(pending_times, child_times)
      }
    }
    ordering_tree_res$subtree <- cell_tree
    ordering_tree_res
  }
)
body(monocle1_extract_good_branched_ordering_compat) <- as.call(extract_body)
environment(monocle1_extract_good_branched_ordering_compat) <-
  asNamespace("monocle")

# reduceDimension() resolves ica_helper in the monocle namespace.  Replace it
# only for this isolated R process; the installed package files are untouched.
assignInNamespace("ica_helper", monocle1_ica_helper_compat, ns = "monocle")
assignInNamespace("pq_helper", monocle1_pq_helper_compat, ns = "monocle")
assignInNamespace(
  "extract_good_branched_ordering",
  monocle1_extract_good_branched_ordering_compat,
  ns = "monocle"
)
meta <- read.csv(file.path(input_dir, "cell_metadata.csv"), stringsAsFactors = FALSE)
features <- read.csv(file.path(input_dir, "feature_metadata.csv"), stringsAsFactors = FALSE)
cell_ids <- as.character(meta$cell_id); gene_ids <- make.unique(as.character(features$gene_id))
root_index <- match(root_cell_id, cell_ids); if (is.na(root_index)) stop("Root cell absent")
expression <- as(Matrix::readMM(file.path(input_dir, "expression.mtx")), "dgCMatrix")
rownames(expression) <- cell_ids; colnames(expression) <- gene_ids
pd <- new("AnnotatedDataFrame", data = data.frame(row.names = cell_ids))
fd <- new("AnnotatedDataFrame", data = data.frame(gene_short_name = gene_ids, row.names = gene_ids))
cds <- newCellDataSet(t(expression), phenoData = pd, featureData = fd)
reduced_checkpoint <- Sys.getenv("MONOCLE1_REDUCED_RDS", unset = "")
if (nzchar(reduced_checkpoint) && file.exists(reduced_checkpoint)) {
  cds <- readRDS(reduced_checkpoint)
} else {
  cds <- reduceDimension(cds)
  if (nzchar(reduced_checkpoint)) saveRDS(cds, reduced_checkpoint)
}
cds <- withCallingHandlers(
  orderCells(cds, root_cell = root_index),
  error = function(condition) {
    message("Monocle1 orderCells call stack:")
    print(sys.calls())
  }
)
out <- data.frame(cell_id = cell_ids, pseudotime = pData(cds)$Pseudotime,
                  branch = paste0("state_", pData(cds)$State),
                  method_details = paste0(
                    "historical_monocle_1.0.0:ICA+cell_MST+orderCells;",
                    "compatibility_fix=truncated_irlba_rank_equals_max_components+",
                    "iterative_pq_tree+iterative_state_and_pseudotime_walks"
                  ),
                  stringsAsFactors = FALSE)
write.csv(out, output_csv, row.names = FALSE)
