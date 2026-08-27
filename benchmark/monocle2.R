# Monocle 2 launcher. The audited implementation is shared to avoid duplication.
fixed_method <- "Monocle2"
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1]
script_dir <- dirname(normalizePath(sub("^--file=", "", script_arg), winslash = "/"))
source(file.path(script_dir, "common", "r_methods_runner.R"), chdir = TRUE)
