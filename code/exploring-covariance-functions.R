library(GauPro)
library(tidyverse)
library(latex2exp)

set.seed(1234)

# Create the data
n <- 15
x <- seq(0, 1, l=n)
dimensions <- 1

# SE: smooth sin wave
se_fx <- sin(2 * pi * x)

# RQ: multiple sin waves at different frequencies
rq_fx <- se_fx + 0.5 * sin(10 * pi * x)

# GE: intermediate roughness
ge_fx <- abs(se_fx)

# Exponential: non-differentiable
exp_fx <- function(x) {
  ifelse(x < 0.3,
      x,
      ifelse(x < 0.7,
        1 - x,     
        x - 0.5
      )
  )
}
exp_fx <- exp_fx(x)
  
# Matern 3/2: smooth wave + shallow kink
m32_fx <- se_fx + 0.2 * abs(x - 0.5)

# Matern 5/2: smooth wave + deep kink
m52_fx <- se_fx * exp(-x)


# Assemble all functions into vector
stationary_fxs <- cbind(
  se_fx,
  exp_fx
)
function_names <- c("Smooth", "Rough")

# Add noise to each column of results
sigma_n <- 0.05
noise_matrix <- matrix(
  rnorm(
    n = nrow(stationary_fxs) * ncol(stationary_fxs),
    mean = 0,
    sd = sigma_n
  ), ncol = ncol(stationary_fxs)
)

stationary_ys <- stationary_fxs + noise_matrix

# Define all kernels to use
stationary_kernels <- c(
  # k_Triangle(D=dimensions),
  k_Gaussian(D = dimensions),
  k_RatQuad(D = dimensions),
  k_PowerExp(D = dimensions),
  k_Exponential(D = dimensions),
  k_Matern32(D = dimensions),
  k_Matern52(D = dimensions)
)

nonstationary_kernels <- c(
  # k_Periodic(D = dimensions)
)

factor_kernels <- c()


# Fit a Gaussian process model using each stationary kernel and each supplied dataset

# For each kernel
stationary_models <- lapply(
  stationary_kernels,
  function(stationary_kernel) {
    
    # For each dataset
    kernel_models <- lapply(
      seq_along(function_names),
      function(dataset_index){
        # Fit the model to the stationary models data
        model <- gpkm(
          X = x,
          Z = stationary_ys[, dataset_index],
          kernel = stationary_kernel$clone(deep = TRUE)
        )
        
        # Add the function evaluations to the model object and return it
        attr(model, "function_fx") <- stationary_fxs[,dataset_index]
        attr(model, "function_name") <- function_names[dataset_index]
        
        return (model)
      }
    )
    
    return(kernel_models)
  }
)

# Flatten
stationary_models <- unlist(stationary_models, recursive = FALSE)

# Plot the results and print model summaries
models <- stationary_models

# Define xlim and ylims
x_bound <- 0.125
y_bound <- 0.35

limx <- c(min(x) - x_bound, max(x) + x_bound)
limy <- c(min(stationary_fxs) - y_bound, max(stationary_fxs) + y_bound)

dp <- 2

for (model in models) {
  
  # Collect info on kernel
  model_kernel <- model[["kernel"]]
  kernel_name <- str_split(attributes(model_kernel)$class[1], "_")[[1]][3]
  dataset_name <- attr(model, "function_name")
  
  # Get noise info
  signal_var <- round(model[["kernel"]]$s2, dp)
  noise_var <- round(model$nug * signal_var, dp)
  
  # Calculate length scale based on model beta
  beta <- model_kernel$beta
  length_scale <- round(exp(-beta/2), dp)
  
  title_str <- sprintf(
    " (kernel: %s, dataset: %s, $l$: %s, $\\sigma_f^2$: %s, $\\sigma_n^2$: %s",
    kernel_name, dataset_name, length_scale, signal_var, noise_var
  )
  
  # Add model parameters if applicable
  alpha <- model_kernel$alpha
  if (!is.null(alpha)){
    title_str <- paste0(title_str, ", $\\alpha$: ", round(alpha, dp))
  }
  
  title_str <- paste0(title_str, ")")
  
  # Extract data-generating function and input data, and produce a line
  model.X_fx_data <- data.frame(
    X  = x,
    fx = attr(model, "function_fx")
  )
  
  model.X_fx <- geom_line(
    data = model.X_fx_data,              
    aes(x = X, y = fx),           
    inherit.aes = FALSE,
    color = "blue",
    size = 1.5
  )
  
  # Plot variance in data and in model alongside expected function
  p1 <- model$plot1D() + 
    ggtitle(
      TeX(
        paste0(
          "Mean and per-sample variances", title_str
        )
      )
    ) + 
    model.X_fx # +
    # xlim(limx) + ylim(limy)
  
  # Plot some function drawn from predictive distribution
  p2 <- model$cool1Dplot() + 
    ggtitle(
      TeX(
        paste0(
          "Draws from function distribution", title_str
        )
      )
    ) + 
    model.X_fx # +
    # xlim(limx) + ylim(limy)
  
  # Plot all graphs
  directory <- "img/"
  metadata_str <- paste0(
    directory, kernel_name, "_", dataset_name
  )
  
  px_width <- 3207
  px_height <- 2181
  dpi <- 300
  
  ggsave(
    filename = paste0(metadata_str, "_mean.png"),
    plot = p1,
    width = px_width,
    height = px_height,
    units = "px",
    dpi = dpi
  )     
  
  ggsave(
    filename = paste0(metadata_str, "_draws.png"),
    plot = p2,
    width = px_width,
    height = px_height,
    units = "px",
    dpi = dpi
  )     
  
  # Print model info
  print(metadata_str)
  print(model$summary())
  print("\n\n")
}
