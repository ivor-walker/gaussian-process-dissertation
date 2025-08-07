library(GauPro)
library(tidyverse)

set.seed(1234)

# Create the data
n <- 20
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
  
# Matern 3/2: smooth wave + shallow kink
m32_fx <- se_fx + 0.2 * abs(x - 0.5)

# Matern 5/2: smooth wave + deep kink
m52_fx <- se_fx * exp(-x)


# Assemble all functions into vector
stationary_fxs <- cbind(
  se_fx,
  rq_fx,
  ge_fx,
  exp_fx(x),
  m32_fx,
  m52_fx
)

# Add noise to all results
sigma_n <- 0.05
stationary_ys <- stationary_fxs + rnorm(n, sd = sigma_n)

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

dataset_index <- 1
# Fit a Gaussian process model using each supplied kernel and each supplied dataset
stationary_models <- lapply(
  stationary_kernels,
  function(kernel) {

    # Fit the model to the stationary models data
    model <- gpkm(x, ys[,dataset_index],
      kernel = kernel
    )
    
    # Add the function evaluations to the model object and return it
    attr(model, "fx") <- fxs[,dataset_index]
    
    dataset_index <<- dataset_index + 1
    
    return (model)
  }
)

nonstationary_models <- lapply(
  nonstationary_kernels,
  function(kernel) {
    
    model <- gpkm(x, y,
      kernel = kernel
    )
    
    attr(model, "fx") <- fx
    return (model)
  }
)

factor_models <- lapply(
  factor_kernels,
  function(kernel) {
    
    model <- gpkm(x, y,
      kernel = kernel
    )
    
    attr(model, "fx") <- fx
    return (model)
  }
)

# Plot the results and print model summaries
models <- c(stationary_models, nonstationary_models, factor_models)

# Define xlim and ylims
x_bound <- 0.125
y_bound <- 0.35

limx <- c(min(x) - x_bound, max(x) + x_bound)
limy <- c(min(fx) - y_bound, max(fx) + y_bound)

for (model in models) {
  kernel_name <- str_split(attributes(model[["kernel"]])$class[1], "_")[[1]][3]
  
  # Extract data-generating function and input data, and produce a line
  model.X_fx_data <- data.frame(
    X  = model$X,
    fx = attr(model, "fx")
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
      paste0("Mean and per-sample variances (kernel: ", kernel_name,")"
    )) + 
    model.X_fx # +
    # xlim(limx) + ylim(limy)
  
  # Plot some function drawn from predictive distribution
  p2 <- model$cool1Dplot() + 
    ggtitle(
      paste0("Draws from function distribution (kernel: ", kernel_name,")"
    )) + 
    model.X_fx # +
    # xlim(limx) + ylim(limy)
  
  # Plot all graphs
  print(p1)
  print(p2)
  
  # Print model info
  print(kernel_name)
  print(model$summary())
  print("\n\n")
}
