library(GauPro)
library(tidyverse)

set.seed(1234)

# Create the data
n <- 20
x <- seq(0, 1, l=n)

dimensions <- 1
fx <- abs(sin(2 * pi * x)) ^ .8

sigma_n <- 0.1
y <- fx + rnorm(n, sd = sigma_n)
ggplot(
  aes(x, y), 
  data=cbind(x, y)
) +
  geom_point()

# Fit a linear model first
ggplot(
  aes(x, y), 
  data=cbind(x, y)
) +
  geom_point() +
  geom_smooth(method="lm")
# Fits straight line through data
# Not good descriptor of underlying function

# Define all kernels to use
stationary_kernels <- c(
  # k_Triangle(D=dimensions),
  k_Gaussian(D = dimensions),
  k_Matern52(D = dimensions),
  k_Matern32(D = dimensions),
  k_Exponential(D = dimensions),
  k_PowerExp(D = dimensions),
  k_RatQuad(D = dimensions)
)

nonstationary_kernels <- c(
  k_Periodic(D = dimensions)
)

factor_kernels <- c()

# Fit a Gaussian process model using each supplied kernel
stationary_models <- lapply(
  stationary_kernels,
  function(kernel) {
    
    # Fit the model to the stationary models data
    model <- gpkm(x, y,
      kernel = kernel
    )
    
    # Add the function evaluations to the model object and return it
    attr(model, "fx") <- fx
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
