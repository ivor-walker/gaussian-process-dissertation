library(GauPro)
library(tidyverse)

# Create the data
n <- 100
x <- seq(0, 1, l=n)

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

# Fit a Gaussian process model
kern.exp <- k_Exponential(D=10)
gp <- gpkm(
  x, y,
  kernel = kern.exp
)

# Plot model
gp$plot1D()