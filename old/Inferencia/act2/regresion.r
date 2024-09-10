data <- read.csv("Datos_2D.csv", header = FALSE)
head(data)

# Ejemplo de un plano
M <- matrix(0, nrow = 3, ncol = 3)
b <- matrix(0, nrow = 3, ncol = 1)
n <- nrow(data)
x1 <- data[,1]
x2 <- data[,2]
y <- data[,3]

M[1,1] <- n
M[1,2] <- sum(x1)
M[1,3] <- sum(x2)
b[1,1] <- sum(y)

M[2,1] <- sum(x1)
M[2,2] <- sum(x1^2)
M[2,3] <- sum(x1*x2)
b[2,1] <- sum(y*x1)

M[3,1] <- sum(x2)
M[3,2] <- sum(x1*x2)
M[3,3] <- sum(x2^2)
b[3,1] <- sum(y*x2)

a <- solve(M, b)
y1 <- a[1,1] + a[2,1]*x1 + a[3,1]*x2

SSE1 <- sqrt(sum((y-y1)**2))
my <- mean(y)
vy <- sum((y-my)**2)
R21 <- 1-SSE1/vy
R21
id <- order(x2)

plot(x2, y, xlab = "x2", ylab = "y", col = "blue")
lines(x2[id], y1[id], col = "red", type = "p")

# Ejemplo de un plano con corelación
M <- matrix(0, nrow = 4, ncol = 4)
b <- matrix(0, nrow = 4, ncol = 1)
n <- nrow(data)
x1 <- data[,1]
x2 <- data[,2]
y <- data[,3]
M[1,1] <- n
M[1,2] <- sum(x1)
M[1,3] <- sum(x2)
M[1,4] <- sum(x1*x2)
b[1,1] <- sum(y)
M[2,1] <- sum(x1)
M[2,2] <- sum(x1^2)
M[2,3] <- sum(x1*x2)
M[2,4] <- sum(x1^2*x2)
b[2,1] <- sum(y*x1)
M[3,1] <- sum(x2)
M[3,2] <- sum(x1*x2)
M[3,3] <- sum(x2^2)
M[3,4] <- sum(x1*x2^2)
b[3,1] <- sum(y*x2)
M[4,1] <- sum(x1*x2)
M[4,2] <- sum(x1^2*x2)
M[4,3] <- sum(x2^2*x1)
M[4,4] <- sum((x1*x2)^2)
b[4,1] <- sum(y*x1*x2)
a <- solve(M, b)
y2 <- a[1,1] + a[2,1]*x1 + a[3,1]*x2 + a[4,1]*x1*x2
SSE2 <- sqrt(sum((y-y2)**2))
R22 <- 1-SSE2/vy
R22
id <- order(x2)

plot(x2, y, xlab = "x2", ylab = "y", col = "blue")
lines(x2[id], y2[id], col = "red", type = "p")