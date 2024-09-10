  # Lectura de datos
datos <- read.csv("datos.csv", header = FALSE)
# Transponer los datos
n <- ncol(datos)
x<-matrix(0, nrow = n)
x[,1]<-datos[1,]
y<-matrix(0, nrow = n)
y[,1]<-datos[2,]
x<-as.double(x)
y<-as.double(y)
# Ajuste de la regresión lineal y cuadratica
Sx <- sum(x)
Sx2 <- sum(x^2)
Sx3 <- sum(x^3)
Sx4 <- sum(x^4)
Sy <- sum(y)
Sxy <- sum(x*y)
Sx2y <- sum(x^2*y)
M1 <- matrix(0, nrow = 3, ncol = 3)
v1 <- matrix(0, nrow = 3, ncol = 1)
M1[1,1] <- n
M1[1,2] <- Sx
M1[1,3] <- Sx2
M1[2,1] <- Sx
M1[2,2] <- Sx2
M1[2,3] <- Sx3
M1[3,1] <- Sx2
M1[3,2] <- Sx3
M1[3,3] <- Sx4
v1[1,1] <- Sy
v1[2,1] <- Sxy
v1[3,1] <- Sx2y
m <- (Sxy - Sx*Sy/n)/(Sx2 - Sx^2/n)
b <- (Sy/n) - (m*Sx/n)
M2 <- solve(M1,v1)
# Simulación de modelos
y2 <- m*x + b
y3 <- M2[3,1]*x^2 + M2[2,1]*x + M2[1,1]
# Calculo de metricas
SSE1 <- sum((y-y2)^2)
MSE1 <- sqrt(SSE1)/n
vy <- sum((y-Sy/n)^2)
R21 <- 1-(SSE1/vy)
print(SSE1)
print(MSE1)
print(R21)
SSE2 <- sum((y-y3)^2)
MSE2 <- sqrt(SSE2)/n
R22 <- 1-(SSE2/vy)
print(SSE2)
print(MSE2)
print(R22)
# Graficar los datos y los modelos
plot(x, y, type = "p", col = "red", pch=19, cex = 1, xlab = "x", ylab = "f(x)", main = "Regresión Lineal", xlim=c(min(x), max(x)), ylim=c(min(y2), max(y)))
lines(x, y2, type='l', col="blue", lwd=2)
plot(x, y3, type = "p", col = "red", pch=19, cex = 1, xlab = "x", ylab = "f(x)", main = "Regresión Cuadratica")
lines(x, y2, type='l', col="blue")
# Detección de datos atípicos
# Detección de outliers
M3 <- matrix(0, nrow = 3, ncol = n)
SSE3 <- matrix(0, nrow = 1, ncol = n)
MSE3 <- matrix(0, nrow = 1, ncol = n)
R23 <- matrix(0, nrow = 1, ncol = n)
vm = min(x);
for(i1 in 1:n)
{
  id <- x>=vm
  id[i1] <- FALSE
  xs <- x[id]
  ys <- y[id]
  Sx <- sum(xs)
  Sx2 <- sum(xs^2)
  Sx3 <- sum(xs^3)
  Sx4 <- sum(xs^4)
  Sy <- sum(ys)
  Sxy <- sum(xs*ys)
  Sx2y <- sum(xs^2*ys)
  M1[1,1] <- n-1
  M1[1,2] <- Sx
  M1[1,3] <- Sx2
  M1[2,1] <- Sx
  M1[2,2] <- Sx2
  M1[2,3] <- Sx3
  M1[3,1] <- Sx2
  M1[3,2] <- Sx3
  M1[3,3] <- Sx4
  v1[1,1] <- Sy
  v1[2,1] <- Sxy
  v1[3,1] <- Sx2y
  M3[,i1] <- solve(M1,v1)
  y3 <- M3[3,i1]*xs^2 + M3[2,i1]*xs + M3[1,i1]
  SSE3[1,i1] <- sum((ys-y3)^2)
  MSE3[1,i1] <- sqrt(SSE3[1,i1])/(n-1)
  vy <- sum((ys-Sy/(n-1))^2)
  R23[1,i1] <- 1-(SSE3[1,i1]/vy)
}
imax <- which.max(R23)
print(SSE3[1,imax])
print(MSE3[1,imax])
print(R23[1,imax])
plot(x, R23, type = "p", col = "blue", pch=19, cex = 1)
points(x[imax], R23[1,imax], col="red", pch=19, cex = 1)