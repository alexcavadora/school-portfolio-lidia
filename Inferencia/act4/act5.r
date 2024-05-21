# Predominio = P / (P + N)
# Exactitud = (TP + TN) / (P + N)
# Precisión = TP / PP
# Sensibilidad = TP / P
# Espeficicidad = TN / N

datos <- read.csv("diabetes.csv")

#Preprocessing
missingBMI <- datos$BMI == 0
datos$BMI[missingBMI] <- NA

BMI <- datos$BMI[!missingBMI]
Outcome = datos$Outcome[!missingBMI]

#maxBMI <- max(BMI)
#minBMI <- min(BMI)

#minBMI_list <- which(BMI == minBMI)
#maxBMI_list <- which(BMI == maxBMI)


#(datos$Outcome[minBMI_list])
#(datos$Outcome[maxBMI_list])

Positive <- length(which(Outcome == 1))
Negative <- length(which(Outcome == 0))
size <- length(Outcome)

(meanPositive <- mean(BMI[Positive]))
(stdPositive <- sd(BMI[Positive]))

(meanNegative <- mean(BMI[Negative]))
(stdNegative <- sd(BMI[Negative]))

SPositive <- (BMI[Positive] - meanPositive) / stdPositive
SNegative <- (BMI[Negative] - meanNegative) / stdNegative

SH <- (BMI-meanPositive) / stdPositive
(positiveSH <- sum(SH<1.96&SH>-1.96))
