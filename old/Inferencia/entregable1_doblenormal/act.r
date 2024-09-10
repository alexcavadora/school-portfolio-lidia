# Load the dataset
trafficData <- read.csv("Metro_Interstate_Traffic_Volume_pdf.csv", header=FALSE)
colnames(trafficData) <- c("DayOfWeek", "Hour", "TrafficVolume")

# Function to calculate separate Gaussian distributions
calculate_gaussians <- function(hours, trafficFlow, splitHour) {
  # Morning data
  morningHours <- hours[hours < splitHour]
  morningFlow <- trafficFlow[hours < splitHour]
  morningMean <- sum(morningHours * morningFlow) / sum(morningFlow)
  morningVar <- sum((morningHours - morningMean)^2 * morningFlow) / sum(morningFlow)
  
  # Evening data
  eveningHours <- hours[hours >= splitHour]
  eveningFlow <- trafficFlow[hours >= splitHour]
  eveningMean <- sum(eveningHours * eveningFlow) / sum(eveningFlow)
  eveningVar <- sum((eveningHours - eveningMean)^2 * eveningFlow) / sum(eveningFlow)
  
  # Generate points for Gaussian distribution
  x_points <- seq(min(hours), max(hours), length.out = 100)
  
  # Gaussian distributions
  gaussianMorning <- dnorm(x_points, morningMean, sqrt(morningVar))
  gaussianEvening <- dnorm(x_points, eveningMean, sqrt(eveningVar))
  
  # Scaling factors
  scalingFactorMorning <- max(morningFlow) / max(gaussianMorning)
  scalingFactorEvening <- max(eveningFlow) / max(gaussianEvening)
  
  # Scale the Gaussian models
  gaussianScaledMorning <- gaussianMorning * scalingFactorMorning
  gaussianScaledEvening <- gaussianEvening * scalingFactorEvening
  
  return(list(morning = list(x = x_points, y = gaussianScaledMorning), 
              evening = list(x = x_points, y = gaussianScaledEvening)))
}

# Loop through each day of the week
for (dayIndex in 1:7) {
  # Filter data for the day
  dayData <- subset(trafficData, DayOfWeek == dayIndex)
  dayData <- dayData[order(dayData$Hour),]
  
  # Estimate a reasonable split hour for morning and evening peaks
  splitHour <- 12 # This is a simplified assumption, you may need a more data-driven approach
  
  # Calculate Gaussian distributions for morning and evening
  gaussians <- calculate_gaussians(dayData$Hour, dayData$TrafficVolume, splitHour)
  
  # Plotting with a fixed y-axis range from 0 to 1000
  plot(dayData$Hour, dayData$TrafficVolume, type='h', col='grey', lwd=10, xlab="Hour", ylab="Traffic Volume",
       main=paste("Traffic Volume - Day", dayIndex), ylim=c(0, 1000))
  lines(gaussians$morning$x, gaussians$morning$y, col='red', type='l', lwd=2)
  lines(gaussians$evening$x, gaussians$evening$y, col='blue', type='l', lwd=2)
  
  # Displaying the split hour line
  abline(v=splitHour, col="darkgreen", lwd=2, lty=2)
  

}

# Note: The standard deviation lines are shown for the overall mean hour. If you need to show
# separate lines for the before and after segments, you would need to calculate those separately.
