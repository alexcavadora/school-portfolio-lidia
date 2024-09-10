datos<-read.csv("Metro_Interstate_Traffic_Volume_pdf.csv", header=FALSE)
head(datos)
lb_dias = c("Domingo", "Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado")
np <- 100
id_1 <- sample(1:7, 1)
id_s <- datos[,1]==id_1
hrs <- datos[id_s,2]
nhrs <- length(hrs)
fl <- datos[id_s,3]
md <- sum(hrs*fl)/sum(fl)
id_vl <- ceiling(md)
vl <- fl[id_vl]
vls <- round(vl*(id_vl-md))
vli <- vl-vls
di <- fl[hrs<id_vl]
di[id_vl] <- vli
dd <- fl[hrs>=id_vl]
dd <- c(vls, dd)
ndi = sum(di)
md1 <- sum(hrs[1:id_vl]*di)/ndi
vr1 <- sum((hrs[1:id_vl]**2)*di)/ndi-md1**2
mx1 <- max(di)
ndd <- sum(dd)
md2 <- sum(hrs[id_vl:nhrs]*dd)/ndd
vr2 <- sum((hrs[id_vl:nhrs]**2)*dd)/ndd-md2**2
mx2 <- max(dd)

vhrs <- (0:(23*np))/np
mdl1 <- mx1*exp((-(vhrs-md1)**2)/(2*vr1))
mdl2 <- mx2*exp((-(vhrs-md2)**2)/(2*vr2))
mdlt <- mdl1
id_lt <- vhrs>md
mdlt[id_lt] <- mdl2[id_lt]

xr1 <- mx1*exp((-(md-md1)**2)/(2*vr1))
xr2 <- mx2*exp((-(md-md2)**2)/(2*vr2))
md
xr1
xr2

a <- vr1-vr2
b <- 2*(vr2*md1-vr1*md2)
c <- vr1*md2**2-vr2*md1**2-2*vr1*vr2*log(mx2/mx1)
x1 <- (-b+sqrt(b**2-4*a*c))/(2*a)
x2 <- (-b-sqrt(b**2-4*a*c))/(2*a)
xr <- ifelse(x1>md1&x1<md2, x1, x2)
x1
x2
xr

mdlt2 <- mdl1
id_lt2 <- vhrs>xr
mdlt2[id_lt2] <- mdl2[id_lt2]

h_sel <- 1:23
fl_sel <- ifelse(h_sel<xr, mx1*exp((-(h_sel-md1)**2)/(2*vr1)), mx2*exp((-(h_sel-md2)**2)/(2*vr2)))
sum((fl_sel-fl[h_sel+1])**2)
100*sum(abs(fl_sel-fl[h_sel+1]))/max(fl)
100*sum(max(abs(fl_sel-fl[h_sel+1])))/max(fl)

hrs_mn <- min(hrs);
hrs_mx <- max(hrs);
fl_mn <- min(fl)
fl_mx <- max(fl)
plot(vhrs, mdl1, type="l", col='blue', xlim=c(hrs_mn, hrs_mx), ylim=c(fl_mn, fl_mx), xlab="Hora", ylab="Volumen de tráfico")
par(new=TRUE)
plot(vhrs, mdl2, type="l", col='red',  xlim=c(hrs_mn, hrs_mx), ylim=c(fl_mn, fl_mx), xlab="Hora" , ylab="Volumen de tráfico")

xb <- barplot(fl, names.arg=hrs, xlab="Hora", ylab="Volumen de tráfico")
par(new=TRUE)
xbi <- xb[1]
dxb <- xb[2]-xb[1]
lines(dxb*vhrs+xbi, mdlt, type="l", col='red', lwd=2)

xb <- barplot(fl, names.arg=hrs, xlab="Hora", ylab="Volumen de tráfico")
par(new=TRUE)
xbi <- xb[1]
dxb <- xb[2]-xb[1]
lines(dxb*vhrs+xbi, mdlt2, type="l", col='red', lwd=2)