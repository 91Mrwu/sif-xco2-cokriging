
library(sp)
library(spacetime)
library(rgdal)
library(gstat)

data(meuse)
data(meuse.grid)

coordinates(meuse) = ~x+y
variogram(log(zinc)~1, meuse)

proj4string(meuse) = CRS("+init=epsg:28992")
meuse.ll = spTransform(meuse, CRS("+proj=longlat +datum=WGS84"))
# variogram of unprojected data, using great-circle distances, returning km as units
plot(variogram(log(zinc) ~ 1, meuse.ll, covariogram=TRUE))


# subsetting the meuse dataset to have data at different locations:
Cd.data <- meuse[sample.int(nrow(meuse),0.7*nrow(meuse)), c("x","y","cadmium")]
Zn.data <- meuse[sample.int(nrow(meuse),0.7*nrow(meuse)), c("x","y","zinc")]
meuse.subset <- merge(Cd.data, Zn.data, by = c("x","y"), all = TRUE)

# Plot cross covariogram
g = gstat(NULL, id = "Cd", formula = cadmium~1, data = meuse, locations = ~x+y)
g <- gstat(g, id = "Zn", formula = zinc~1, data = meuse, locations = ~x+y)
Cd.cov <- variogram(g, cutoff = 1500, width = 50, covariogram = TRUE, cloud = TRUE)
plot(Cd.cov)

