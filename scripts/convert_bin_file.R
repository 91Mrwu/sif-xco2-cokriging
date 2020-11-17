
# Read in TransCom basis function data from binary array and save as NetCDF.

# File description:
#' The file 'smoothmap.fix.2.dat' contains a single real, binary 
#' array dimensioned 360 x 180. The array contains the numbers 1 
#' through 22, denoting each of the 22 basis functions in the 
#' TransCom 3 experiment. This file was written on an SGI Origin
#' 2000 hosting UNIX.


#' Read in a binary array, likely written with IDL
#' 
#' @param x path to file (auto-expanded & tested for existence)
#' @param n number of `float` elements to read in
#' @param endian endian-ness (default `big`)
#' @return numeric vector of length `n`
#' 
#' Credit: https://stackoverflow.com/questions/47947793/reading-a-binary-map-file-in-r
read_binary_float <- function(x, n, endian="big") {
  
  x <- normalizePath(path.expand(x))
  
  x <- readBin(con = x, what = "raw", n = file.size(x))
  
  first4 <- x[1:4] # extract front bits
  last4 <- x[(length(x)-3):length(x)] # extract back bits
  
  # convert both to long ints      
  
  f4c <- rawConnection(first4)
  on.exit(close(f4c), add=TRUE)
  f4 <- readBin(con = f4c, what = "integer", n = 1, size = 4L, endian=endian)
  
  l4c <- rawConnection(last4)      
  on.exit(close(l4c), add=TRUE)      
  l4 <- readBin(con = l4c, what = "integer", n = 1, size = 4L, endian=endian)
  
  # validation
  
  stopifnot(f4 == l4) # check front/back are equal
  stopifnot(f4 == n*4) # check if `n` matches expected record count
  
  # strip off front and back bits
  
  x <- x[-(1:4)]
  x <- x[-((length(x)-3):length(x))]
  
  # slurp it all in
  
  rc <- rawConnection(x)      
  on.exit(close(rc), add=TRUE)
  
  readBin(con = rc, what = "numeric", n = n, size = 4L, endian=endian)
  
}

# Read the data
library(magrittr)
path <- "./tmp/basis_function_map_all/smoothmap.fix.2.bin"
basis <- read_binary_float(path, 360*180) %>% 
  matrix(nrow = 360, ncol = 180)

# Check in plot
library(RColorBrewer)
coul <- brewer.pal(11, "RdBu") 
coul <- colorRampPalette(coul)(22)
image(basis, col=rev(coul))

# Write to netcdf
library(ncdf4)
fname <- "./tmp/transcom3_basis_map.nc"

lon <- as.double(seq(-179.5, 179.5))
lat <- as.double(seq(-89.5, 89.5))
londim <- ncdim_def("lon","Longitude", lon) 
latdim <- ncdim_def("lat","Latitude", lat)
basis_def <- ncvar_def(
  "region",
  "",
  list(londim, latdim),
  longname="TransCom 3 basis function (region) map. The array contains the numbers 1 through 22, denoting each of the 22 basis functions in the TransCom 3 experiment."
)

ncout <- nc_create(fname, list(basis_def), force_v4=TRUE)
ncvar_put(ncout, basis_def, basis)
ncatt_put(ncout,"lon","axis","X")
ncatt_put(ncout,"lat","axis","Y")

ncout
