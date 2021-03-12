
library(tidyverse)
library(sp)
library(spacetime)
library(gstat)

# Read and format data so location advances before time
data_path <- "../data/exp_pro/OCO2_5deg_monthly_conus.csv"
df <- read_csv(data_path) # %>%
  # fill(xco2_res, sif_res) # NOTE: dangerous, but need to check if full data will fix the issue 
                          # Fixes missing values issue, but still "no method for gridded with STFDF / STIDF"
df$loc_id <- df %>% group_by(lat, lon) %>% group_indices()
df <- arrange(df, time, loc_id) 


# Spatiotemporal attempt --------------------------------------------------

locs <- unique(df[, c("lon", "lat")], margin=1)
locs$loc_id <- locs %>% group_by(lat, lon) %>% group_indices()

#check
all(unique(df$loc_id) == locs$loc_id)

# Construct STFDF / STSDF / STIDF classes
locations <- SpatialPoints(coords = locs[, c("lon", "lat")])
times <- as.Date(unique(df$time))

ST_obj <- STFDF(sp = locations,
                time = times,
                data = df %>% select(xco2_res, sif_res))

ST_test <- stConstruct(
  x = as.data.frame(df %>% drop_na() %>% select(-loc_id)), 
  space = c("lat", "lon"), 
  time = c("time"))

# proj4string(STobj) <- CRS("+proj=longlat +ellps=WGS84")

# Empirical covariograms
g <- gstat(NULL, id = "xco2_res", formula = xco2_res~1, data = ST_obj)
g <- gstat(g, id = "sif_res", formula = sif_res~1, data = ST_obj)

cv_st <- variogramST(
  g, 
  data = ST_obj,
  # pseudo = TRUE,
  cutoff = 500,
  width = 10,
  # tlags = 2,
  covariogram = TRUE,
  # cross = TRUE, # this will break
  # na.omit = TRUE,
  assumeRegular = TRUE,
  # cores = 16
)

cv <- variogram(
  g,
  covariogram = TRUE,
  # width = 10,
  # cutoff = 500,
  cross = TRUE,
  # projected = FALSE
)

plot(cv_st)
# NOTE: this is identical to the cov for SIF



## Look at individual covariograms

cov_xco2 <- variogramST(
  xco2_res~1, 
  data = ST_obj, 
  cutoff = 500, 
  width = 10, 
  covariogram = TRUE,
  assumeRegular = TRUE
  )
cov_sif <- variogramST(
  sif_res~1, 
  data = ST_obj, 
  cutoff = 500, 
  width = 10, 
  covariogram = TRUE,
  assumeRegular = TRUE
  )

plot(cov_xco2)
plot(cov_sif)



# Spatial Only ------------------------------------------------------------

df_xco2 <- df %>% 
  filter(time == "2019-08-01") %>%
  select(-sif_res, -time) %>%
  drop_na()

df_sif <- df %>% 
  filter(time == "2019-07-01") %>%
  select(-xco2_res, -time) %>% 
  drop_na()

coordinates(df_xco2) <- c("lon", "lat")
coordinates(df_sif) <- c("lon", "lat")

g <- gstat(NULL, id = "xco2_res", formula = xco2_res~1, data = df_xco2)
g <- gstat(g, id = "sif_res", formula = sif_res~1, data = df_sif)

cv <- variogram(
  g,
  covariogram = TRUE,
  width = 10,
  cutoff = 500,
  cross = TRUE,
  # projected = FALSE
)
plot(cv, main = "5-degree monthly average, locally detrended then standardized, cont. US subset for XCO2 (2019-08) and SIF (2019-07)", cex.main = 0.9)
