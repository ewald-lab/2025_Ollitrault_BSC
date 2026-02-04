require(dplyr)
require(arrow)
require(data.table)


######## 1. Parse arguments and define parameters
args <- commandArgs(trailingOnly = TRUE)

bmd_path <- args[1]
# meta_pod_path <- args[2] # remove the information if it is below or above the meta POD
pod_path <- args[2]
# meta_nm <- args[4]

######## 2. Select POD as minimum BMD across gmd and all categories
bmd <- read_parquet(bmd_path) %>% as.data.frame()

# This is what the SD of residuals must be less than
bmd$res.thresh <- 3 * bmd$SDctrl
bmd <- as.data.table(bmd)

bmd <- bmd[all.pass == TRUE]
bmd <- bmd[SDres < res.thresh] # Filter by the SD of the residual

min_bmd <- bmd[,
               .SD[bmd == min(bmd)],
               by = Metadata_compound,
               .SDcols = c("gene.id", "mod.name", "bmd", "all.pass", "adv.incr",
                           "b", "c", "d", "e", "f",
                           "bmdl", "bmdu", "bmr", "lof.p")]

min_bmd <- as.data.frame(min_bmd)

if (FALSE) {
    ######## 3. Read in Meta data PODs
    meta <- read_parquet(meta_pod_path) %>% as.data.frame()
    meta$bmd[meta$all.pass != TRUE] <- 9999
    meta <- meta[, c("Metadata_compound", "bmd")]
    colnames(meta)[2] <- paste0(meta_nm, "_POD")

    min_bmd <- merge(min_bmd, meta, by = "Metadata_compound")
    min_bmd[,paste0(meta_nm, "_PAC_below_meta_POD")] <- min_bmd$bmd < min_bmd$meta_POD
}

######## 4. Write out results
write_parquet(min_bmd, pod_path)
