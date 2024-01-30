import os


# STANDARD_COLUMN_NAME
REDSHIFT_COL = "redshift"
LAE_CLASS = "LAE"
ID_COL = "ID"
PREDICT_COL = "COMBINED_PREDICTION"


with open(f"{os.path.dirname(__file__)}/COSMOS2020.cols", "r") as f:
    COSMOS2020_COLUMNS = f.read().splitlines()

with open(f"{os.path.dirname(__file__)}/COSMOS2020.filters", "r") as f:
    COSMOS2020_FILTERS = f.read().splitlines()


# STAMPS INFO
STAMP_SURVEY = "survey"
STAMP_FILTER = "filter"

# NONLAE VARS
NONLAE_TABLE_BASENAME = "COSMOS_NONLAE"


# LAE_INFO_COLS
LAE_LUM_COL = "Lya_Lum"
LAE_EW_COL = "EW_0"
LAE_FLUX_COL = "Flux"
LAE_ERR_UP_SUFFIX = "_err_up" 
LAE_ERR_DOWN_SUFFIX = "_err_down" 
