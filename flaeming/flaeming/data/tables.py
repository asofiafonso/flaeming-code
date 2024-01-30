import logging
import os

import numpy as np
import pandas as pd
from astropy.table import Table, join
from flaeming import __data__
from flaeming.data.base import PhotTable
from flaeming.data.static_variables import (
    LAE_CLASS,
    NONLAE_TABLE_BASENAME,
    REDSHIFT_COL,
)
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class SC4KTable(PhotTable):
    ID = "ID_SC4K"
    RA = "RA"
    DEC = "DEC"
    ID_COSMOS = "ID"

    def __init__(self, features: list[str] = None):
        super(SC4KTable, self).__init__()
        self.load_table(features=features)
        self.load_coordinates()

    def load_table(self, features: list[str] = None) -> pd.DataFrame:
        """Reads SC4K table from file and performs the necessary
        data transofrmations for Machine Learning use
        """
        # Redshift mapping from NB/IB band names
        redshifts = {
            "NB392": ((2.24 - 2.20) / 2) + 2.20,
            "IA427": ((2.59 - 2.42) / 2) + 2.42,
            "IA464": ((2.90 - 2.72) / 2) + 2.72,
            "IA484": ((3.98 - 2.89) / 2) + 2.89,
            "NB501": ((3.16 - 3.08) / 2) + 3.08,
            "IA505": ((3.26 - 3.07) / 2) + 3.07,
            "IA527": ((3.43 - 3.23) / 2) + 3.23,
            "IA574": ((3.85 - 3.63) / 2) + 3.63,
            "IA624": ((4.25 - 4.00) / 2) + 4.00,
            "IA679": ((4.72 - 4.44) / 2) + 4.44,
            "IA709": ((4.95 - 4.69) / 2) + 4.69,
            "NB711": ((4.89 - 4.83) / 2) + 4.83,
            "IA738": ((5.19 - 4.92) / 2) + 4.92,
            "IA767": ((5.47 - 5.17) / 2) + 5.17,
            "NB816": ((5.75 - 5.65) / 2) + 5.65,
            "IA827": ((5.92 - 5.64) / 2) + 5.64,
        }

        table_cosmos = Table.read(
            f"{__data__}/tables/SC4K/SC4K_COSMOS2020.fits", format="fits"
        )
        table_sc4k = Table.read(
            f"{__data__}/tables/SC4K/SC4K_final_Santos2020_with_MAGPHYS_properties_v2.fits",
            format="fits",
        )
        table = join(table_sc4k, table_cosmos, keys=["ID_SC4K", "RA", "DEC"])
        sc4k = table.to_pandas()

        # Create column with Lya detection band
        sc4k[self.ID] = sc4k[self.ID].str.decode("utf-8").str.strip()

        sc4k.loc[:, "BAND"] = sc4k[self.ID].str.split("-", expand=True)[1]
        # Add classification column
        sc4k.loc[:, LAE_CLASS] = 1
        # Add redshift from detection band
        sc4k.loc[:, REDSHIFT_COL] = sc4k["BAND"].map(redshifts)

        if features is not None:
            self.table = sc4k.loc[:, features + [self.ID, self.RA, self.DEC, LAE_CLASS]]
        else:
            self.table = sc4k
        return sc4k


class COSMOSTable(PhotTable):
    ID = "ID"
    RA = "RA"
    DEC = "DEC"

    def __init__(self, sc4k_IDs=None):
        super(COSMOSTable, self).__init__()
        self.load_table(sc4k_IDs)
        self.load_coordinates()

    def load_table(self, sc4k_IDs: list = None) -> pd.DataFrame:
        """Reads COSMOS2020 table and removes SC4K LAEs found in `sc4k_table`-"""
        cosmos2020 = Table.read(
            f"{__data__}/tables/COSMOS/COSMOS2020_zBEST2to6.fits", format="fits"
        ).to_pandas()

        if sc4k_IDs is not None:
            cosmos2020_noLAES = cosmos2020.loc[~cosmos2020[self.ID].isin(sc4k_IDs)]
        else:
            cosmos2020_noLAES = cosmos2020

        # Add classification column
        cosmos2020_noLAES.loc[:, LAE_CLASS] = 0

        # rename redshift column
        cosmos2020_noLAES.rename(
            columns={
                "lp_zBEST": REDSHIFT_COL,
                "ALPHA_J2000": self.RA,
                "DELTA_J2000": self.DEC,
            },
            inplace=True,
        )

        self.table = cosmos2020_noLAES
        return cosmos2020_noLAES


class COSMOSSamplingTable(PhotTable):
    ID = "ID"
    RA = "RA"
    DEC = "DEC"

    def __init__(self, sample_number: int = None, features: list[str] = None):
        super().__init__()
        self.load_table(sample_number, features=features)
        self.load_coordinates()

    def load_table(
        self, sample_number: int, features: list[str] = None
    ) -> pd.DataFrame:
        """Reads COSMOS2020 table and removes SC4K LAEs found in `sc4k_table`-"""
        if isinstance(sample_number, int):
            suffix = f"{sample_number}"
        else:
            suffix = "_ALL"

        file_name = f"{__data__}/samples/{NONLAE_TABLE_BASENAME}{suffix}.csv"
        cosmos_samples = pd.read_csv(file_name, dtype={self.ID: "str"})

        if features is not None and suffix != "_ALL":
            cosmos_all = pd.read_csv(
                f"{__data__}/samples/{NONLAE_TABLE_BASENAME}_ALL.csv",
                dtype={self.ID: "str"},
            )
            cosmos_samples = pd.merge(
                cosmos_samples,
                cosmos_all.loc[:, [self.ID] + features],
                how="left",
                on=self.ID,
            )

        # Add classification column
        cosmos_samples.loc[:, LAE_CLASS] = 0

        self.table = cosmos_samples
        return cosmos_samples


def p(z: float, sc4k_table: pd.DataFrame):
    """Computes photometric probability distribution from `sc4k_table`
    Deprecated (not used now)
    """
    h, bins = np.histogram(sc4k_table["zBAND"], bins=np.linspace(2, 6, 17))
    binidx = np.digitize(z, bins)
    # if binidx == 0 or binidx == len(bins):
    #   return 0
    weights = h[binidx - 1] / np.sum(h)
    return weights / np.sum(weights)


def bin_matching(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    z_col: str = "redshift",
    z_bins: np.array = np.linspace(2, 6, 17),
    id_col: str = "ID",
    ids_exclude: list = None,
) -> pd.DataFrame:
    """Returns a subsample of `table2` that matches the number of objects
    in `table1` using `z_col` sampled in bins of `z_bins`.

    It is used to create subsample with matching redshift distributions by default.
    """
    h, bins = np.histogram(table1[z_col], bins=z_bins)

    redshift_col = table2[z_col]
    sub_dfs = []
    for i in range(len(bins) - 1):
        selection = (redshift_col >= bins[i]) & (redshift_col < bins[i + 1])
        # exclude existing IDs from selection
        if ids_exclude is not None:
            logger.debug(f"Excluding existing IDs")
            selection = selection & (~table2[id_col].isin(ids_exclude))
        df_to_select = table2.loc[selection]

        sub_dfs.append(df_to_select.sample(n=h[i]))

    return pd.concat(sub_dfs)


def generate_quantile_bins(
    variable: pd.Series, n_bins: int = 5, outlier_frac: float = 0.02
) -> NDArray:
    return np.concatenate(
        [
            [variable.min() - 1e-3],
            np.linspace(
                variable.quantile(outlier_frac / 2),
                variable.quantile(1 - outlier_frac / 2),
                n_bins - 1,
            ),
            [variable.max()],
        ]
    )


def pd_bin_matching(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    match_cols: list,
    match_bins: dict,
    id_col: str = "ID",
    ids_to_exclude: list = None,
    n_bins_default: int = 5,
) -> pd.DataFrame:
    data = table1.loc[:, match_cols]
    bins = []
    for col in match_cols:
        if col in match_bins.keys():
            bins.append(match_bins[col])
        else:
            logger.warning(
                f"{col} has no bins defined, creating {n_bins_default} bins."
            )
            bins.append(
                np.linspace(
                    data[col].quantile(0.001),
                    data[col].quantile(0.999),
                    n_bins_default + 1,
                )
            )
    bin_cuts = [pd.cut(data[col], bins=bin) for col, bin in zip(match_cols, bins)]
    bin_counts = data[match_cols[0]].groupby(bin_cuts).count()

    table_for_selection = table2.loc[~table2[id_col].isin(ids_to_exclude)].copy()

    logger.info(
        f"Selecting a total of {bin_counts.sum()} samples from {len(table_for_selection)} sources."
    )
    dfs = []
    ids_selected = []
    for bins, count in zip(bin_counts.index, bin_counts):
        if count == 0:
            continue

        selection = np.product(
            [
                table_for_selection[col].between(bin.left, bin.right).values
                for col, bin in zip(match_cols, bins)
            ],
            axis=0,
        ).astype(bool)
        selection_IDs = ~table_for_selection[id_col].isin(ids_selected)

        n_to_select = len(table_for_selection.loc[selection & selection_IDs])
        k = 1
        while n_to_select < count:
            logger.warning(f"{n_to_select} is smaller than {count}.")
            logger.debug(f"Increasing k to {k}.")
            pad = [0] * (len(bins) - 1) + [k]
            selection = np.product(
                [
                    table_for_selection[col]
                    .between(bin.left - p * bin.length, bin.right)
                    .values
                    for col, bin, p in zip(match_cols, bins, pad)
                ],
                axis=0,
            ).astype(bool)
            n_to_select = len(table_for_selection.loc[selection & selection_IDs])
            k += 1

        sub_sample = table_for_selection.loc[selection & selection_IDs].sample(
            n=min(count, n_to_select)
        )
        ids_selected += sub_sample[id_col].tolist()
        dfs.append(sub_sample)

    return pd.concat(dfs)
    # return selection_vars


def nd_bin_matching(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    match_cols: list,
    match_bins: dict,
    id_col: str = "ID",
    ids_to_exclude: list = None,
    n_bins_default: int = 5,
) -> pd.DataFrame:
    data = table1.loc[:, match_cols]
    bins = []
    logger.debug(f"Binning using columns: {match_cols}")
    for col in match_cols:
        if col in match_bins.keys():
            bins.append(match_bins[col])
        else:
            logger.warning(
                f"{col} has no bins defined, creating {n_bins_default} bins."
            )
            bins.append(
                np.linspace(
                    data[col].min(),
                    data[col].max(),
                    n_bins_default + 1,
                )
            )

    H, bin_edges = np.histogramdd(data.values, bins)
    logger.info(f"Selecting a total of {np.sum(H)} samples.")

    n_vars = len(bin_edges)
    if n_vars > 2:
        raise NotImplementedError

    sub_dfs = []
    if ids_to_exclude is not None:
        logger.debug(f"Excluding existing IDs")
        base_selection = ~table2[id_col].isin(ids_to_exclude)

    for i in range(len(bin_edges[0]) - 1):
        selection_i = (
            base_selection
            & (table2[match_cols[0]] > bin_edges[0][i])
            & (table2[match_cols[0]] <= bin_edges[0][i + 1])
        )
        k = 1
        for j in range(len(bin_edges[1]) - 1):
            selection_ij = (
                selection_i
                & (table2[match_cols[1]] > bin_edges[1][j])
                & (table2[match_cols[1]] <= bin_edges[1][j + 1])
            )

            df_to_select = table2.loc[selection_ij]
            n_to_select = int(H[i, j])

            logger.debug(
                f"Selected a total of {len(df_to_select)} potential targets to sample {n_to_select} from."
            )
            while len(df_to_select) < n_to_select:
                logger.warning(f"{len(df_to_select)} is smaller than {n_to_select}.")
                logger.debug(f"Increasing k to {k}.")
                selection_ij = (
                    selection_i
                    & (table2[match_cols[1]] > bin_edges[1][j - k])
                    & (table2[match_cols[1]] <= bin_edges[1][j + 1 - k])
                )
                df_to_select = table2.loc[selection_ij]
                k += 1

            sub_dfs.append(df_to_select.sample(n=n_to_select))

    return pd.concat(sub_dfs)
