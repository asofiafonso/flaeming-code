from astropy.table import Table, join, vstack

from flaeming import __data__
from flaeming.data.base import SpecTable


class VUDSTable(SpecTable):
    ID = "ident"
    RA = "alpha"
    DEC = "delta"
    ZSPEC = "z"
    name = "VUDS"

    def load_table(self):
        main_table_name = "sample_cosmos_allflags_wredshift.txt"
        main_table = Table.read(
            f"{__data__}/{self.BASE_FOLDER}/{main_table_name}", format="ascii"
        )
        ew_table_name = "EW_Lya_VUDS_v4.txt"
        added_info = Table.read(f"{__data__}/Tables/{ew_table_name}", format="ascii")
        added_info["ID"].name = VUDSTable.ID
        self.table = join(main_table, added_info, keys=VUDSTable.ID, join_type="left")
        return


class DEIMOSTable(SpecTable):
    ID = "ID"
    RA = "ra"
    DEC = "dec"
    ZSPEC = "zspec"
    ZQUAL = "Qf"
    name = "DEIMOS"

    def load_table(self):
        table_name = "deimos_redshift_linksIRSA.tbl"
        self.table = Table.read(
            f"{__data__}/{self.BASE_FOLDER}/{table_name}", format="ascii"
        )

        return


class zCOSMOSTable(SpecTable):
    ID = "id"
    RA = "ra"
    DEC = "dec"
    ZSPEC = "zpec"
    ZQUAL = "cc"
    name = "zCOSMOS"

    def load_table(self):
        table_name = "zcosbrightspec20k_dr3.csv"
        self.table = Table.read(
            f"{__data__}/{self.BASE_FOLDER}/{table_name}", format="ascii.csv"
        )
        self.table = self.table[
            (self.table[self.RA] != -99.9) & (self.table[self.DEC] != -99.9)
        ]
        return


class C3R2KMOSTable(SpecTable):
    ID = "objID"
    RA = "RAdeg"
    DEC = "DEdeg"
    ZSPEC = "zsp"
    ZQUAL = "q_zsp"
    name = "C3R2-KMOS"

    def load_table(self):
        table_name = "C3R2_KMOS_Guglielmo2020.fits"
        self.table = Table.read(
            f"{__data__}/{self.BASE_FOLDER}/{table_name}", format="fits"
        )
        self.table = self.table[
            (self.table[self.RA] != -99.9) & (self.table[self.DEC] != -99.9)
        ]
        return


class C3R2DEIMOSTable(SpecTable):
    ID = "ID"
    RA = "RAdeg"
    DEC = "DEdeg"
    ZSPEC = "zspec"
    ZQUAL = "Qual"
    name = "C3R2-DEIMOS"

    def load_table(self):
        table1_name = "DEIMOS_C3R2_DR3_Stanford2021.fits"
        table2_name = "DEIMOS_C3R2_Masters2019.fits"

        table1 = Table.read(
            f"{__data__}/{self.BASE_FOLDER}/{table1_name}", format="fits"
        )
        table2 = Table.read(
            f"{__data__}/{self.BASE_FOLDER}/{table2_name}", format="fits"
        )
        self.table = vstack([table1, table2])

        self.table[self.RA] = (
            self.table["RAh"] * 15 + self.table["RAm"] / 60 + self.table["RAs"] / 3600
        )
        DE_sign = (self.table["DE-"] == "+").astype(int) * 2 - 1
        self.table[self.DEC] = (
            DE_sign * self.table["DEd"]
            + self.table["DEm"] / 60
            + self.table["DEs"] / 3600
        )

        return


class FMOSTable(SpecTable):
    ID = "FMOS_ID"
    RA = "RA"
    DEC = "DEC"
    ZSPEC = "ZBEST"
    ZQUAL = "ZFLAG"
    name = "FMOS-COSMOS"

    def load_table(self):
        table_name = "fmos-cosmos_catalog_2019.fits"
        self.table = Table.read(
            f"{__data__}/{self.BASE_FOLDER}/{table_name}", format="fits"
        )
        self.table = self.table[
            (self.table[self.RA] != -99.9) & (self.table[self.DEC] != -99.9)
        ]
        return


class MUSETable(SpecTable):
    ID = "ID"
    RA = "RAdeg"
    DEC = "DEdeg"
    ZSPEC = "z"
    ZQUAL = "q_z"
    name = "MUSE"

    def load_table(self):
        table_name = "MUSE_Rosani2020.fits"
        self.table = Table.read(
            f"{__data__}/{self.BASE_FOLDER}/{table_name}", format="fits"
        )
        self.table = self.table[
            (self.table[self.RA] != -99.9) & (self.table[self.DEC] != -99.9)
        ]
        return


class LEGACTable(SpecTable):
    ID = "ID"
    RA = "RA"
    DEC = "DEC"
    ZSPEC = "Z_SPEC"
    ZQUAL = "FLAG_SPEC"
    name = "LEGA-C"

    def load_table(self):
        table_name = "LEGAC_VanderWel2021.fits"
        self.table = Table.read(
            f"{__data__}/{self.BASE_FOLDER}/{table_name}", format="fits"
        )
        self.table = self.table[
            (self.table[self.RA] != -99.9) & (self.table[self.DEC] != -99.9)
        ]
        return


class OIITable(SpecTable):
    ID = "ID"
    RA = "RAdeg"
    DEC = "DEdeg"
    ZSPEC = "z"
    name = "OII"

    def load_table(self):
        table_name = "OIIsurvey_Comparat2015.fits"
        self.table = Table.read(
            f"{__data__}/{self.BASE_FOLDER}/{table_name}", format="fits"
        )
        self.table = self.table[
            (self.table[self.RA] != -99.9) & (self.table[self.DEC] != -99.9)
        ]
        self.table[self.ID] = list(range(len(self.table)))
        return


class HETDEXTable(SpecTable):
    ID = "source_id"
    RA = "RA"
    DEC = "DEC"
    ZSPEC = "z_hetdex"
    SOURCE_TYPE = "source_type"
    name = "HETDEX"

    def load_table(self):
        # table_name = "hetdex_sc1_v3.2.ecsv"
        # self.table = Table.read(
        #     f"{__data__}/tables/HETDEX/{table_name}", format="ascii"
        # )
        table_name = "hetdex_sc1_detinfo_v3.2.fits"
        self.table = Table.read(f"{__data__}/tables/HETDEX/{table_name}", format="fits")
        self.table = self.table[
            (self.table[self.RA] != -99.9) & (self.table[self.DEC] != -99.9)
        ]
        return
