import logging
from abc import ABC, abstractmethod

from astropy.coordinates import SkyCoord

logger = logging.getLogger(__name__)


class BaseTable(ABC):
    ID = None
    RA = None
    DEC = None

    def __init__(self):
        pass

    @abstractmethod
    def load_table(self):
        return

    def load_coordinates(self):
        self.coordinates = SkyCoord(
            self.table[self.RA], self.table[self.DEC], unit="deg"
        )
        return


class PhotTable(BaseTable):
    pass


class SpecTable(BaseTable):
    BASE_FOLDER = "tables/SPEC"
    ZSPEC = None


    def __init__(self):
        self.load_table()
        self.load_coordinates()


