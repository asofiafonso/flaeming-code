from flaeming import __data__
from sedpy.observate import Filter as sedpyFilter


class Filter(sedpyFilter):
    """Simple class to override sedpy.obervate.Filter
    to have the default filter directory that of the
    FLAEMING project.
    """

    def __init__(self, name: str, **kwargs):
        """Passes the name of the filter and additional keywords
        to the parent class.

        NOTE: It overrides the directory argument if given.

        Parameters
        ----------
        name : str
            The name of the filter one wants to load.
        """
        try:
            kwargs.pop("directory")
        except KeyError:
            pass

        super(Filter, self).__init__(
            kname=name, directory=f"{__data__}/filters", **kwargs
        )
