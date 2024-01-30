import logging
import os
import time
from pathlib import Path
from typing import Callable, Union

import astropy.units as u
import numpy as np
import yaml
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
from matplotlib import pyplot as mpl
from numpy.typing import NDArray
from scipy import ndimage as nd

from flaeming.data import static_variables as sv
from flaeming.data.base import PhotTable


logger = logging.getLogger(__name__)

with open(f"{os.path.dirname(__file__)}/info_images.yaml", "r") as f:
    image_info = yaml.safe_load(f)


def save_cutouts(
    phot_table: PhotTable,
    fits_file: str,
    suffix: str,
    size: u.Quantity,
    output_dir: str = ".",
) -> None:
    with fits.open(fits_file) as hdul:
        # Get the image data and header
        data = hdul[0].data
        header = hdul[0].header

        # Loop over each row in the coordinates table
        for _, row in phot_table.table.iterrows():
            name = row[phot_table.ID]
            output_filename = f"{output_dir}/{name}/stamp_{suffix}.fits"
            if os.path.isfile(output_filename):
                logger.info(f"{output_filename} exists, skipping.")
                continue
            
            os.makedirs(f"{output_dir}/{name}/", exist_ok=True)

            coord = SkyCoord(row[phot_table.RA], row[phot_table.DEC], unit="deg")

            # Create a Cutout2D object
            cutout = Cutout2D(data, coord, size=size, wcs=WCS(header))

            # Save the cutout as a new FITS file
            hdu = fits.PrimaryHDU(cutout.data, header=cutout.wcs.to_header())

            logger.debug(f"Saving cutout to {output_filename}")
            hdu.writeto(output_filename, overwrite=True)

    return None


def __get_stamp_info(file_name: str) -> dict[str, str]:
    logger.debug(f"Processing {file_name}")
    info = file_name.split("_")[-1].split("-")
    return {sv.STAMP_SURVEY: info[0], sv.STAMP_FILTER: info[1]}


def stamps_quick_check(stamps: dict[str, NDArray], **kwargs):
    n_stamps = len(stamps)
    fig, ax = mpl.subplots(1, n_stamps)
    if n_stamps == 1:
        ax = [ax]

    for i, (key, data) in enumerate(stamps.items()):
        ax[i].imshow(data, **kwargs)
        ax[i].set_title(key)
        ax[i].tick_params(labelleft=False, labelbottom=False)

    return fig, ax


def create_rgb(
    stamps: dict[str, NDArray], rgb_filters: list[str], debug=False, **kwargs
) -> NDArray:
    """_summary_

    :param stamps: _description_
    :type stamps: dict[str, NDArray]
    :param rgb_filters: The ordered filter list. It assumes there are three filters,
        and they correspond, in order, to r, g, and b channels.
    :type rgb_filters: list[str]
    :param debug: _description_, defaults to False
    :type debug: bool, optional
    :return: _description_
    :rtype: NDArray
    """
    assert len(rgb_filters) == 3, f"`rgb_filters` must have exactly three elements"
    assert all(
        fltr in stamps.keys() for fltr in rgb_filters
    ), f"At least one of rgb_filters {rgb_filters} has no corresponding stamp in {stamps.keys()}"
    image_list = [stamps[fltr] for fltr in rgb_filters]

    image = make_lupton_rgb(*image_list, **kwargs)
    if debug:
        mpl.imshow(image)
    return image


def load_stamps(
    folder: str,
    surveys: Union[list[str], str],
    filters: Union[list[str], str],
    size: tuple[int] = (64, 64),
) -> dict[str, NDArray]:
    folder_path = Path(folder)
    file_list = folder_path.glob("*.fits")

    if isinstance(surveys, str):
        surveys = [surveys]
    if isinstance(filters, str):
        filters = [filters]

    normalized_filters = []
    for fltr in filters:
        normalized_filters += fltr.split("+")

    stamps = {}
    for file in file_list:
        stamp_info = __get_stamp_info(file.stem)
        logger.debug(f"{file.name} has info {stamp_info}")

        if (
            stamp_info[sv.STAMP_SURVEY] in surveys
            and stamp_info[sv.STAMP_FILTER] in normalized_filters
        ):
            stamps[stamp_info[sv.STAMP_FILTER]] = __get_standard_size_stamp(
                fits.getdata(file), size
            )

    for fltr in filters:
        if "+" in fltr:
            individual_filters = fltr.split("+")
            stamps[fltr] = np.sum(
                [stamps[f] for f in individual_filters], axis=0
            ) / len(individual_filters)

    logger.info(f"Loaded filters are {stamps.keys()}")
    return stamps


def create_rgb_images(
    folder: Path,
    stamps: dict[str, NDArray],
    filter_list: list[str],
    extension: str = "jpg",
    **kwargs,
):
    out_name = f"{folder.absolute()}/rgb_{''.join(filter_list)}.{extension}"
    logger.info(f"Saving rgb to {out_name}")
    create_rgb(stamps, filter_list, debug=False, filename=out_name, **kwargs)
    return


def create_image_dataset(
    func: Callable,
    folder_root: Path,
    source_names: list[str],
    filter_list: list[str],
    surveys: list[str],
    size: tuple[int] = (64, 64),
    **kwargs,
):
    n = 0
    n_galaxies = len(source_names)
    for folder in folder_root.iterdir():
        if folder.name in source_names:
            if n % 1000 == 0 and n > 0:
                print(f"\t ====> Processed {n}/{n_galaxies}.")

            stamps = load_stamps(folder.absolute(), surveys, filter_list, size=size)
            func(folder, stamps, filter_list, **kwargs)
            del stamps
            n += 1
    return


def __edge_pad_or_cut(size_diff):
    if size_diff % 2 == 0:
        start, end = size_diff // 2, -(size_diff // 2)
    else:
        start, end = size_diff // 2, -(size_diff // 2) - 1

    logger.debug(f"Limits for {size_diff} are {start}:{end}")
    return start, end


def __get_standard_size_stamp(stamp_image: NDArray, size: tuple[int]):
    N, M = stamp_image.shape
    if N == size[0] and M == size[1]:
        return stamp_image

    n_diff = N - size[0]
    m_diff = M - size[1]

    if n_diff < 0 or m_diff < 0:
        raise NotImplementedError

    n_start, n_end = __edge_pad_or_cut(n_diff)
    m_start, m_end = __edge_pad_or_cut(m_diff)
    return stamp_image[n_start:n_end, m_start:m_end]


def create_numpy_array(
    folder_root: Path,
    source_names: list[str],
    filter_list: list[str],
    surveys: list[str],
    size: tuple[int] = (64, 64),
):
    n_galaxies = len(source_names)
    n_x, n_y = size
    n_channels = len(filter_list)

    image_array = np.empty([n_galaxies, n_x, n_y, n_channels])

    for i, name in enumerate(source_names):
        folder = os.path.join(folder_root, name)
        stamps = load_stamps(folder, surveys, filter_list, size=size)
        for j, fltr in enumerate(filter_list):
            image_array[i, :, :, j] = stamps[fltr]

    return image_array


def validate_stamp(filepath: str, nan_threshold: int = 0):
    image_data = fits.getdata(filepath)
    min_value = np.nanmin(image_data)
    max_value = np.nanmax(image_data)

    num_nans = len(image_data[np.isnan(image_data)])
    has_nans = num_nans > nan_threshold

    file_identifier = f"{filepath.parent}/{filepath.name}"
    if min_value == max_value:
        logger.warning(f"{file_identifier} image is invalid (min=max)")
        return "nodata"
    elif has_nans:
        logger.warning(
            f"{file_identifier} image is invalid (found {num_nans} NaN values)"
        )
        return "nans"
    else:
        return "ok"
