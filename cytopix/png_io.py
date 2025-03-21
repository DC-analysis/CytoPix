import errno
import hashlib
import logging
import os
import pathlib

from dcnum import write
import h5py
import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)


def png_files_to_dc(png_paths: list | str | pathlib.Path,
                    dc_path: pathlib.Path):
    """Convert a set of PNG files to an .rtdc file

    Parameters
    ----------
    png_paths:
        List of PNG files or a directory containing PNG files.
        If a directory is specified, then it is search for PNG files
        recursively.
    dc_path:
        Path to an .rtdc file that is created from the .png files.
        If the .rtdc file already exists and the input file names and
        sizes match that of the path specified, then the existing file
        is used. Otherwise, a FileExistsError is raised.
    """
    # Get list of input PNG files
    input_files = []
    if isinstance(png_paths, (pathlib.Path, str)):
        path = pathlib.Path(png_paths)
        if path.is_dir():
            input_files += path.rglob("*.png", case_sensitive=False)
        else:
            input_files += [png_paths]
    input_files = sorted(input_files)

    logger.info(f"Loading {input_files} PNG files...")

    # Get a list of file names and file sizes
    png_sizes = [pp.st_size for pp in input_files]
    png_names = [pp.name for pp in input_files]

    # Compute a unique hash of the input files
    hasher = hashlib.md5()
    hasher.update(
        "".join([f"{info}" for info in zip(png_names, png_sizes)]).encode())
    png_hash = hasher.hexdigest()

    # If the output file already exists, check whether the hashes match.
    log_name = "logs/cytopix-png-files"
    if dc_path.exists():
        file_matches = False
        with h5py.File(dc_path) as h5:
            if log_name in h5 and h5[log_name].attrs["hash"] == png_hash:
                # We can reuse this file
                logger.info(f"Reusing existing file {dc_path}")
                file_matches = True

        if not file_matches:
            logger.info(f"Cannot use existing {dc_path} (hash mismatch)")
            raise FileExistsError(
                errno.EEXIST, os.strerror(errno.EEXIST), str(dc_path))
    else:
        # This is the actual workload of this function. Populate the .rtdc
        # file with the image data.
        logger.info(f"Writing .rtdc file {dc_path}")
        with write.HDF5Writer(dc_path) as hw:
            # store the input file information as a log
            log_ds = hw.store_log(
                log=log_name,
                data=[f"{d[0]} {d[1]}" for d in zip(input_files, png_sizes)]
            )
            log_ds.attrs["hash"] = png_hash

            # store the image data to the output file
            image_data = []
            data_size = 0
            for ii in range(len(input_files)):
                im = np.array(Image.open(input_files[ii]), dtype=np.uint8)
                if len(im.shape) == 3:
                    # convert RGB to grayscale by taking the red channel
                    im = im[:, :, 0]
                data_size += im.size
                image_data.append(im)

                # write buffered images to output
                if data_size > 25_600_000:  # ~24 MB
                    hw.store_feature_chunk(
                        feat="image",
                        data=np.array(image_data),
                    )
                    image_data.clear()
                    data_size = 0
