import errno
import hashlib
import logging
import os
import pathlib
import itertools

import dclab
from dcnum import write
import h5py
import numpy as np
from PIL import Image

from .seg_session import SegmentationSession

logger = logging.getLogger(__name__)


def dc_to_png_files(dc_path: pathlib.Path | str,
                    png_dir: pathlib.Path | str,
                    export_labels: bool = False,
                    indices: np.ndarray = None,
                    ):
    """Export images from an .rtdc file to PNG

    Parameters
    ----------
    dc_path: pathlib.Path
        Input .rtdc file
    png_dir: pathlib.Path
        PNG files output directory
    export_labels: bool
        If specified, save uint8 label images alongside input images
        (with additional `_label` stem suffix) using the .dcseg session file.
    indices: np.ndarray
        Only export images with given indices
    """
    dc_path = pathlib.Path(dc_path)
    png_dir = pathlib.Path(png_dir)
    png_dir.mkdir(exist_ok=True, parents=True)

    if export_labels:
        ses = SegmentationSession(dc_path)
    else:
        ses = None

    with dclab.new_dataset(dc_path) as ds:

        if indices is None:
            indices = np.arange(len(ds))
        digits = len(str(int(np.max(indices))))

        for idx in indices:
            im = Image.fromarray(ds["image"][idx])
            im.save(png_dir / f"image_{str(idx).zfill(digits)}.png")
            if ses is not None:
                # store label image
                frame = ds["frame"][idx]
                larr = ses.get_labels(frame)
                # normalize
                # limit in case there are >255 labels
                max_val = min(larr.max(), 255)
                larr = np.array(larr / max_val * 255, dtype=np.uint8)
                lim = Image.fromarray(larr)
                lim.save(png_dir / f"image_{str(idx).zfill(digits)}_label.png")


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
    else:
        input_files += list(png_paths)
    input_files = sorted(input_files)

    if not input_files:
        raise ValueError("No PNG files specified or found!")

    logger.info(f"Loading {len(input_files)} PNG files...")

    # Get a list of file names and file sizes
    png_sizes = [pp.stat().st_size for pp in input_files]
    png_names = [pp.name for pp in input_files]

    # Compute a unique hash of the input files
    hasher = hashlib.md5()
    hasher.update(
        "".join([f"{info}" for info in zip(png_names, png_sizes)]).encode())
    png_hash = hasher.hexdigest()

    # If the output file already exists, check whether the hashes match.
    log_name = "cytopix-png-files"
    if dc_path.exists():
        file_matches = False
        with h5py.File(dc_path) as h5:
            if (log_name in h5["logs"]
                    and h5["logs"][log_name].attrs["hash"] == png_hash):
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

            num_images = len(input_files)
            hw.store_feature_chunk(
                feat="frame",
                data=np.arange(1, num_images+1),
            )

            # store the image data to the output file
            image_size = Image.open(input_files[0]).size
            chunk_size = hw.get_best_nd_chunks(item_shape=image_size,
                                               feat_dtype=np.uint8)[0]

            for indices in itertools.batched(range(num_images), chunk_size):
                image_data = []
                for ii in indices:
                    im = np.array(Image.open(input_files[ii]), dtype=np.uint8)
                    if len(im.shape) == 3:
                        # convert RGB to grayscale by taking the red channel
                        im = im[:, :, 0]
                    image_data.append(im)
                # store the image chunk
                hw.store_feature_chunk(
                    feat="image",
                    data=np.array(image_data, dtype=np.uint8),
                )
