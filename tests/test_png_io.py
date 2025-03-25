import pathlib
import shutil

import numpy as np
from PIL import Image

from cytopix import png_io
from cytopix import seg_session

data_path = pathlib.Path(__file__).parent / 'data'


def test_png_export(tmp_path):
    """Test basic png export"""
    shutil.copy2(data_path / "blood_minimal.rtdc", tmp_path)
    # open session
    ses = seg_session.SegmentationSession(
        path_dc=tmp_path / "blood_minimal.rtdc",
    )
    assert not ses.complete

    # write a label image to the file
    labels = ses.get_labels(240105)
    labels[:] = 0
    labels[:, 0] = 1
    labels[:, 2] = 2
    labels[:, 3] = 3

    ses.write_user_labels(frame=240105, labels=labels)

    index = ses.get_index(240105)

    png_dir = tmp_path / "png"

    # export png files
    png_io.dc_to_png_files(
        dc_path=ses.path_dc,
        png_dir=png_dir,
        export_labels=True,
    )

    # check whether that worked
    assert len(list(png_dir.glob("*.png"))) == 56

    # open a png file and check labels
    im = np.array(Image.open(png_dir / f"image_{index}_label.png"))
    assert np.all(im[:, 0] == int(255 / 3))
    assert np.all(im[:, 1] == 0)
    assert np.all(im[:, 2] == int(255 / 3 * 2))
    assert np.all(im[:, 3] == int(255))
