import pathlib
import shutil

import dclab
import numpy as np
from PIL import Image

from cytopix import png_io
from cytopix import seg_session

data_path = pathlib.Path(__file__).parent / 'data'


def test_dc_to_png_files(tmp_path):
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


def test_png_files_to_dc(tmp_path):
    """Test converting PNG files to PNG"""
    png_dir = tmp_path / "png"
    dc_path = tmp_path / "data.rtdc"
    png_dir.mkdir()
    # First, create a few grayscale PNG files
    for ii in range(10):
        image = np.zeros((80, 320), dtype=np.uint8)
        image[0, ii] = 10 * ii
        Image.fromarray(image).save(png_dir / f"image_{ii:02d}.png")

    # Then, convert those files to .rtdc
    png_io.png_files_to_dc(
        png_paths=png_dir,
        dc_path=dc_path,
    )

    # open the file
    with dclab.new_dataset(dc_path) as ds:
        assert len(ds) == 10
        assert ds["image"][1][0, 1] == 10
        assert ds["image"][2][0, 2] == 20
        assert ds["image"][3][0, 3] == 30
        assert ds["image"][4][0, 4] == 40
        assert ds["image"][5][0, 5] == 50
        assert ds["image"][6][0, 6] == 60
        assert ds["image"][7][0, 7] == 70
        assert ds["image"][8][0, 8] == 80
        assert ds["image"][9][0, 9] == 90
