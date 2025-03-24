import pathlib

import numpy as np

from cytopix import seg_session


data_path = pathlib.Path(__file__).parent / 'data'


def test_basic_session(tmp_path):
    """Test basic session functionality"""
    # open session
    ses = seg_session.SegmentationSession(
        path_dc=data_path / "blood_minimal.rtdc",
        path_session=tmp_path / "blood_minimal_session.dcseg",
    )
    assert not ses.complete

    # write a label image to the file
    labels = ses.get_labels(240105)
    labels[:] = 0
    labels[:, 0] = 1
    labels[:, 2] = 2
    labels[:, 3] = 3

    ses.write_user_labels(frame=240105, labels=labels)

    assert np.all(ses.get_labels(240105) == labels)
