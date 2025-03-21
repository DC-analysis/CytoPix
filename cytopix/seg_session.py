import pathlib
from typing import Dict

import dclab
from dcevent.dc_segment import get_available_segmenters
from dcevent.cli import task_process
from dcnum.segm import MPOSegmenter, Segmenter
import h5py
import numpy as np
import scipy.fftpack
import scipy.interpolate
from skimage.draw import polygon2mask
from skimage import measure


class SegmentDisabled(MPOSegmenter):
    requires_background_correction = False
    mask_postprocessing = True
    mask_default_kwargs = {
        "clear_border": False,
        "fill_holes": False,
        "closing_disk": 0,
    }

    def segment_algorithm(self, image):
        return np.zeros_like(image, dtype=bool)


get_available_segmenters.cache_clear()
_def_methods = get_available_segmenters()
if "mlunetcpu" in _def_methods:
    # only there if e.g. pytorch is available
    DEFAULT_SEGMENTER_CLASS = _def_methods["mlunetcpu"]
else:
    # Fallback segmenter
    DEFAULT_SEGMENTER_CLASS = _def_methods["watershed"]


class SegmentationSession:
    def __init__(self,
                 path_rtdc: pathlib.Path | str,
                 path_session: pathlib.Path | str = None,
                 segmenter_class: Segmenter = None,
                 segmenter_kwargs: Dict = None,
                 ):
        """Create and manipulate .dcseg manual segmentation session files

        In contrast to regular .rtdc files, mask data in .dcseg
        files are stored as np.uint8 arrays where each integer
        value corresponds to one mask. Since the user can
        freely choose with which label to draw, there may be multiple
        masks with the same label identifier. This class helps
        to manage these cases.

        Notes
        -----
        If the input .rtdc file used for labeling contains duplicate
        frames, then only one label image is used in the .dcseg file,
        the other image will be all-zero!

        Everything should still work if the frames in the input .rtdc
        file are not in order, but this is untested.
        """
        path_rtdc = pathlib.Path(path_rtdc)
        if path_session is None:
            path_session = path_rtdc.with_suffix(".dcseg")
        if path_session.suffix != ".dcseg":
            path_session = path_session.with_name(path_session.name + ".dcseg")

        if segmenter_class is None:
            segmenter_class = DEFAULT_SEGMENTER_CLASS

        # read-in session information
        self.ds = dclab.new_dataset(path_rtdc)
        self.path_session = path_session
        with h5py.File(path_session, "a") as h5:
            size = len(self.ds)
            sx, sy = self.ds["image"][0].shape
            umask = h5.require_dataset(name="user_labels",
                                       shape=(size, sx, sy),
                                       # not more than 254 events in a frame
                                       dtype=np.uint8)

            h5.require_dataset(name="events_processed",
                               shape=(len(self.ds),),
                               dtype=float)
            if "user_contours" in h5:
                # It was decided that the manual segmentation will only
                # use mask data, no contour data.
                ucont = h5["user_contours"]
                # Convert (initial) old contour keys to new keys.
                for key in ucont:
                    if not key.count(":"):
                        old_index = int(key)
                        frame_num = self.ds["frame"][old_index]
                        event_in_frame = 0
                        while True:
                            # Don't override anything
                            new_key = f"{int(frame_num)}:{event_in_frame}"
                            if new_key not in ucont:
                                break
                            else:
                                event_in_frame += 1
                        data = ucont[key]
                        del ucont[key]
                        ucont[new_key] = data
                # Convert contour data to labeled image data
                for key in ucont:
                    frame, subidx = [int(it) for it in key.split(":")]
                    label = subidx + 1
                    cont = ucont[key]
                    mask = polygon2mask((sx, sy), np.array(cont)[:, ::-1] - .5)
                    idx = self.get_index(frame)
                    umask[idx] += np.array(mask * label, dtype=np.uint8)
                del h5["user_contours"]

        self.path_rtdc = path_rtdc
        self.path_session = path_session
        self._complete = False
        #: list of unique frames in the original .rtdc file
        self.unique_frames = sorted(
            np.unique(np.array(self.ds["frame"][:], dtype=np.uint64)))
        #: current frame of interest
        self.current_frame = None
        #: current event of interest in the current frane
        self.current_event_in_frame = 0

        # figure out how we want to do mask processing
        self.segm_kwargs = segmenter_kwargs
        if "model_file" in segmenter_kwargs:
            pass  # nothing to do here
        elif hasattr(segmenter_class, "default_models"):
            # determine default model
            self.segm_kwargs["model_file"] = (
                task_process.infer_ml_model_file_from_input(
                    segmentation_method=segmenter_class.get_ppid_code(),
                    data_path=self.path_rtdc,
                    pixel_size=None))

        self.segm_class = segmenter_class
        self.segm = self.segm_class(**self.segm_kwargs)
        self.segm_func = self.segm.segment_algorithm_wrapper()

        print(f"Segmenter class is: {self.segm_class}")
        print(f"Segmenter kwargs: {self.segm_kwargs}")

    def __len__(self):
        """Number of events in the original .rtdc dataset"""
        return len(self.ds)

    @property
    def current_index(self):
        """Current index in the original dataset

        If there are duplicate frames in the orginal dataset,
        then this always returns the first index.
        """
        if self.current_frame is None:
            self.current_frame = self.unique_frames[0]
        return self.get_index(self.current_frame)

    @property
    def current_index_unique(self):
        """Current index in `self.unique_frames`

        Here we are not enumerating the original dataset, but
        the unique frames therein.
        """
        if self.current_frame is None:
            self.current_frame = self.unique_frames[0]
        return self.unique_frames.index(self.current_frame)

    @property
    def complete(self):
        if not self._complete:
            with h5py.File(self.path_session, "a") as h5:
                if np.nanmin(h5["events_processed"][:] > 0) > 0:
                    self._complete = True
        return self._complete

    def get_event(self,
                  frame: int,
                  event_in_frame: int = 0):
        """Get event data for this index"""
        frame = int(frame)
        index = self.get_index(frame)
        image = self.ds["image"][index]
        if "image_bg" in self.ds:
            image_bg = self.ds["image_bg"][index]
        else:
            image_bg = np.median(image)
        # get the already-processed contour, otherwise the original one
        mask = self.get_mask(frame, event_in_frame)
        self.current_frame = frame
        self.current_event_in_frame = event_in_frame
        return image, image_bg, mask, frame

    def get_index(self, frame):
        if frame is None:
            index = -1
        else:
            cands = np.where(self.ds["frame"] == frame)[0]
            if cands.size == 0:
                raise IndexError(f"Could not find video frame {frame}!")
            index = cands[0]
        return index

    def get_mask(self, frame, event_in_frame):
        index = self.get_index(frame)
        with h5py.File(self.path_session) as h5:
            mask = h5["user_labels"][index] == event_in_frame + 1
        return mask

    def get_labels(self, frame):
        index = self.get_index(frame)
        with h5py.File(self.path_session) as h5:
            labels = h5["user_labels"][index][:]
            if np.sum(labels) == 0:  # user did not edit this frame
                # retrieve image with optional background correction
                if self.segm_class.requires_background_correction:
                    image = (np.array(self.ds["image"][index], dtype=int)
                             - self.ds["image_bg"][index])
                else:
                    image = self.ds["image"][index]
                mask = self.segm_func(image)
                # perform optional mask postprocessing
                if self.segm_class.mask_postprocessing:
                    mask = self.segm_class.process_mask(
                        mask, **self.segm.kwargs_mask)
                # label the masks
                labels = measure.label(mask, background=0)
        return labels

    def get_next_frame(self):
        """Get the next unedited events (fallback to +1 if all were edited)"""
        if self.current_frame is None:
            frame = self.unique_frames[0]
        else:
            # If there are two consecutive frames that are identical, we
            # have to skip to the second-next index.
            curidx = self.unique_frames.index(self.current_frame)
            nextidx = (curidx + 1) % len(self.unique_frames)
            frame = self.unique_frames[nextidx]

        image, image_bg, _, _ = self.get_event(frame)
        labels = self.get_labels(frame)
        return image, image_bg, labels, frame

    def get_prev_frame(self):
        """Get the events from the previous frame"""
        curidx = self.unique_frames.index(self.current_frame)
        nextidx = (curidx - 1) % len(self.unique_frames)
        frame = self.unique_frames[nextidx]

        image, image_bg, _, _ = self.get_event(frame)
        labels = self.get_labels(frame)
        return image, image_bg, labels, frame

    def invalidate_frame(self, frame):
        """That particular image cannot be processed"""
        index = self.get_index(frame)
        with h5py.File(self.path_session, "a") as h5:
            h5["events_processed"][index] = np.nan

    def write_user_labels(self, frame, labels):
        frame = int(frame)
        index = self.get_index(frame)
        with h5py.File(self.path_session, "a") as h5:
            h5["user_labels"][index] = labels
            h5["events_processed"][index] = np.max(labels)

    def write_user_mask(self, frame, event_in_frame, mask):
        """Write the floating point contour data to the segmentation session"""
        frame = int(frame)
        index = self.get_index(frame)
        with h5py.File(self.path_session, "a") as h5:
            # Only save one single mask
            h5["events_processed"][index] = max(
                h5["events_processed"][index], event_in_frame + 1)
            label = event_in_frame + 1
            label_img = h5["user_labels"][index]
            label_img[label_img == label] = 0
            label_img[mask] = label
            h5["user_labels"][index] = label_img


def smoothen_contour(cont):
    """Return a smooth, depixelated version of a contour

    If this algorithm was not applied to the contours, then manual
    segmentation would be very cumbersome. The contour is defined
    on a discretized grid which dramatically increases manual labor.
    The solution is to smoothen the contour with smooth splines.
    """
    x = cont[:, 0]
    y = cont[:, 1]
    # get the cumulative distance along the contour
    dist = np.sqrt((x[:-1] - x[1:]) ** 2 + (y[:-1] - y[1:]) ** 2)
    dist_along = np.concatenate(([0], dist.cumsum()))
    spline, u = scipy.interpolate.splprep([x, y], u=dist_along, s=len(x)/10)

    length = dist_along[-1]
    out_length = int(max(5, length // 5))

    # resample it at smaller distance intervals
    interp_d = np.linspace(dist_along[0], dist_along[-1], out_length)
    interp_x, interp_y = scipy.interpolate.splev(interp_d, spline)

    res = np.zeros((out_length, 2))
    res[:, 0] = interp_x
    res[:, 1] = interp_y
    return res
