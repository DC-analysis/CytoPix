import logging
import pathlib
from typing import Dict

import dclab
from dcnum.segm import MPOSegmenter, Segmenter, get_available_segmenters
import h5py
import numpy as np

from skimage import measure


logger = logging.getLogger(__name__)


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


class SegmentationSession:
    def __init__(self,
                 path_dc: pathlib.Path | str,
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
        the other image will be all-zero.

        Everything should still work if the frames in the input .rtdc
        file are not in order, but this is untested.
        """
        path_dc = pathlib.Path(path_dc)
        if path_session is None:
            path_session = path_dc.with_suffix(".dcseg")
        if path_session.suffix != ".dcseg":
            path_session = path_session.with_name(path_session.name + ".dcseg")

        if segmenter_class is None:
            segmenter_class = _def_methods["thresh"]

        # read-in session information
        self.ds = dclab.new_dataset(path_dc)
        self.path_session = path_session
        with h5py.File(path_session, "a") as h5:
            size = len(self.ds)
            sx, sy = self.ds["image"][0].shape
            # label information
            h5.require_dataset(name="user_labels",
                               shape=(size, sx, sy),
                               # not more than 254 events in a frame
                               dtype=np.uint8)
            # indicates how many labels are processed (int) or invalid (np.nan)
            h5.require_dataset(name="events_processed",
                               shape=(len(self.ds),),
                               dtype=float)

        self.path_dc = path_dc
        self.path_session = path_session
        # whether segmentation is complete
        self._complete = False
        #: list of unique frames in the original .rtdc file
        self.unique_frames = sorted(
            np.unique(np.array(self.ds["frame"][:], dtype=np.uint64)))
        #: current frame of interest
        self.current_frame = None
        #: current event of interest in the current frame
        self.current_event_in_frame = 0

        # figure out how we want to do mask processing
        self.segm_kwargs = segmenter_kwargs or {}

        self.segm_class = segmenter_class
        self.segm = self.segm_class(**self.segm_kwargs)
        self.segm_func = self.segm.segment_algorithm_wrapper()

        logger.info(f"Segmenter class is: {self.segm_class}")
        logger.info(f"Segmenter kwargs: {self.segm_kwargs}")

    def __len__(self):
        """Number of events in the original .rtdc dataset"""
        return len(self.ds)

    @property
    def current_index(self):
        """Current index in the original dataset

        If there are duplicate frames in the original dataset,
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
        """Property indicating whether the current session is complete"""
        if not self._complete:
            with h5py.File(self.path_session, "a") as h5:
                if np.nanmin(h5["events_processed"][:] > 0) > 0:
                    self._complete = True
        return self._complete

    def get_image_bg(self, frame: int):
        """Return background image"""
        index = self.get_index(frame)
        if "image_bg" in self.ds:
            image_bg = self.ds["image_bg"][index]
        else:
            image_bg = np.full(self.ds["image"].shape[1:],
                               np.median(self.ds["image"][index]))
        return image_bg

    def get_event(self,
                  frame: int,
                  event_in_frame: int = 0):
        """Get event data for this index"""
        frame = int(frame)
        index = self.get_index(frame)
        image = self.ds["image"][index]
        image_bg = self.get_image_bg(frame)
        # get the already-processed contour, otherwise the original one
        mask = self.get_mask(frame, event_in_frame)
        self.current_frame = frame
        self.current_event_in_frame = event_in_frame
        return image, image_bg, mask, frame

    def get_index(self, frame: int):
        """Return the index in the dataset given a frame number"""
        if frame is None:
            index = -1
        else:
            cands = np.where(self.ds["frame"] == frame)[0]
            if cands.size == 0:
                raise IndexError(f"Could not find frame {frame}!")
            index = cands[0]
        return index

    def get_mask(self, frame: int, event_in_frame: int):
        """Return mask of an event (event index starts at 0) in a frame"""
        index = self.get_index(frame)
        with h5py.File(self.path_session) as h5:
            mask = h5["user_labels"][index] == event_in_frame + 1
        return mask

    def get_labels(self, frame: int):
        """Return full integer label image for a frame index"""
        index = self.get_index(frame)
        with h5py.File(self.path_session) as h5:
            labels = h5["user_labels"][index][:]
            if np.sum(labels) == 0:  # user did not edit this frame
                # retrieve image with optional background correction
                if self.segm_class.requires_background_correction:
                    image = (np.array(self.ds["image"][index], dtype=int)
                             - self.get_image_bg(frame))
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

    def get_labeled_indices(self):
        """Return indices in `path_dc` that have been labeled"""
        with h5py.File(self.path_session, "a") as h5:
            processed = h5["events_processed"][:]

        # We have to remove all indices that have not been procecces
        indices = np.arange(len(processed))
        used = np.ones_like(indices, dtype=bool)
        used[np.isnan(processed)] = False
        return indices[used]

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

    def invalidate_frame(self, frame: int):
        """That particular image cannot be processed"""
        index = self.get_index(frame)
        with h5py.File(self.path_session, "a") as h5:
            h5["events_processed"][index] = np.nan

    def write_user_labels(self,
                          frame: int,
                          labels: np.ndarray):
        """Write the user-defined labels to the session file"""
        frame = int(frame)
        index = self.get_index(frame)
        with h5py.File(self.path_session, "a") as h5:
            h5["user_labels"][index] = labels
            h5["events_processed"][index] = np.max(labels)

    def write_user_mask(self,
                        frame: int,
                        event_in_frame: int,
                        mask: np.ndarray):
        """Write the mask data of one event in a frame to the session file

        This overrides any previously labeling data for this particular
        event but preserves other labeling information (unless there is
        overlap).
        """
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
