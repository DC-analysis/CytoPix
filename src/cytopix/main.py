import colorsys
from importlib import import_module
import logging
import pathlib
import signal
import sys
import traceback
import webbrowser

from dcnum.meta import paths as dcnum_paths
from dcnum.segm import get_available_segmenters
import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QKeySequence, QShortcut
import pyqtgraph as pg
from scipy.ndimage import binary_fill_holes
from skimage import segmentation


from ._version import version
from .main_ui import Ui_MainWindow
from . import colorize
from . import png_io
from . import seg_session
from . import splash


pg.setConfigOptions(imageAxisOrder='row-major')


class CytoPix(QtWidgets.QMainWindow):

    def __init__(self, *arguments):
        """Initialize CytoPix GUI

        If you pass the "--version" command line argument, the
        application will print the version after initialization
        and exit.
        """
        super(QtWidgets.QMainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Disable native menu bar (e.g. on macOS)
        self.ui.menubar.setNativeMenuBar(False)

        self.wid = self.ui.graphics_widget

        self.v1a = self.wid.addViewBox(row=1, col=0, lockAspect=True)
        self.v1a.setDefaultPadding(0)

        self.pg_image = pg.ImageItem(np.arange(80 * 320).reshape((80, -1)))

        self.v1a.addItem(self.pg_image)
        self.v1a.disableAutoRange('xy')
        self.v1a.autoRange()

        kern = np.array([[0]])
        self.pg_image.setDrawKernel(kern, mask=kern, center=(0, 0),
                                    mode=self.on_draw)

        self.pg_image.setLevels([1, 80*320])

        #: current visualization state:
        #: - 0: show labels with different colors
        #: - 1: show all labels as one color
        self.vis_state = 0

        # Current image as grayscale
        self.image = None
        self.image_bg = None
        self.subtract_bg = True
        self.labels = None
        self.show_labels = True
        self.current_drawing_label = 1  # 1 to 9
        self.auto_contrast = True
        self.label_saturation = 0.4
        self.segses = None

        # Settings are stored in the .ini file format. Even though
        # `self.settings` may return integer/bool in the same session,
        # in the next session, it will reliably return strings. Lists
        # of strings (comma-separated) work nicely though.
        QtCore.QCoreApplication.setOrganizationName("DC-Analysis")
        QtCore.QCoreApplication.setOrganizationDomain("mpl.mpg.de")
        QtCore.QCoreApplication.setApplicationName("CytoPix")
        QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
        #: CytoPix settings
        self.settings = QtCore.QSettings()

        # register search paths with dcnum
        for path in self.settings.value("segm/torch_model_files", []):
            path = pathlib.Path(path)
            if path.is_dir():
                dcnum_paths.register_search_path("torch_model_files", path)

        self.logger = logging.getLogger(__name__)

        # GUI
        self.setWindowTitle(f"CytoPix {version}")

        # File menu
        self.ui.actionSegmentRrtdcFile.triggered.connect(
            self.on_action_segment_rtdc)
        self.ui.actionSegmentPngImages.triggered.connect(
            self.on_action_segment_png)
        self.ui.actionExportImages.triggered.connect(
            self.on_action_export_images)
        self.ui.actionQuit.triggered.connect(self.on_action_quit)
        # Help menu
        self.ui.actionDocumentation.triggered.connect(self.on_action_docs)
        self.ui.actionSoftware.triggered.connect(self.on_action_software)
        self.ui.actionAbout.triggered.connect(self.on_action_about)

        cwid = self.centralWidget()

        # Shortcuts
        # Keyboard shortcuts
        # navigate next
        self.shortcut_next = QShortcut(QKeySequence('Right'), cwid)
        self.shortcut_next.activated.connect(self.goto_next)
        # navigate previous
        self.shortcut_prev = QShortcut(QKeySequence('Left'), cwid)
        self.shortcut_prev.activated.connect(self.goto_prev)
        # hide mask
        self.shortcut_toggle_cont = QShortcut(QKeySequence('Space'), cwid)
        self.shortcut_toggle_cont.activated.connect(self.toggle_mask)
        # visualization state
        self.shortcut_toggle_vis = QShortcut(QKeySequence('V'), cwid)
        self.shortcut_toggle_vis.activated.connect(self.next_visualization)
        # auto-contrast
        self.shortcut_toggle_contr = QShortcut(QKeySequence('C'), cwid)
        self.shortcut_toggle_contr.activated.connect(self.toggle_contrast)
        # background correction
        self.shortcut_toggle_bg = QShortcut(QKeySequence('B'), cwid)
        self.shortcut_toggle_bg.activated.connect(self.toggle_background)
        # Current label
        self.shortcuts_label = []
        for ii in range(1, 10):
            sci = QShortcut(QKeySequence(str(ii)), cwid)
            sci.activated.connect(self.change_drawing_label)
            self.shortcuts_label.append(sci)
        # Label saturation
        self.shortcut_plus = QShortcut(QKeySequence('.'), cwid)
        self.shortcut_plus.activated.connect(self.saturation_plus)
        self.shortcut_minus = QShortcut(QKeySequence('-'), cwid)
        self.shortcut_minus.activated.connect(self.saturation_minus)

        # if "--version" was specified, print the version and exit
        if "--version" in arguments:
            print(version)
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 300)
            sys.exit(0)

        splash.splash_close()

        # finalize
        self.show()
        self.activateWindow()
        self.setWindowState(QtCore.Qt.WindowState.WindowActive)
        self.showMaximized()

        for arg in arguments:
            if isinstance(arg, str):
                pp = pathlib.Path(arg)
                if pp.suffix in [".rtdc", ".dc"]:
                    self.on_action_segment_rtdc(pp)
                    break
                elif pp.is_dir():
                    self.on_action_segment_png(pp)
                    break

    @QtCore.pyqtSlot(QtCore.QEvent)
    def dragEnterEvent(self, e):
        """Whether files are accepted"""
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    @QtCore.pyqtSlot(QtCore.QEvent)
    def dropEvent(self, e):
        """Add dropped files to view"""
        urls = sorted(e.mimeData().urls())
        pathlist = [pathlib.Path(ff.toLocalFile()) for ff in urls]
        if pathlist:
            # check whether the first file is a DC file
            pp0 = pathlist[0]
            if pp0.is_file() and pp0.suffix in [".rtdc", ".dc"]:
                self.on_action_segment_rtdc(pp0)
                return
            # we have a list of PNG files or a directory
            png_files = [pp for pp in pathlist
                         if pp.is_file() and pp.suffix == ".png"]
            if png_files:
                self.on_action_segment_png(png_files)
            else:
                # recurse into the directory
                png_files_recursive = []
                for pp in pathlist:
                    if pp.is_dir():
                        png_files_recursive += sorted(pp.rglob("*.png"))
                self.on_action_segment_png(png_files_recursive)

    @QtCore.pyqtSlot()
    def on_action_about(self) -> None:
        """Show imprint."""
        gh = "DC-analysis/CytoPix"
        rtd = "cytopix.readthedocs.io"
        about_text = (
            f"CytoPix. GUI for pixel-based manual segmentation of DC images."
            f"<br><br>"
            f"Author: Paul Müller and others<br>"
            f"GitHub: "
            f"<a href='https://github.com/{gh}'>{gh}</a><br>"
            f"Documentation: "
            f"<a href='https://{rtd}'>{rtd}</a><br>")  # noqa 501
        QtWidgets.QMessageBox.about(self,
                                    f"CytoPix {version}",
                                    about_text)

    @QtCore.pyqtSlot()
    def on_action_export_images(self, path=None):
        if path is None:
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Select directory for PNG export",
            )
        if path:
            png_io.dc_to_png_files(
                dc_path=self.segses.path_dc,
                png_dir=path,
                export_labels=True,
                indices=self.segses.get_labeled_indices(),
            )

    @QtCore.pyqtSlot()
    def on_action_segment_png(self, path=None):
        """Open a dialog to load a directory of PNG files"""
        if path is None:
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                'Select directory containing PNG images',
                )
        if path:
            path = pathlib.Path(path)
            # convert directory of PNG images to .rtdc
            dc_path = path.with_name(path.name + ".rtdc")
            png_files = [p for p in path.glob("*.png")
                         if not p.name.endswith("_label.png")]
            try:
                png_io.png_files_to_dc(png_files, dc_path)
            except FileExistsError:
                QtWidgets.QMessageBox.critical(
                    self,
                    "PNG session inconsistency",
                    f"CytoPix detected an inconsistency while comparing the "
                    f"PNG files in '{path}' and the .rtdc file generated "
                    f"in a previous session.\n\n"
                    f"This usually means that somebody modified the PNG files "
                    f"or added/removed new files. This is a problem, because "
                    f"CytoPix creates a static .rtdc file from these PNG "
                    f"files.\n\n "
                    f"To resolve this issue, you can either "
                    f"\n- delete the file '{dc_path}' and its associated "
                    f"session file '{dc_path.with_suffix('.dcseg').name}' "
                    f"(This will destroy your previous session),"
                    f"\n- restore the PNG directory to its previous state, or"
                    f"\n- directly open '{dc_path}' instead."
                )
            else:
                # open session
                self.open_session(dc_path)

    @QtCore.pyqtSlot()
    def on_action_segment_rtdc(self, path=None):
        """Open dialog to add a single .rtdc file"""
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                'Select DC data',
                '',
                'RT-DC data (*.rtdc)')
        if path:
            # open session
            self.open_session(path)

    @QtCore.pyqtSlot()
    def on_action_docs(self):
        webbrowser.open("https://cytopix.readthedocs.io")

    @QtCore.pyqtSlot()
    def on_action_software(self) -> None:
        """Show used software packages and dependencies."""
        libs = ["dcnum",
                "h5py",
                "numpy",
                "pillow",
                "pyqtgraph",
                "torch",
                ]

        sw_text = f"CytoPix {version}\n\n"
        sw_text += f"Python {sys.version}\n\n"
        sw_text += "Modules:\n"
        for lib in libs:
            try:
                mod = import_module(lib)
            except ImportError:
                pass
            else:
                sw_text += f"- {mod.__name__} {mod.__version__}\n"
        sw_text += f"- PyQt6 {QtCore.QT_VERSION_STR}\n"

        QtWidgets.QMessageBox.information(self, "Software", sw_text)

    @QtCore.pyqtSlot()
    def on_action_quit(self) -> None:
        """Determine what happens when the user wants to quit"""
        self.save_labels()
        QtCore.QCoreApplication.quit()

    def open_session(self, path):
        segmenter_class = get_available_segmenters()["thresh"]
        segmenter_kwargs = {
            "kwargs_mask": {
                "clear_border": False,  # for training, we need all events
                "fill_holes": True,
                "closing_disk": 1
            }
        }
        self.segses = seg_session.SegmentationSession(
            path_dc=path,
            segmenter_class=segmenter_class,
            segmenter_kwargs=segmenter_kwargs)
        self.show_event(*self.segses.get_next_frame())
        # give graphics widget mouse/event focus
        self.wid.setFocus()

    @QtCore.pyqtSlot()
    def closeEvent(self, event):
        """Determine what happens when the user wants to quit"""
        self.save_labels()
        event.accept()

    @QtCore.pyqtSlot()
    def change_drawing_label(self):
        sender = self.sender()
        self.current_drawing_label = int(sender.key().toString())
        self.ui.label_label_current.setText(
            f"<b>{self.current_drawing_label}</b>")

    def get_labels_from_ui(self):
        return np.copy(self.labels) if self.labels is not None else None

    @QtCore.pyqtSlot()
    def goto_next(self):
        """Go to next unlabeled event"""
        self.save_labels()
        if self.segses:
            # get the next event
            self.show_event(*self.segses.get_next_frame())

    @QtCore.pyqtSlot()
    def goto_prev(self):
        """Go one frame back"""
        self.save_labels()
        if self.segses:
            self.show_event(*self.segses.get_prev_frame())

    def on_draw(self, dk, image, mask, ss, ts, ev):
        """Called when the user draws"""
        if self.image is None:
            self.logger.warning("No image data, not drawing")
            return
        elif self.show_labels is None:
            return

        if hasattr(ev, "isFinish"):
            # drag ended
            finish = ev.isFinish()
        else:
            # only single click
            finish = True

        fill_holes = finish
        # Set the pixel value accordingly.
        mdf = ev.modifiers().value
        if mdf == (QtCore.Qt.Modifier.SHIFT | QtCore.Qt.Modifier.ALT).value:
            # delete the mask at that location
            segmentation.flood_fill(self.labels,
                                    seed_point=(ts[0].start, ts[1].start),
                                    new_value=0,
                                    in_place=True)
        else:
            delete = mdf == QtCore.Qt.Modifier.SHIFT.value
            value = 0 if delete else self.current_drawing_label
            self.labels[ts] = value

        if finish:
            # redraw all labels
            self.update_plot(fill_holes=fill_holes)
        else:
            # only draw white points where user dragged mouse
            image_rgb = np.array(self.pg_image.image, dtype=int)
            image_rgb[ts] = 250
            image_rgb = np.array(np.clip(image_rgb, 0, 255), dtype=np.uint8)
            self.update_plot(image_rgb=image_rgb, draw_labels=False)

    def saturation_minus(self):
        self.label_saturation -= .05
        self.label_saturation = max(self.label_saturation, .05)
        self.update_plot()  # inefficient, but no optimization necessary

    def saturation_plus(self):
        self.label_saturation += .05
        self.label_saturation = min(self.label_saturation, 1)
        self.update_plot()  # inefficient, but no optimization necessary

    def save_labels(self):
        """Save the current contour in the session"""
        if self.segses:
            frame = self.segses.current_frame
            labels = self.get_labels_from_ui()
            if labels is not None:
                self.segses.write_user_labels(frame, labels)

    def show_event(self, image, image_bg, labels, frame):
        self.image = np.array(image, dtype=int)
        self.image_bg = image_bg
        self.labels = np.array(labels, dtype=np.uint8)

        # reset the view
        self.v1a.setRange(
            xRange=(0, self.image.shape[1]),
            yRange=(0, self.image.shape[0]))

        self.update_plot()

        if self.segses:
            totframes = len(self.segses.unique_frames)
            self.ui.lineEdit_source.setText(str(self.segses.path_dc))
            self.ui.lineEdit_session.setText(str(self.segses.path_session))
            self.ui.label_frame.setText(
                f"{self.segses.current_index_unique + 1}/{totframes}"
            )

    @QtCore.pyqtSlot()
    def toggle_background(self):
        self.subtract_bg = not self.subtract_bg
        self.update_plot()

    @QtCore.pyqtSlot()
    def toggle_contrast(self):
        self.auto_contrast = not self.auto_contrast
        self.update_plot()

    @QtCore.pyqtSlot()
    def update_plot(self, fill_holes=False, draw_labels=True, image_rgb=None):
        """Update plot in case visualization changed

        The number of calls to this function should be minimized.
        However, since we are dealing with small images, anything
        to reduce the number of calls is premature optimization.
        """
        if self.image is None:
            self.logger.warning("Not updating plot, no data loaded")
            return
        if self.subtract_bg:
            image = self.image - self.image_bg + 128
        else:
            image = self.image

        if image_rgb is not None:
            pass
        elif self.show_labels and draw_labels:
            if self.vis_state == 0:
                self.ui.widget_labels.setVisible(True)
                labels = self.labels
                if fill_holes:
                    for ii in range(1, labels.max() + 1):
                        maski = labels == ii
                        labels[binary_fill_holes(maski)] = ii
            else:
                self.ui.widget_labels.setVisible(False)
                labels = self.labels > 0
                if fill_holes:
                    labels = binary_fill_holes(labels)
            # draw RGB image
            image_rgb, hues = colorize.colorize_image_with_labels(
                image,
                labels=labels,
                saturation=self.label_saturation,
                ret_hues=True)
            colab = "Frame Labels: "
            for ii, lid in enumerate(hues):
                r, g, b = np.array(colorsys.hsv_to_rgb(hues[lid], 1, 1)) * 255
                color = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
                cur_lab = (f"<span "
                           f"style='color:{color}; background-color:black'"
                           f"><b>{lid}</b></span>")
                colab += cur_lab

                if ii + 1 == self.current_drawing_label:
                    self.ui.label_label_current.setText(cur_lab)

            self.ui.label_labels_available.setText(colab)

        else:
            # draw grayscale image
            image_rgb = image

        (rx1, rx2), (ry1, ry2) = np.array(self.v1a.viewRange(), dtype=int)
        cropped = image[slice(max(0, ry1), ry2), slice(max(0, rx1), rx2)]

        if self.auto_contrast and cropped.size:
            levels = (min(120, cropped.min()), max(136, cropped.max()))
        else:
            levels = (0, 255)

        # adjust contrast according to currently visible area
        kwargs = dict(
            levels=levels,
            levelMode="mono"
        )

        self.pg_image.setImage(image_rgb, **kwargs)

    @QtCore.pyqtSlot()
    def next_visualization(self):
        self.vis_state = (self.vis_state + 1) % 2
        self.update_plot()

    @QtCore.pyqtSlot()
    def toggle_mask(self):
        self.show_labels = not self.show_labels
        self.update_plot()


def excepthook(etype, value, trace):
    """
    Handler for all unhandled exceptions.

    :param `etype`: the exception type (`SyntaxError`,
        `ZeroDivisionError`, etc...);
    :type `etype`: `Exception`
    :param string `value`: the exception error message;
    :param string `trace`: the traceback header, if any (otherwise, it
        prints the standard Python header: ``Traceback (most recent
        call last)``.
    """
    vinfo = f"Unhandled exception in CytoPix version {version}:\n"
    tmp = traceback.format_exception(etype, value, trace)
    exception = "".join([vinfo]+tmp)
    try:
        # Write to the control logger, so errors show up in the
        # cytopix-warnings log.
        main = get_main()
        main.control.logger.error(exception)
    except BaseException:
        # If we send things to the logger and everything is really bad
        # (e.g. cannot write to output hdf5 file or so, then we silently
        # ignore this issue and only print the error message below.
        pass
    QtWidgets.QMessageBox.critical(
        None,
        "CytoPix encountered an error",
        exception
    )


def get_main():
    app = QtWidgets.QApplication.instance()
    for widget in app.topLevelWidgets():
        if isinstance(widget, QtWidgets.QMainWindow):
            return widget


# Make Ctr+C close the app
signal.signal(signal.SIGINT, signal.SIG_DFL)
# Display exception hook in separate dialog instead of crashing
sys.excepthook = excepthook
