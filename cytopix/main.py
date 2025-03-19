import atexit
from importlib import import_module, resources
import inspect
import logging
import pathlib
import signal
import sys
import traceback
import webbrowser

from dcnum.feat import feat_background
from dcnum.meta import paths as dcnum_paths
from dcnum.segm import get_available_segmenters
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QStandardPaths

from ._version import version
from .main_ui import Ui_MainWindow
from . import png_io
from . import splash


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
        pathlist = [ pathlib.Path(ff.toLocalFile()) for ff in urls ]
        if pathlist:
            # check whether the first file is a DC file
            pp0 = pathlist[0]
            if pp0.is_file() and pp0.suffix in [".rtdc", ".dc"]:
                self.on_action_segment_rtdc(pp0)
                return
            # we have a list of PNG files or a directory
            png_files = [ pp for pp in pathlist
                          if pp.is_file() and pp.suffix == ".png" ]
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
    def on_action_export_images(self):
        raise NotImplementedError("Not implemented")


    @QtCore.pyqtSlot()
    def on_action_segment_png(self, path=None):
        """Open a dialog to load a directory of PNG files"""
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenDirectoryName(
                self,
                'Select directory of PNG images',
                '')
        if path:
            # convert directory of PNG images to .rtdc
            dc_path = path.with_name(path.name + ".rtdc")
            png_io.png_files_to_dc(path, dc_path)
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
        QtCore.QCoreApplication.quit()


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
