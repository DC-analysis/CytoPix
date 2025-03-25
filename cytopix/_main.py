import logging


class DevNull:
    """Effectively a file-like object for piping everything to nothing."""

    def write(self, *args, **kwargs):
        pass


class PrettyFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(levelname)s %(name)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def main():
    from importlib import resources
    import multiprocessing as mp
    import sys
    from PyQt6 import QtWidgets, QtCore, QtGui

    mp.freeze_support()

    # In case we have a frozen application, and we encounter errors
    # in subprocesses, then these will try to print everything to stdout
    # and stderr. However, if we compiled the app with PyInstaller with
    # the --noconsole option, sys.stderr and sys.stdout are None and
    # an exception is raised, breaking the program.
    if sys.stdout is None:
        sys.stdout = DevNull()
    if sys.stderr is None:
        sys.stderr = DevNull()

    # Tell the root logger to pretty-print logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(PrettyFormatter())
    root_logger.addHandler(handler)

    from .main import CytoPix

    app = QtWidgets.QApplication(sys.argv)
    ref_ico = resources.files("cytopix.img") / "cytopix_icon.png"
    with resources.as_file(ref_ico) as path_icon:
        app.setWindowIcon(QtGui.QIcon(str(path_icon)))

    # Use dots as decimal separators
    QtCore.QLocale.setDefault(QtCore.QLocale(QtCore.QLocale.c()))

    window = CytoPix(*app.arguments()[1:])  # noqa: F841

    sys.exit(app.exec())
