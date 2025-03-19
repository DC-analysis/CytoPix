import PyQt6


class DevNull:
    """Effectively a file-like object for piping everything to nothing."""
    def write(self, *args, **kwargs):
        pass


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

    from .main import CytoPix

    app = QtWidgets.QApplication(sys.argv)
    ref_ico = resources.files("cytopix.img") / "cytopix_icon.png"
    with resources.as_file(ref_ico) as path_icon:
        app.setWindowIcon(QtGui.QIcon(str(path_icon)))

    # Use dots as decimal separators
    QtCore.QLocale.setDefault(QtCore.QLocale(QtCore.QLocale.c()))

    window = CytoPix(*app.arguments()[1:])  # noqa: F841

    sys.exit(app.exec())
