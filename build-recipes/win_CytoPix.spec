# -*- mode: python ; coding: utf-8 -*-
from os.path import exists
import warnings

import cytopix

NAME = "CytoPix"

if not exists("./{}Launcher.py".format(NAME)):
    warnings.warn("Cannot find {}Launcher.py'! ".format(NAME) +
                  "Please run pyinstaller from the 'build-recipes' directory.")

gui_a = Analysis(
    [NAME + "Launcher.py"],
    pathex=["."],
    hookspath=["."],
    runtime_hooks=None)

gui_pyz = PYZ(gui_a.pure)

gui_splash = Splash(
    '../docs/artwork/cytopix_splash.png',
    binaries=gui_a.binaries,
    datas=gui_a.datas,
    text_pos=(44, 163),
    text_size=10,
    text_color='black',
    minify_script=True)

gui_exe = EXE(
      gui_pyz,
      gui_a.scripts,
      gui_splash,
      [],
      exclude_binaries=True,
      name=NAME + ".exe",
      debug=False,
      strip=False,
      upx=False,
      icon=NAME + ".ico",
      console=bool(cytopix.__version__.count("post") or cytopix.__version__.count("a")))

coll = COLLECT(
    gui_exe,
    gui_a.binaries,
    gui_a.zipfiles,
    gui_a.datas,
    gui_splash.binaries,
    strip=False,
    upx=False,
    name=NAME)
