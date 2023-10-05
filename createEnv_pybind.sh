#!/bin/bash
# -*- coding: utf-8 -*-
# info: create pybind python and cpp headers

python3 -m venv envPybind
source envPybind/bin/activate
export TMPDIR=${cDir}

pip3 install --no-cache-dir pybind11 pybind11[global] mpi4py
pip3 install --no-cache-dir torch torchvision torchaudio

#eof