#!/bin/sh

cd modules/chamfer
python setup.py build_ext --inplace

cd ../emd
python setup.py build_ext --inplace

cd ..
