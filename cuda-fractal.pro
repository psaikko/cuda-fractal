TEMPLATE = app
TARGET = cuda-fractal
INCLUDEPATH += .
DEFINES += QT_DEPRECATED_WARNINGS

# Input
HEADERS += fractalcompute.h viewer.h
SOURCES += main.cpp viewer.cpp

CONFIG += c++14
QT += widgets

#
# CUDA configuration adapted from 
# https://stackoverflow.com/a/27055971
#

# Define output directories
CUDA_OBJECTS_DIR = release/cuda

# CUDA settings
CUDA_SOURCES += fractalcompute.cu

# include paths
INCLUDEPATH += $$(CUDA_HOME)/include

# library directories
QMAKE_LIBDIR += $$(CUDA_HOME)/lib64

# Add the necessary libraries
LIBS += -lcuda -lcudart

# Configuration of the Cuda compiler
cuda.input = CUDA_SOURCES
cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
cuda.commands = $$(CUDA_HOME)/bin/nvcc -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
cuda.dependency_type = TYPE_C
QMAKE_EXTRA_COMPILERS += cuda
