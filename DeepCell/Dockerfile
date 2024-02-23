# This Dockerfile is used to generate the docker image dsarchive/histomicstk
# This docker image includes the HistomicsTK python package along with its
# dependencies.
#
# All plugins of HistomicsTK should derive from this docker image

FROM tensorflow/tensorflow:2.8.0-gpu

# Needed for GPU access in plugins
LABEL com.nvidia.volumns.needed="nvidia_driver"
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# The following are implemented in the DeepCell-tf Dockerfile
# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

LABEL maintainer="Sam Border - Computational Microscopy and Imaging Laboratory. <samuel.border@medicine.ufl.edu>"

RUN apt-get update && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    libcurl4-openssl-dev \
    libexpat1-dev \
    graphviz \
    unzip \
    libhdf5-dev \
    software-properties-common \
    libssl-dev \
    # Standard build tools \
    build-essential \
    cmake \
    autoconf \
    automake \
    libtool \
    pkg-config \
    libmemcached-dev && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update ##[edited]

RUN apt-get install libxml2-dev libxslt1-dev -y

# Required for opencv-python (cv2)
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

RUN apt-get update && apt-get install libpq-dev gdal-bin libgdal-dev -y
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /
# Make Python3 the default and install pip.  Whichever is done last determines
# the default python version for pip.

#
RUN which  python && \
    python --version

ENV build_path=$PWD/build
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Setting deepcell access token (expires after 90 days)
ARG DEEPCELL_ACCESS_TOKEN=DEEPCELL_ACCESS_TOKEN
ENV DEEPCELL_ACCESS_TOKEN=$DEEPCELL_ACCESS_TOKEN

# Copying location of deepcell plugin scripts
ENV deep_cell_path=$PWD/deepcell_plugin
RUN mkdir -p $deep_cell_path

RUN apt-get update && \
    apt-get install -y --no-install-recommends memcached && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY . $deep_cell_path/
WORKDIR $deep_cell_path

#   Upgrade setuptools, as the version in Conda won't upgrade cleanly unless it
# is ignored.

# Installing packages in setup.py
RUN pip install --no-cache-dir --upgrade --ignore-installed pip setuptools && \
    pip install --no-cache-dir . && \
    rm -rf /root/.cache/pip/*

# Show what was installed
RUN python --version && pip --version && pip freeze

# define entrypoint through which all CLIs can be run
WORKDIR $deep_cell_path/deepcell_plugin/cli

# Test our entrypoint.  If we have incompatible versions of numpy and
# openslide, one of these will fail
RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli
RUN python -m slicer_cli_web.cli_list_entrypoint DeepCell_Plugin --help


ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]




















