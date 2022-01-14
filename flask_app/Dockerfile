FROM arm64v8/ubuntu:18.04

# setup environment
# ENV DEBIAN_FRONTEND noninteractive
# ENV QEMU_EXECVE 1
# COPY apt.conf /etc/apt/apt.conf
ENV TZ=Asia/Karachi
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update -y
RUN apt-get install -y --fix-missing \
        software-properties-common \
        apt-utils \
        apt-transport-https \
        sudo \
        wget \
        curl \
        rsync \
        vim \
        git \
        expect \
        terminator \
        bash-completion \
        net-tools \
        build-essential \
        pkg-config \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        python3-dev \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# tensor flow dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libhdf5-serial-dev \
        hdf5-tools \
        libhdf5-dev \
        zlib1g-dev \
        zip \
        libjpeg8-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED 1

RUN pip3 install --upgrade pip==21.0.1
RUN pip3 install --upgrade wheel
RUN pip3 install -U pip testresources setuptools==56.0.0 
# grpcio takes forever to build... (tensorflow dependency)
RUN pip3 install --no-cache-dir --force-reinstall -Iv grpcio
# install newer numpy... the python3-numpy package is 0xb, whereas pytorch is compiled against 0xc
RUN pip3 install --upgrade numpy==1.19.3
RUN pip3 install --upgrade Cpython
# RUN apt install cython3
# ... and so does the libraries below (i.e. they take a while to build)
# RUN pip3 install -U --no-cache-dir h5py==2.10
RUN apt-get install libhdf5-dev
# RUN apt-get install python3-h5py
# RUN pip3 install -U --no-build-isolation h5py
# ARG HDF5_DIR="/usr/lib/aarch64-linux-gnu/hdf5/serial/"
# RUN pip3 install --no-cache-dir --verbose h5py==2.10.0
# RUN pip3 install --no-cache-dir --verbose future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
# RUN apt install make g++ cmake
# RUN pip3 install --upgrade pip \
#     && git clone -b v1.8.1 https://github.com/onnx/onnx.git \
#     && cd onnx \
#     && git submodule update --init --recursive \
#     && sed -i -e "s/numpy>=1.16.6/numpy<=1.19.0/g" setup.py \
#     && python3 setup.py install \
#     && cd .. && rm -rf onnx \
#     && pip3 install gdown \
#     && pip3 install cmake \
#     && pip3 install ninja \
#     && pip3 install yapf \
#     && pip3 install six \
#     && pip3 install wheel \
#     && pip3 install moc \
#     && pip3 install cython \
#     && pip3 install keras_applications==1.0.8 --no-deps \
#     && pip3 install keras_preprocessing==1.1.0 --no-deps \
#     && pip3 install numpy==1.19.0 \
#     && pip3 install h5py==2.9.0 \
#     && pip3 install pybind11 \
#     && ldconfig
# RUN git clone https://github.com/h5py/h5py.git --branch 2.10.0 && cd h5py/ && pip3 install .
RUN H5PY_SETUP_REQUIRES=0 pip3 install --no-deps --no-build-isolation h5py==2.10.0
# RUN sudo env H5PY_SETUP_REQUIRES=0 pip3 install -U --no-build-isolation h5py==3.1.0
RUN pip3 install pandas

# install tensorflow dependencies
RUN pip3 install \
    astor gast six \
    protobuf tensorflow_estimator \
    absl-py tensorboard \
    keras-applications keras-preprocessing \
    py-cpuinfo psutil portpicker mock requests termcolor wrapt google-pasta \
    pillow
# RUN pip3 install scikit-learn # fails installing (installed as last layer)

# install other useful python libraries
RUN pip3 install \
    docopt \
    tornado \
    moviepy \
    greenlet \
    proglog \
    imageio-ffmpeg \
    MarkupSafe \
    python-engineio \
    python-socketio \
    click \
    itsdangerous \
    Jinja2 \
    flask \
    gunicorn \
    dnspython \
    monotonic \
    eventlet

# opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas-dev \
        build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev \
        libv4l-dev v4l-utils qv4l2 \
    && rm -rf /var/lib/apt/lists/*

# ref: https://jkjung-avt.github.io/opencv-on-nano/
# ref: https://devtalk.nvidia.com/default/topic/1049972/jetson-nano/opencv-cuda-python-with-jetson-nano/1
# ref: https://github.com/AastaNV/JEP/blob/master/script/install_opencv4.0.0_Nano.sh
# ref: https://github.com/jkjung-avt/jetson_nano/blob/master/install_opencv-3.4.6.sh
# ARG OPENCV=3.4.6
# ARG OPENCV=4.1.0
# not using these options: -D WITH_QT=ON -D WITH_OPENGL=ON
# RUN wget https://github.com/opencv/opencv/archive/${OPENCV}.tar.gz -O /tmp/opencv-${OPENCV}.tar.gz > /dev/null 2>&1 \
#     && wget https://github.com/opencv/opencv_contrib/archive/${OPENCV}.tar.gz -O /tmp/opencv_contrib-${OPENCV}.tar.gz > /dev/null 2>&1 \
#     && cd /tmp \
#     && tar zxvf opencv-${OPENCV}.tar.gz \
#     && tar zxvf opencv_contrib-${OPENCV}.tar.gz \
#     && cd opencv-${OPENCV}/ \
#     && mkdir build \
#     && cd build/ \
#     && cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D CUDA_ARCH_BIN="5.3" -D CUDA_ARCH_PTX="" -D WITH_CUBLAS=ON -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-${OPENCV}/modules -D WITH_GSTREAMER=ON -D ENABLE_NEON=ON -D OPENCV_ENABLE_NONFREE=ON -D WITH_LIBV4L=ON -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF .. \
#     && make -j$(nproc) \
#     && make install \
#     && rm -rf /tmp/opencv*

# RUN python3 -m pip install --upgrade pip
# RUN add-apt-repository universe
RUN pip3 install opencv-python 
# RUN apt-get install python3-opencv
RUN python3 -m pip install -U tensorflow -f https://tf.kmtea.eu/whl/stable.html -f https://ext.kmtea.eu/whl/stable.html
# installing tensorflow
# https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html
# https://devtalk.nvidia.com/default/topic/1048776/official-tensorflow-for-jetson-nano-/
# ARG TENSORFLOW_WHL=tensorflow_gpu-1.13.1+nv19.5-cp36-cp36m-linux_aarch64.whl
# RUN wget https://developer.download.nvidia.com/compute/redist/jp/v42/tensorflow-gpu/${TENSORFLOW_WHL} -O /tmp/${TENSORFLOW_WHL} > /dev/null 2>&1 \
#     && pip3 install /tmp/${TENSORFLOW_WHL} \
#     && rm /tmp/${TENSORFLOW_WHL}

# keras dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         python-pydot python-pydot-ng \
#         graphviz \
#         libatlas-base-dev \
#         python3-scipy \
#     && rm -rf /var/lib/apt/lists/*
# install keras
# RUN pip3 install keras

# install pytorch 1.1.0
# https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/
# https://github.com/pytorch/pytorch/blob/master/README.md#nvidia-jetson-platforms
# https://nvidia.box.com/v/torch-stable-cp36-jetson-jp42/
# RUN wget https://nvidia.box.com/shared/static/j2dn48btaxosqp0zremqqm8pjelriyvs.whl -O /tmp/torch-1.1.0-cp36-cp36m-linux_aarch64.whl > /dev/null 2>&1 \
#     && pip3 install /tmp/torch-1.1.0-cp36-cp36m-linux_aarch64.whl \
#     && rm /tmp/torch-1.1.0-cp36-cp36m-linux_aarch64.whl

# torchvision
# TODO: needs libcudart.so.10.0 to compile torchvision
# RUN cd /tmp \ 
#     && git clone https://github.com/pytorch/vision \
#     && cd vision \
#     && python3 setup.py install \
#     && cd .. \
#     && rm -rf vision
# RUN apt-get install ffmpeg libsm6 libxext6  -y
# additional tools & libraries
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         libblas3 liblapack3 \
#         gstreamer1.0-tools libxv1 \
#         gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-gl \
#         libclutter-1.0-0 libclutter-gst-3.0-0 libavresample3 libpostproc54 libzmq5 librubberband2 libmysofa0 \
#         ffmpeg \
#     && rm -rf /var/lib/apt/lists/*

# RUN pip3 install Cython
# RUN pip3 install scikit-learn

# COPY cam-test.sh /opt/tools/
# COPY tf-cuda-test.py /opt/tools/
# COPY tegra-cam.py /opt/tools/


# setup docker user
# ARG user=jetson
# ARG group=jetson
# ARG uid=1000
# ARG gid=1000
# ARG home=/home/jetson
# RUN groupadd -g ${gid} ${group} \
#     && useradd -d ${home} -u ${uid} -g ${gid} -m -s /bin/bash ${user} \
#     && echo "${user} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/sudoers_${user} \
#     && addgroup --gid 108 i2c \
#     && usermod -a -G i2c ${user} \
#     && addgroup --gid 104 input \
#     && usermod -a -G input ${user} \
#     && usermod -a -G video ${user}
#     # && usermod -aG docker ${user}

# RUN cd /usr/local && ln -s cuda-10.0 cuda
# RUN pip install -r requirements.txt
WORKDIR usr/src/flask_app
COPY . .


# USER ${user}
# WORKDIR ${home}

# ENTRYPOINT [ "/entrypoint.sh" ]
# CMD [ "/bin/bash" ]