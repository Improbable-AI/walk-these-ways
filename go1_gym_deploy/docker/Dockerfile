# syntax=docker/dockerfile:experimental

FROM nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3

#ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
#ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# add new sudo user
ENV USERNAME improbable
ENV HOME /home/$USERNAME
RUN useradd -m $USERNAME && \
        echo "$USERNAME:$USERNAME" | chpasswd && \
        usermod --shell /bin/bash $USERNAME && \
        usermod -aG sudo $USERNAME && \
        mkdir /etc/sudoers.d && \
        echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
        chmod 0440 /etc/sudoers.d/$USERNAME && \
        # Replace 1000 with your user/group id
        usermod  --uid 1000 $USERNAME && \
        groupmod --gid 1000 $USERNAME


# install package
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        sudo \
        less \
        emacs \
        apt-utils \
        tzdata \
        git \
        tmux \
        bash-completion \
        command-not-found \
        libglib2.0-0 \
        gstreamer1.0-plugins-* \
        libgstreamer1.0-* \
        libgstreamer-plugins-*1.0-* \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

#COPY config/nvidia_icd.json /usr/share/vulkan/icd.d/


USER root


#RUN apt-get update && apt-get install -y python3-pip && pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html


# ==================================================================
# Useful Libraries for Development
# ------------------------------------------------------------------
#RUN apt update && apt install -y apt-transport-https ca-certificates curl software-properties-common
#RUN curl -fsSL https://download.sublimetext.com/sublimehq-pub.gpg | apt-key add - && add-apt-repository "deb https://download.sublimetext.com/ apt/stable/" && apt update && apt install sublime-text


# ==================================================================
# Python dependencies defined in requirements.txt
# ------------------------------------------------------------------
#RUN pip3 install --upgrade pip
# copy local requirements file for pip install python deps
#COPY ./requirements.txt /home/$USERNAME
#WORKDIR /home/$USERNAME
#RUN pip3 install -r requirements.txt

# LCM
RUN apt-get -y update && apt-get install -y make gcc-8 g++-8
RUN cd /home/$USERNAME && git clone https://github.com/lcm-proj/lcm.git && cd lcm && mkdir build && cd build && cmake .. && make -j && make install
RUN cd /home/$USERNAME/lcm/lcm-python && pip3 install -e .


RUN apt-get install -y vim
#RUN pip3 install pandas

# ROS
# ENV ROS_DISTRO melodic

RUN apt-get install -y gnupg

# COPY install_scripts/install_ros.sh /tmp/install_ros.sh
# RUN chmod +x /tmp/install_ros.sh
# RUN /tmp/install_ros.sh

# # bootstrap rosdep
# RUN rosdep init \
#   && rosdep update

# # create catkin workspace
# ENV CATKIN_WS=/root/catkin_ws
# RUN bash /opt/ros/melodic/setup.bash
# RUN mkdir -p $CATKIN_WS/src
# WORKDIR ${CATKIN_WS}
# RUN catkin init
# RUN catkin config --extend /opt/ros/$ROS_DISTRO \
#     --cmake-args -DCMAKE_BUILD_TYPE=Release -DCATKIN_ENABLE_TESTING=False
# WORKDIR $CATKIN_WS/src


RUN apt-get update && apt-get install -y freeglut3-dev libudev-dev
#COPY ./install_scripts/install_vision_opencv.sh /tmp/install_vision_opencv.sh
#RUN chmod +x /tmp/install_vision_opencv.sh
#RUN /tmp/install_vision_opencv.sh


RUN apt-get install -y libgl1-mesa-dev libudev1 libudev-dev


#RUN apt-get install unzip
#
#RUN cd ~ && \
#    wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.1.zip && \
#    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.1.zip && \
#    unzip opencv.zip && \
#    unzip opencv_contrib.zip && \
#    mv opencv-4.5.1 opencv && \
#    mv opencv_contrib-4.5.1 opencv_contrib && \
#    rm opencv.zip && \
#    rm opencv_contrib.zip
#
#RUN cd ~/opencv && \
#    mkdir build && \
#    cd build && \
#    cmake -D CMAKE_BUILD_TYPE=RELEASE \
#        -D CMAKE_INSTALL_PREFIX=/usr \
#        -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
#        -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
#        -D WITH_OPENCL=OFF \
#        -D WITH_CUDA=OFF \
#        -D CUDA_ARCH_BIN=5.3 \
#        -D CUDA_ARCH_PTX="" \
#        -D WITH_CUDNN=OFF \
#        -D WITH_CUBLAS=OFF \
#        -D ENABLE_FAST_MATH=ON \
#        -D CUDA_FAST_MATH=OFF \
#        -D OPENCV_DNN_CUDA=OFF \
#        -D ENABLE_NEON=ON \
#        -D WITH_QT=OFF \
#        -D WITH_OPENMP=ON \
#        -D WITH_OPENGL=ON \
#        -D BUILD_TIFF=ON \
#        -D WITH_FFMPEG=ON \
#        -D WITH_GSTREAMER=ON \
#        -D WITH_TBB=ON \
#        -D BUILD_TBB=ON \
#        -D BUILD_TESTS=OFF \
#        -D WITH_EIGEN=ON \
#        -D WITH_V4L=ON \
#        -D WITH_LIBV4L=ON \
#        -D OPENCV_ENABLE_NONFREE=ON \
#        -D INSTALL_C_EXAMPLES=OFF \
#        -D INSTALL_PYTHON_EXAMPLES=OFF \
#        -D BUILD_NEW_PYTHON_SUPPORT=ON \
#        -D BUILD_opencv_python3=TRUE \
#        -D OPENCV_GENERATE_PKGCONFIG=ON \
#        -D BUILD_EXAMPLES=OFF .. && \
#    make -j4 && cd ~ && \
#    # sudo rm -r /usr/include/opencv4/opencv2 && \
#    cd ~/opencv/build && \
#    sudo make install && \
#    sudo ldconfig && \
#    make clean && \
#    sudo apt-get update

RUN apt-get install -y libgtk2.0-dev pkg-config
RUN pip3 install opencv-python opencv-contrib-python

####################################################################################
###### START HERE -- Install whatever dependencies you need specific to this project!
####################################################################################


#COPY ./rsc/IsaacGym_Preview_2_Package.tar.gz /home/$USERNAME/
#RUN cd /home/$USERNAME && tar -xvzf IsaacGym_Preview_2_Package.tar.gz
#COPY ./rsc/learning_to_walk_in_minutes.zip /home/$USERNAME/
#RUN apt-get install unzip && cd /home/$USERNAME/ && unzip learning_to_walk_in_minutes.zip && cd ./code/rl-pytorch && pip3 install -e .
#RUN cd /home/$USERNAME/isaacgym/python && pip3 install -e .
#RUN cd /home/$USERNAME/code/isaacgym_anymal && pip3 install -e .
#COPY ./src/isaacgym_anymal/ /home/$USERNAME/code/isaacgym_anymal/

# setup entrypoint
COPY entrypoint.sh /

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
