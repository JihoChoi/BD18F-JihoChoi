
# =======================================================
#
#   BDDL 2018 HW 02 **Ultimate Setup Script**
#
# =======================================================
#
#   https://github.com/swsnu/bd2018/tree/master/hw2
#   https://github.com/snuspl/parallax/blob/master/doc/installation.md
#
#   1. Executing line by line is recommended 
#       - Source URLs might have changed
#       - The default version of packages might have changed
#
#   2. The setup takes about 2.5 hours just by executing the commands
#       - pull, wget, compile, ...
#
#   3. After the setup, whenever re-login the ssh session
#       - SOURCE >> source parallax_venv/bin/activate
#
#
#
#   Jiho Choi (jihochoi@snu.ac.kr)
#       - https://github.com/JihoChoi
#
# =======================================================

# -------------------------------------------------------
#   Setup for Ubuntu 16.04 on AWS (2 cores, 8G RAM, Without GPU **)
# -------------------------------------------------------


# Install packages
#       https://github.com/swsnu/bd2018/issues/22
lsb_release -a

sudo apt-get update -y
sudo apt-get install python-dev python-pip net-tools vim openssh-server git wget unzip tar autoconf automake libtool -y
python --version # --> Python 2.7.12
pip --version    # --> pip 8.1.1

cd ~ # $HOME -> /home/ubuntu
git clone https://github.com/swsnu/bd2018.git



# -------------------------------------------------------
# OpenMPI 3.0.0 Setup                                     <-- Takse about 10 minutes
# -------------------------------------------------------

# OpenMPI dependencies
sudo apt-get install autoconf automake libtool

# 4.0.0 not yet tested in Parallax

# cd ~
# wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.0.tar.gz
# gunzip -c openmpi-4.0.0.tar.gz | tar xf -
# cd openmpi-4.0.0
# ./configure --prefix=/usr/local
# sudo make all install
# ls /usr/local/lib

cd ~
wget https://download.open-mpi.org/release/open-mpi/v3.0/openmpi-3.0.0.tar.gz
gunzip -c openmpi-3.0.0.tar.gz | tar xf -
cd openmpi-3.0.0
./configure --prefix=/usr/local
sudo make all install
ls /usr/local/lib


# =======================================================
# Parallax
# =======================================================
# Parallax
#     - TensorFlow
#         - Bazel (Google Build System)                   <-- Takse about 2 hours
#     - Horovod
#         - OpenMPI
# =======================================================

# -------------------------------------------------------
# Bazel
#       https://docs.bazel.build/versions/master/install-ubuntu.html
# -------------------------------------------------------
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python

# Bazel 0.18.1
wget https://github.com/bazelbuild/bazel/releases/download/0.18.1/bazel-0.18.1-installer-linux-x86_64.sh
chmod +x bazel-0.18.1-installer-linux-x86_64.sh
./bazel-0.18.1-installer-linux-x86_64.sh --user

# Step 4: Set up your environment
# export PATH="$PATH:$HOME/bin" >> ~/.bashrc
# restart

# -------------------------------------------------------
# TensorFlow (Build from source)
# 	https://www.tensorflow.org/install/source
# -------------------------------------------------------

cd ~
git clone --recurse-submodules https://github.com/snuspl/parallax.git
sudo apt-get install python-pip python-dev python-virtualenv

virtualenv parallax_venv
source parallax_venv/bin/activate

# Switch to [cpu_enable] branch
cd ./parallax
git branch -a
git checkout cpu_enable

# Switch to [r1.11] branch
cd ./tensorflow
git branch -a
git checkout r1.11
pip install numpy
./configure

# To handle errors
# https://github.com/swsnu/bd2018/issues/22
pip install enum
pip install enum34
# pip install keras_applications
# pip install keras_preprocessing
pip install mock
pip install keras_applications==1.0.4 --no-deps
pip install keras_preprocessing==1.0.2 --no-deps

# Takes about an hour
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package ./target
pip install ./target/tensorflow-*.whl



# -------------------------------------------------------
# Horovod
# -------------------------------------------------------

cd ../horovod
python setup.py sdist
HOROVOD_WITHOUT_PYTORCH=True pip install --no-cache-dir dist/horovod-*.tar.gz


# -------------------------------------------------------
# Parallax
# -------------------------------------------------------

cd ../parallax
bazel build //parallax/util:build_pip_package


bazel-bin/parallax/util/build_pip_package ./target

pip install ./target/parallax-*.whl


