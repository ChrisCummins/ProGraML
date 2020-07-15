# macOS

## step 1. Install homebrew:
```
test -f /usr/local/bin/brew || yes | /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

## step 2.
```
brew install bazel
brew install coreutils
brew install exempi
brew install findutils
brew install gnu-indent
brew install gnu-sed
brew install gnu-tar
brew install gnu-time
brew install gnu-which
brew install libomp
brew install python
brew install cmake
brew install boost
brew install sdl2
brew install wget
brew install ffmpeg
```

# Linux env

## step 1. Install bazel:
```
curl -L -o /tmp/bazel.sh https://github.com/bazelbuild/bazel/releases/download/2.0.0/bazel-2.0.0-installer-linux-x86_64.sh && sudo bash /tmp/bazel.sh && rm /tmp/bazel.sh
```

## step 2. Ubuntu only
```
sudo apt-get update  # Update package index:
sudo apt install -y --no-install-recommends ca-certificates
sudo apt install -y --no-install-recommends curl
sudo apt install -y --no-install-recommends wget
sudo apt install -y --no-install-recommends g++
sudo apt install -y --no-install-recommends git
sudo apt install -y --no-install-recommends ocl-icd-opencl-dev  # Needed for -lOpenCL
sudo apt install -y --no-install-recommends opencl-c-headers  # Needed for #include <CL/cl.h>
sudo apt install -y --no-install-recommends pkg-config
sudo apt install -y --no-install-recommends python
sudo apt install -y --no-install-recommends python-dev  # Python headers needed for pybind
sudo apt install -y --no-install-recommends python3.6
sudo apt install -y --no-install-recommends python3.6-dev
sudo apt install -y --no-install-recommends python3-distutils
sudo apt install -y --no-install-recommends unzip
sudo apt install -y --no-install-recommends zip
sudo apt install -y --no-install-recommends zlib1g-dev
sudo apt install -y --no-install-recommends openjdk-11-jdk
sudo apt install -y --no-install-recommends m4
sudo apt install -y --no-install-recommends libexempi-dev
sudo apt install -y --no-install-recommends rsync
sudo apt install -y --no-install-recommends python3-numpy
sudo apt install -y --no-install-recommends build-essential
sudo apt install -y --no-install-recommends libsdl2-dev
sudo apt install -y --no-install-recommends libjpeg-dev
sudo apt install -y --no-install-recommends nasm
sudo apt install -y --no-install-recommends tar
sudo apt install -y --no-install-recommends libbz2-dev
sudo apt install -y --no-install-recommends libgtk2.0-dev
sudo apt install -y --no-install-recommends cmake
sudo apt install -y --no-install-recommends libfluidsynth-dev
sudo apt install -y --no-install-recommends libgme-dev
sudo apt install -y --no-install-recommends libopenal-dev
sudo apt install -y --no-install-recommends timidity
sudo apt install -y --no-install-recommends libwildmidi-dev
sudo apt install -y --no-install-recommends libboost-all-dev
sudo apt install -y --no-install-recommends libsdl2-dev
sudo apt install -y --no-install-recommends patch # Required by bazel workspace rules:
sudo apt-get install -y --no-install-recommends libmysqlclient-dev 
sudo apt-get install -y --no-install-recommends libtinfo-dev 
sudo apt-get install -y --no-install-recommends python3-tk 
```
in one line: 

```bash
sudo apt-get update && sudo apt install -y --no-install-recommends ca-certificates curl wget g++ git ocl-icd-opencl-dev opencl-c-headers  pkg-config python3.6 python3.6-dev python3-pip python3-distutils unzip zip zlib1g-dev openjdk-11-jdk m4 libexempi-dev rsync python3-numpy build-essential libsdl2-dev libjpeg-dev nasm tar libbz2-dev libgtk2.0-dev cmake libfluidsynth-dev libgme-dev libopenal-dev timidity libwildmidi-dev libboost-all-dev libsdl2-dev patch libmysqlclient-dev libtinfo-dev
```

## step 3. python packages manually installed
```
python -m pip install 'pybind11==2.4.3' # Required for building pyopencl:
python3 -m pip install 'setuptools==49.2.0' # Required for installing tensorflow
python3 -m pip install 'tensorflow==1.14.0'
```
in one line
```bash
python3 -m pip install 'pybind11==2.4.3'  'setuptools==49.2.0' && python3 -m pip install 'tensorflow==1.14.0'
```
