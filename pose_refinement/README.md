## Setup
```
# Install the submodule, for spatiotemporal constrained pose refinement, we modified the bundle adjustment in colmap
# The core modification is the additional submodules in /scripts/mvs, and the modification in bundle_adjustment.cc

cd stpr

sudo apt-get install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev

mkdir build
cd build

cmake .. -GNinja
ninja
sudo ninja install
```

## 🚀 Start up

```
cd stpr/scripts/mvs

bash waymo.sh
```

