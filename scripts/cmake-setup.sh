sudo apt remove cmake
sudo apt purge --auto-remove cmake

version=3.11
build=4
mkdir ~/temp
cd ~/temp
wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz
tar -xzvf cmake-$version.$build.tar.gz
cd cmake-$version.$build/

./bootstrap
make -j4
sudo make install