# Step 1. Update and install dependencies
apt-get -qy update
apt-get -qy install git make cmake build-essential libboost-all-dev

# Step 2. Insall python packages
pip3 install protobuf tqdm wheel numpy pandas scipy

# Step 3. Clone repository and build
git clone --branch=stable https://github.com/bigartm/bigartm.git
cd bigartm
mkdir build && cd build
cmake -DPYTHON=python3 ..
make

# Step 4. Install BigARTM
#make install
#export ARTM_SHARED_LIBRARY=/usr/local/lib/libartm.so

# Alternative step 4 - installing only the Python package
pip3 install python/bigartm*.whl
