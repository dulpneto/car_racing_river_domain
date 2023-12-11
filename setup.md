python 3.7
https://linuxize.com/post/how-to-install-python-3-7-on-ubuntu-18-04/#installing-python-37-on-ubuntu-from-source

sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev wget libbz2-dev

cd /tmp 

wget https://www.python.org/ftp/python/3.7.4/Python-3.7.4.tgz

tar -xf Python-3.7.4.tgz

./configure --enable-optimizations

make -j 8
For faster build time, modify the -j flag according to your processor. If you do not know the number of cores in your processor, you can find it by typing nproc. The system used in this guide has 8 cores, so we are using the -j8 flag.


make install


#pip

sudo apt install python3-pip


# deps for cv2
sudo apt-get install ffmpeg libsm6 libxext6

# deps for gym box2d
sudo apt-get install swig

#deps
pip3 install gymnasium
pip3 install gymnasium[box2d]
pip3 install tensorflow==2.9.2
pip3 install opencv-contrib-python


