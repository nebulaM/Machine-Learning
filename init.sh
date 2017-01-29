#!/bin/bash
#required python environment, run as root
#a0
sudo apt-get install -y libfreetype6-dev python-pip
sudo apt-get install -y libblas-dev
sudo apt-get install -y liblapack-dev
sudo apt-get install -y gfortran
sudo pip install numpy scipy
#a1
sudo pip install utils
sudo apt-get install -y python-matplotlib
sudo pip install scikit-learn
