# install dependencies
brew install libtool
brew install swig
 
# checkout and patch
# see http://sourceforge.net/p/ghmm/patches/15/
svn checkout svn://svn.code.sf.net/p/ghmm/code/trunk/ghmm ghmm
cd ghmm/ghmm
wget http://sourceforge.net/p/ghmm/patches/15/attachment/block_compression.patch
patch block_compression.c block_compression.patch
 
# configure and build
# make sure configure with --without-python option
cd ..
./autogen.sh
./configure --prefix=$HOME --without-python
./make
./make install
 
# # build python interface
# cd ghmmwrapper
# vim setup.py ghmmhelper.py
# # insert the following to line 2 of both files
# # -*- coding:utf8 -*-
# python setup.py build
# sudo python setup.py install