#! /bin/bash

wget --no-clobber http://apache.cs.utah.edu//commons/io/binaries/commons-io-2.6-bin.zip
if [ ! -d commons-io-2.6 ]
then
  unzip commons-io-2.6-bin.zip
  cp commons-io-2.6/commons-io-2.6.jar commons-io-2.6.jar
fi
