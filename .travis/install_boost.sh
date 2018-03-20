#!/bin/bash

export BOOST_ROOT=${HOME}/${BOOST_BASENAME}
if [ "$TRAVIS_OS_NAME" = "osx" ]; then
    export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${BOOST_ROOT}/stage/lib
fi

if [ ! -e ${BOOST_ROOT}/boost/config.hpp ]; then
    pushd ${HOME}
    wget https://downloads.sourceforge.net/project/boost/boost/1.66.0/${BOOST_BASENAME}.tar.bz2
    rm -rf $BOOST_BASENAME
    tar xf ${BOOST_BASENAME}.tar.bz2
    (cd ${BOOST_BASENAME} && ./bootstrap.sh --with-libraries=chrono,date_time,filesystem,program_options,system,thread,test && ./b2)
    popd
fi

