# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/mariem/Documents/SGI/growing-shapes/build/_deps/libigl-src"
  "/home/mariem/Documents/SGI/growing-shapes/build/_deps/libigl-build"
  "/home/mariem/Documents/SGI/growing-shapes/build/_deps/libigl-subbuild/libigl-populate-prefix"
  "/home/mariem/Documents/SGI/growing-shapes/build/_deps/libigl-subbuild/libigl-populate-prefix/tmp"
  "/home/mariem/Documents/SGI/growing-shapes/build/_deps/libigl-subbuild/libigl-populate-prefix/src/libigl-populate-stamp"
  "/home/mariem/Documents/SGI/growing-shapes/build/_deps/libigl-subbuild/libigl-populate-prefix/src"
  "/home/mariem/Documents/SGI/growing-shapes/build/_deps/libigl-subbuild/libigl-populate-prefix/src/libigl-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/mariem/Documents/SGI/growing-shapes/build/_deps/libigl-subbuild/libigl-populate-prefix/src/libigl-populate-stamp/${subDir}")
endforeach()