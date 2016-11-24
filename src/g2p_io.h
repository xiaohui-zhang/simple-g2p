// g2p_io.h

// Copyright     2016  Johns Hopkins University (Author: Daniel Povey)
//               2016  Xiaohui Zhang
//               2016  Ke Li

// See ../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABILITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef G2P_IO_H
#define G2P_IO_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include "g2p_io_inl.h"


/// WriteBasicType is the name of the write function for integer types and 
/// floating-point types. They all throw on error.
template<class T> void WriteBasicType(std::ostream &os, bool binary, T t);

/// ReadBasicType is the name of the read function for integer types and
/// floating-point types. They all throw on error.
template<class T> void ReadBasicType(std::istream &is, bool binary, T *t);

// Declare specialization for float. 
template<>
void WriteBasicType<float>(std::ostream &os, bool binary, float f);

template<>
void WriteBasicType<double>(std::ostream &os, bool binary, double f);

template<>
void ReadBasicType<float>(std::istream &is, bool binary, float *f);

template<>
void ReadBasicType<double>(std::istream &is, bool binary, double *f);

/// Function for writing STL vectors of integer types.
template<class T> inline void WriteIntegerPairVector(std::ostream &os, bool binary,
                                                     const std::vector<std::pair<T, T> > &v);

template<class T> inline void ReadIntegerPairVector(std::istream &is, bool binary,
                                                     const std::vector<std::pair<T, T> > *v);

/// The WriteToken functions are for writing nonempty sequences of non-space
/// characters. They are not for general strings. 
void WriteToken(std::ostream &os, bool binary, const char *token);
void WriteToken(std::ostream &os, bool binary, const std::string &token);

/// ExpectToken tries to read in the given token, and throws an exception
/// on failure. 
void ExpectToken(std::istream &is, bool binary, const char *token);
void ExpectToken(std::istream &is, bool binary, const std::string &token);

#endif
