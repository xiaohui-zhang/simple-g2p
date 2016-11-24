// g2p_io_inl.h

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

#ifndef G2P_IO_INL_H 
#define G2P_IO_INL_H

#include <limits>
#include <vector>
#include <assert.h>

typedef int16_t int16;
typedef int32_t  int32;

// Template that covers integers.
template<class T> void WriteBasicType(std::ostream &os,
                                      bool binary, T t) {
  if (binary) {
    char len_c = (std::numeric_limits<T>::is_signed ? 1 : -1)
        * static_cast<char>(sizeof(t));
    os.put(len_c);
    os.write(reinterpret_cast<const char *>(&t), sizeof(t));
  } else {
    if (sizeof(t) == 1)
      os << static_cast<int16>(t) << " ";
    else
      os << t << " ";
  }
  if (os.fail()) {
    throw std::runtime_error("Write failure in WriteBasicType.");
  }
} 

// Template that covers integers. 
template<class T> inline void ReadBasicType(std::istream &is,
                                            bool binary, T *t) {
  assert(t != NULL);
  if (binary) {
    int len_c_in = is.get();
    if (len_c_in == -1)
      std::cerr << "ReadBasicType: encountered end of stream.";
    char len_c = static_cast<char>(len_c_in), len_c_expected
      = (std::numeric_limits<T>::is_signed ? 1 : -1)
      * static_cast<char>(sizeof(*t)); 
    if (len_c != len_c_expected) {
      std::cerr << "ReadBasicType: did not expected integer type, "
                << static_cast<int>(len_c)
                << " vs. " << static_cast<int>(len_c_expected)
                << ". You can change this code to successfully"
                << " read it later, if needed.";
    }
    is.read(reinterpret_cast<char *>(t), sizeof(*t));
  } else {
    if (sizeof(*t) == 1){
      int16 i;
      is >> i;
      *t = i; 
    } else {
      is >> *t;
    }
  }  
  if (is.fail()){
    std::cerr << "Read failure in ReadBasicTyp, file position is "
              << is.tellg() << ", next char is " << is.peek();
  }
}

// Template that covers integers.
template<class T>
inline void WriteIntegerPairVector(std::ostream &os, bool binary,
                                   const std::vector<std::pair<T, T> > &v) {
  if (binary) {
    char sz = sizeof(T); 
    os.write(&sz, 1);
    int32 vecsz = static_cast<int32>(v.size());
    assert((size_t)vecsz == v.size());
    os.write(reinterpret_cast<const char *>(&vecsz), sizeof(vecsz));
    if (vecsz != 0) {
      os.write(reinterpret_cast<const char *> (&(v[0])), sizeof(T) * vecsz * 2);
    }
  } else {
    os << "[ ";
    typename std::vector<std::pair<T, T> >::const_iterator iter = v.begin(),
                                                            end = v.end();
    for (; iter != end; ++iter) {
      if (sizeof(T) == 1) 
        os << static_cast<int16>(iter->first) << ','
           << static_cast<int16>(iter->second) << ' ';
      else
        os << iter->first << ','
           << iter->second << ' ';
    }
    os << "]\n";
  }
  if (os.fail()){
    throw std::runtime_error("Write failure in WriteIntegerPairVector.");
  }
}

// Template that covers integers.
template<class T>
inline void ReadIntegerPairVector(std::istream &is, bool binary,
                                  std::vector<std::pair<T, T> > *v) {
  assert(v != NULL);
  if (binary) {
    int sz = is.peek();
    if (sz == sizeof(T)) {
      is.get();
    } else {  // this is currently just a check.
      std::cerr << "ReadIntegerPairVector: expected to see type of size "
                << sizeof(T) << ", saw instead " << sz << ", at file position "
                << is.tellg();
    }
    int32 vecsz;
    is.read(reinterpret_cast<char *>(&vecsz), sizeof(vecsz));
    if (is.fail() || vecsz < 0) goto bad;
    v->resize(vecsz);
    if (vecsz > 0) {
      is.read(reinterpret_cast<char *>(&((*v)[0])), sizeof(T)*vecsz*2);
    }
  } else {
    std::vector<std::pair<T, T> > tmp_v;  // use temporary so v doesn't use extra memory
                           // due to resizing.
    is >> std::ws;
    if (is.peek() != static_cast<int>('[')) {
      std::cerr << "ReadIntegerPairVector: expected to see [, saw "
                << is.peek() << ", at file position " << is.tellg();
    }
    is.get();  // consume the '['.
    is >> std::ws;  // consume whitespace.
    while (is.peek() != static_cast<int>(']')) {
      if (sizeof(T) == 1) {  // read/write chars as numbers.
        int16 next_t1, next_t2;
        is >> next_t1;
        if (is.fail()) goto bad;
        if (is.peek() != static_cast<int>(','))
          std::cerr << "ReadIntegerPairVector: expected to see ',', saw "
                    << is.peek() << ", at file position " << is.tellg();
        is.get();  // consume the ','.
        is >> next_t2 >> std::ws;
        if (is.fail()) goto bad;
        else
            tmp_v.push_back(std::make_pair<T, T>((T)next_t1, (T)next_t2));
      } else {
        T next_t1, next_t2;
        is >> next_t1;
        if (is.fail()) goto bad;
        if (is.peek() != static_cast<int>(','))
          std::cerr << "ReadIntegerPairVector: expected to see ',', saw "
                    << is.peek() << ", at file position " << is.tellg();
        is.get();  // consume the ','.
        is >> next_t2 >> std::ws;
        if (is.fail()) goto bad;
        else
            tmp_v.push_back(std::pair<T, T>(next_t1, next_t2));
      }
    }
    is.get();  // get the final ']'.
    *v = tmp_v;  // could use std::swap to use less temporary memory, but this
    // uses less permanent memory.
  }
  if (!is.fail()) return;
 bad:
  std::cerr << "ReadIntegerPairVector: read failure at file position "
            << is.tellg();
}

#endif
