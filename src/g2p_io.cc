// g2p_io.cc

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

#include <assert.h>
#include "g2p_io.h"
#include "g2p_io_inl.h"

template<>
void WriteBasicType<float>(std::ostream &os, bool binary, float f) {
  if (binary) {
    char c = sizeof(f);
    os.put(c);
    os.write(reinterpret_cast<const char *>(&f), sizeof(f));
  } else {
    os << f << " ";
  }
}

template<>
void WriteBasicType<double>(std::ostream &os, bool binary, double f) {
  if (binary) {
    char c = sizeof(f);
    os.put(c);
    os.write(reinterpret_cast<const char *>(&f), sizeof(f));
  } else {
    os << f << " ";
  }
}

template<>
void ReadBasicType<float>(std::istream &is, bool binary, float *f) {
  assert(f != NULL);
  if (binary) {
    double d;
    int c = is.peek();
    if (c == sizeof(*f)) {
      is.get();
      is.read(reinterpret_cast<char*>(f), sizeof(*f));
    } else if (c == sizeof(d)) {
      ReadBasicType(is, binary, &d);
      *f = d;
    } else {
      std::cerr<< "ReadBasicType: expected float, saw " << is.peek()
                << ", at file position " << is.tellg();
    }
  } else {
    is >> *f;
  }
  if (is.fail()) {
    std::cerr << "ReadBasicType: failed to read, at file position "
              << is.tellg();
  }
}

template<>
void ReadBasicType<double>(std::istream &is, bool binary, double *d) {
  assert(d != NULL);
  if (binary) {
    float f;
    int c = is.peek();
    if (c == sizeof(*d)) {
      is.get();
      is.read(reinterpret_cast<char*>(d), sizeof(*d));
    } else if (c == sizeof(f)) {
      ReadBasicType(is, binary, &f);
      *d = f;
    } else {
      std::cerr << "ReadBasicType: expected float, saw " << is.peek()
                << ", at file position " << is.tellg();
    }
  } else {
    is >> *d;
  }
  if (is.fail()) {
    std::cerr << "ReadBasicType: failed to read, at file position "
              << is.tellg();
  }
}

void CheckToken(const char *token) {
  if (*token == '\0')
    std::cerr << "Token is empty (not a valid token)";
  const char *orig_token = token;
  while (*token != '\0') {
    if (::isspace(*token))
      std::cerr << "Token is not a valid token (contains space): '"
                << orig_token << "'";
    token++;
  }
}

void WriteToken(std::ostream &os, bool binary, const char *token) {
  // binary mode is ignored;
  // we use space as termination character in either case.
  assert(token != NULL);
  CheckToken(token);  // make sure it's valid (can be read back)
  os << token << " ";
  if (os.fail()) {
    throw std::runtime_error("Write failure in WriteToken.");
  }
}

void WriteToken(std::ostream &os, bool binary, const std::string & token) {
  WriteToken(os, binary, token.c_str());
}

void ExpectToken(std::istream &is, bool binary, const char *token) {
  int pos_at_start = is.tellg();
  assert(token != NULL);
  CheckToken(token);  // make sure it's valid (can be read back)
  if (!binary) is >> std::ws;  // consume whitespace.
  std::string str;
  is >> str;
  is.get();  // consume the space.
  if (is.fail()) {
    std::cerr << "Failed to read token [started at file position "
              << pos_at_start << "], expected " << token;
  }
  if (strcmp(str.c_str(), token) != 0) {
    std::cerr << "Expected token \"" << token << "\", got instead \""
              << str <<"\".";
  }
}

void ExpectToken(std::istream &is, bool binary, const std::string &token) {
  ExpectToken(is, binary, token.c_str());
}
