// g2p_utils.h

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

#ifndef G2P_UTILS_H_
#define G2P_UTILS_H_

#ifdef _MSC_VER
#include <unordered_map>
#include <unordered_set>
using std::unordered_map;
using std::unordered_set;
#elif __cplusplus > 199711L || defined(__GXX_EXPERIMENTAL_CXX0X__)
#include <unordered_map>
#include <unordered_set>
using std::unordered_map;
using std::unordered_set;
#else
#include <tr1/unordered_map>
#include <tr1/unordered_set>
using std::tr1::unordered_map;
using std::tr1::unordered_set;
#endif

#include <cassert>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>
#include <queue>
#include <limits>

typedef int32_t  int32;


#ifndef FLT_EPSILON
#define FLT_EPSILON 1.19209290e-7f
#endif

#if !defined(_MSC_VER) || (_MSC_VER >= 1900)
inline float Exp(float x) { return expf(x); }
#else
#if !defined(__INTEL_COMPILER) && _MSC_VER == 1800 && defined(_M_X64)
// Microsoft CL v18.0 buggy 64-bit implementation of
// expf() incorrectly returns -inf for exp(-inf).
inline float Exp(float x) { return exp(static_cast<float>(x)); }
#else
inline float Exp(float x) { return expf(x); }
#endif  // !defined(__INTEL_COMPILER) && _MSC_VER == 1800 && defined(_M_X64)
#endif  // !defined(_MSC_VER) || (_MSC_VER >= 1900)
inline float Log(float x) { return logf(x); }

#if !defined(_MSC_VER) || (_MSC_VER >= 1700)
inline float Log1p(float x) {  return log1pf(x); }
#else
inline float Log1p(float x) {
  const float cutoff = 1.0e-07;
  if (x < cutoff)
    return x - 2 * x * x;
  else
    return Log(1.0 + x);
}
#endif

static const float kMinLogDiffFloat = Log(FLT_EPSILON);  // negative!

inline float LogAdd(float x, float y) {
  float diff;
  if (x < y) {
    diff = x - y;
    x = y;
  } else {
    diff = y - x;
  }
  // diff is negative.  x is now the larger one.

  if (diff >= kMinLogDiffFloat) {
    float res;
    res = x + Log1p(Exp(diff));
    return res;
  } else {
    return x;  // return the larger one.
  }
}

/// A hashing function-object for vectors of ints.
struct IntVectorHasher {  // hashing function for vector<Int>.
  size_t operator()(const std::vector<int32> &x) const {
    size_t ans = 0;
    typename std::vector<int32>::const_iterator iter = x.begin(), end = x.end();
    for (; iter != end; ++iter) {
      ans *= kPrime;
      ans += *iter;
    }
    return ans;
  }
 private:
  static const int kPrime = 7853;
};

/// A hashing function-object for vectors of pairs of ints
struct IntPairVectorHasher {  // hashing function for pair<int>
  size_t operator()(const std::vector<std::pair<int32, int32> > &x) const {
    size_t ans = 0;
    typename std::vector<std::pair<int32, int32> >::const_iterator iter =
        x.begin(), end = x.end();
    for (; iter != end; ++iter) {
      ans *= kPrime1;
      ans += iter->first + kPrime2 * iter->second;
    }
    return ans;
  }
 private:
  static const int kPrime1 = 7853;
  static const int kPrime2 = 1979;
};

/// A hashing function-object for pairs of ints
struct IntPairHasher {  // hashing function for pair<int>
  size_t operator()(const std::pair<int32, int32> &x) const {
    return x.first + x.second * kPrime;
  }
 private:
  static const int kPrime = 7853;
};

/// During the A-star decoding when applying G2P, the data structure for
/// representing each node in the decoding graph is like this.
struct Node {
  int32 pos; // position in the word sequence.
  std::vector<int32> phones; // partial decoding results
  float cost; // cost = log-prob + distance-to-end
  Node() {}
  Node(const int32& pos, const std::vector<int32>& phones, const float& cost):
       pos(pos), phones(phones), cost(cost) {}
};

struct CompareNode {
  bool operator() (const Node& a, const Node& b) {
    return a.cost < b.cost;
  }
};

struct CompareResult {
  bool operator() (const std::pair<std::vector<int32>, float>& a, 
                   const std::pair<std::vector<int32>, float>& b) {
    return a.second > b.second;
  }
};

/// History type.
typedef std::vector<std::pair<int32, int32> > HistType;

/// The type of the mapping from grapophnes to probs. 
typedef unordered_map<std::pair<int32, int32>, float, IntPairHasher> Graphone2ProbType;

/// The type of n-gram counts or probabilities is two levels of maps:
/// history->graphone->prob/count
typedef unordered_map<HistType, Graphone2ProbType, IntPairVectorHasher> CountType;

/// The queue used A-star decoding.
typedef std::priority_queue<Node, std::vector<Node>, CompareNode> QueueType;

/// A mapping from partial decoding results (position-in-word-sequence, decoded-phones)
/// to the best cost encountered so far.
typedef unordered_map<int32, unordered_map<std::vector<int32>, 
                      float, IntVectorHasher> > BestCostType;

// Read a pair of ints from istream.
void ReadPair(std::istream &is, bool binary, std::pair<int32, int32>* graphone);

// Read a history (vector of pair of integers) from stream
void ReadHistory(std::istream &is, bool binary, HistType* hist, int32 order);

// Read a map from graphone (pair of integers) to prob (float) from istream.
void ReadGraphone2ProbMap(std::istream &is, bool binary, Graphone2ProbType* graphone_map);

#endif
