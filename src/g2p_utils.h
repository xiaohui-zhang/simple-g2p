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

/// History type.
typedef std::vector<std::pair<int32, int32> > HistType;

/// The type of the mapping from grapophnes to probs. 
typedef unordered_map<std::pair<int32, int32>, float, IntPairHasher> Graphone2ProbType;

/// The type of n-gram counts or probabilities is two levels of maps:
/// history->graphone->prob/count
typedef unordered_map<HistType, Graphone2ProbType, IntPairVectorHasher> CountType;


/// For Model I/O

/// Read a pair of ints from istream.
void ReadPair(std::istream &is, bool binary, std::pair<int32, int32>* graphone);

/// Read a history (vector of pair of integers) from stream
void ReadHistory(std::istream &is, bool binary, HistType* hist, int32 order);

/// Read a map from graphone (pair of integers) to prob (float) from istream.
void ReadGraphone2ProbMap(std::istream &is, bool binary, Graphone2ProbType* graphone_map);


/// For Decoding:

/// The basic data structure we use for the A-star decoding when applying G2P, which
/// represents each node in the decoding graph.
struct State {
  int32 pos; // position in the word sequence.
  std::vector<int32> phone_hist; // phone history (the most recent (n-1)-gram)
  State() {}
  State(const int32& pos, const std::vector<int32>& phone_hist):
       pos(pos), phone_hist(phone_hist) {}
  bool operator== (const State &rhs) const {
    return (pos == rhs.pos) && (phone_hist == rhs.phone_hist);
  }
  struct Hasher {
    size_t operator() (const State &s) const {
      size_t ans = s.pos;
      typename std::vector<int32>::const_iterator iter = s.phone_hist.begin(), end = s.phone_hist.end();
      for (; iter != end; ++iter) {
        ans *= 7853;
        ans += *iter;
      }
      return ans;
    }
  };
};

/// A wrapper of "State", which contains the cost associated with that state.
/// In our algorithm we make sure there is a 1-1 mapping between States and Nodes.
struct Node {
  State state;
  float fcost; // forward cost, which is the cost of the path from the start 
  // node to the current node.
  float bcost; // backward cost, which is the heuristic that estimates the
  // cost of the cheapest path from n to the goal.
  Node() {}
  Node(const State& state, const float& fcost, const float& bcost):
       state(state), fcost(fcost), bcost(bcost) {}
  bool operator== (const Node &rhs) const {
    return (state == rhs.state) && (fcost + bcost == rhs.fcost + rhs.bcost);
  }
};

/// Comparator function for the ForwardQueueType.
struct CompareNodesWithCost {
  bool operator() (const std::pair<float, Node*>& a, const std::pair<float, Node*>& b) {
    return a.first < b.first;
  }
};

/// The queue used for forward (1st pass) A-star decoding. 
/// The reason why we store the cost as a seperate field outside Node is that,
/// the cost within each Node could be updated (we later found a better path to 
/// this Node), and we want to keep track of the cost (serving as the priority)
/// when the Node was en-queued.
typedef std::priority_queue<std::pair<float, Node*>, std::vector<std::pair<float, Node*> >, CompareNodesWithCost> ForwardQueueType;

/// For each Node/State, we store the ID of its source node, the graphone
/// and cost on the arc emitted from the source node in SourceInfo. 
struct SourceInfo {
  int32 node_id;
  std::pair<int32, int32> graphone;
  float arc_cost;
  SourceInfo() {}
  SourceInfo(const int32& node_id, const std::pair<int32, int32>& graphone, float arc_cost):
            node_id(node_id), graphone(graphone), arc_cost(arc_cost) {}
};

/// A mapping from the current State to the best cost we encountered so far
/// from the start to here, and the last source Node in the best path.
typedef unordered_map<State, std::pair<Node*, SourceInfo>, State::Hasher> BestCostType;

/// The graph we use to back-trace the decoding graph in the second pass A-star search,
/// which stores SourceInfo for all source Nodes of each State.
typedef unordered_map<State, std::vector<SourceInfo>, State::Hasher> BackTraceGraph;

/// During the second pass A-star search (back-tracing), we store the partial
/// decoding results (phone_seq) when we visit each node (from different paths).
struct Hyp {
  Node* node;
  float fcost; /// We only need to store the forward cost during back-tracing,
  /// because the (perfect) backward cost (rest cost) can be found as node->fcost.
  std::vector<int32> phone_seq;
  Hyp() {}
  Hyp(const float& fcost, const std::vector<int32>& phone_seq, Node* node):
      node(node), fcost(fcost), phone_seq(phone_seq) {}
};

/// Comparator of Hyp (by the priority fcost+bcost. Note that the "fcost" stored
/// in the node serves as backward cost (rest cost) here during back-tracing).
struct CompareHyp {
  bool operator() (const Hyp& a, const Hyp& b) {
    return a.fcost + a.node->fcost < b.fcost + b.node->fcost;
  }
};

/// The queue we use in the second pass A-star search (back-tracing). 
typedef std::priority_queue<Hyp, std::vector<Hyp>, CompareHyp> BackTraceQueueType;

/// Comparator of the final decoding results (compare by the posteriors).
struct CompareResult {
  bool operator() (const std::pair<std::vector<int32>, float>& a, 
                   const std::pair<std::vector<int32>, float>& b) {
    return a.second > b.second;
  }
};

#endif
