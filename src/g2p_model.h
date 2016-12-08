// g2p_model.h

// Copyright     2016  Xiaohui Zhang
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

#ifndef G2P_MODEL_H_
#define G2P_MODEL_H_

#include "g2p_utils.h"
#include <thread>
#include <mutex>

#include <sys/time.h>
#include <unistd.h>

class Timer {
 public:
  Timer() { Reset(); }

  void Reset() { gettimeofday(&this->time_start_, &time_zone_); }

  /// Returns time in seconds.
  double Elapsed() {
    struct timeval time_end;
    gettimeofday(&time_end, &time_zone_);
    double t1, t2;
    t1 =  static_cast<double>(time_start_.tv_sec) +
          static_cast<double>(time_start_.tv_usec)/(1000*1000);
    t2 =  static_cast<double>(time_end.tv_sec) +
          static_cast<double>(time_end.tv_usec)/(1000*1000);
    return t2-t1;
  }

 private:
  struct timeval time_start_;
  struct timezone time_zone_;
};


class G2PModel {
 public:
  // Constructor for test.
  G2PModel(int32 ngram_order, int32 num_graphemes, int32 num_phonemes);
  
  G2PModel(int32 ngram_order, float discounting_constant_min, float discounting_constant_max,
           int32 num_graphemes, int32 num_phonemes, char* train_words_file, char* train_prons_file, char* valid_words_file, char* valid_prons_file);
  
  /// Train the G2P model, assuming we already read in training data into
  /// words_ and prons_ in the constructor.
  void Train(int32 num_threads);

  /// Test the G2P model, given a list of test words, and we output the
  /// predicted prons to the file "output".
  void Test(char* test_words_file, char* output, const int32& num_variants, const int32& k);
 
  /// Decode a word. Return the 10-best phone-sequences and normalized posteriors. 
  /// It's basically A-star search, with the cost as the the log-prob of the
  /// current decoding results (represented in each Node).
  void Decode(const std::vector<int32>& word,
              std::vector<std::pair<std::vector<int32>, float> >* results);

  // Write the whole model into a file.
  void Write(char* file, bool binary) const;

  // Read the whole model from a file.
  void Read(char* file, bool binary);

 private:
  /// Read sequences of integers (must be separated by spaces) from a file. 
  void ReadSequences(int32 vocab_size, char* file, std::vector<std::vector<int32> >* sequences);
  
  /// Get the probability of a graphone given a requested n-gram order and a history. 
  float GetProb(const int32& order, // ngram order. 0 == unigram
                const HistType& h, // history
                const std::pair<int32, int32>& g); // graphone

  /// Do forward-backward computation, accumulating n-gram counts, given a pair
  /// of word and pronuncation, and a specified n-gram order (length of histories).
  float ForwardBackward(const std::vector<int>& word,
                       const std::vector<int>& pron,
                       const int32& order,
                       const bool& skip_backward = false,
                       CountType* counts = NULL);

  static void ForwardBackwardStatic(G2PModel *mdl,
                                    const std::vector<std::vector<int> >& words,
                                    const std::vector<std::vector<int> >& prons,
                                    const int32& order,
                                    CountType* counts,
                                    float *log_like) {
                                    assert(words.size() == prons.size());
                                      for (int32 i = 0; i < words.size(); i++) {
                                       // std::cout << "entering ForwardBackward" << std::endl;
                                        *log_like += mdl->ForwardBackward(words[i], prons[i], order, false, counts);
                                      //  std::cout << "finishing ForwardBackward" << std::endl;
                                     }
                                   }
  
  /// Add count obtained in forward-backward computation into counts_.
  void AddCount(const HistType& h, // history
                const std::pair<int32, int32>& g, // graphone
                const float& value, CountType *counts);

  void MergeCount(CountType& counts, const int32& order);
  
  
  /// Update the model represented by prob_, using the accumulated n-gram counts.
  /// The standard Kneyser-Ney smooting is used. 
  void UpdateProb(const int32 order);

  /// Compute the bound_matrix_ from the model represented by prob_.
  /// See comments near the declaration of bound_matrix_.
  void ComputeBoundMatrix();
  
  /// Enqueue a Node into the queue in A-star search.
  void Enqueue(const Node& node, const std::pair<int32, int32>& g, // graphone
               const HistType& h, // history
               const std::vector<float>& heuristics, // the vector of heuristics.
               ForwardQueueType* q, // 
               std::vector<int32>* num_active,
               std::vector<float>* beam_width);

  /// Write prob_ into stream.
  void WriteProb(std::ostream &os, bool binary) const;

  /// Read prob_ from stream.
  void ReadProb(std::istream &is, bool binary, std::vector<CountType>* prob);


  // For debugging
  void TestWriteRead(char* file, bool binary);
  void PrintWrite();
  void PrintRead(std::vector<CountType>* prob);

  /// void PrintResults(const std::vector<int32> &v);
  
  /// N-gram order of the model. 
  int32 ngram_order_;
  /// Begin-of-sentence symbol.
  int32 bos_;
  /// End-of-sentence symbol.
  int32 eos_;
  /// Epsilon symbol.
  int32 eps_;
  /// Back-off symbol (we define it as a pair because the key of the map from
  /// graphones to probs must be pairs).
  std::pair<int32, int32> backoff_symbol_;
  /// Grapheme ids must be contiguous integers from 0 to num_graphemes_ - 1.
  int32 num_graphemes_;
  /// Phoneme ids must be contiguous integers from 0 to num_phonemes_ - 1.
  int32 num_phonemes_;
  /// Discounting constant in Kneser-Ney smooting. Must be >=0 and <1.
  std::vector<float> discounting_constants_;
  /// N-gram counts.
  std::vector<CountType> counts_;
  /// N-gram probabilities.
  std::vector<CountType> prob_;
  /// Training words.
  std::vector<std::vector<int32> > train_words_;
  /// Training pronunciations.
  std::vector<std::vector<int32> > train_prons_;
  /// Valid words.
  std::vector<std::vector<int32> > valid_words_;
  /// Valid pronunciations.
  std::vector<std::vector<int32> > valid_prons_;
  
  /// All valid graphones (a vector of pairs of integers).
  HistType graphones_;
  /// The look-up table which stores, for each letter l2 given another letter l1 as
  /// the bigram history, the probability of the most probable (letter, phone)
  /// pair given that letter history for *any* phone and phone history.
  /// It's a two level of maps: letter->letter->prob, representing a matrix
  /// p_max(l2|l1). This'll be used to derive the heuristics during A-star search.
  unordered_map<int32, unordered_map<int32, float> > bound_matrix_;
  
  int32 max_num_active_nodes_;

  /// Contains all visited nodes (on top of which we expanded hypothesis).
  /// The indexes are used as "node_id" in the Decoder code.
  std::vector<Node*> nodes_visited_;

  /// The set of all nodes we created during decoding.
  unordered_set<Node*> nodes_all_;
  
  /// The queue we use for the second path A-star search. See g2p_utils.h.
  BackTraceGraph graph_;
  
  /// See "BestCostType" in g2p_utils.h
  BestCostType best_cost_;

  /// number of pronunciation variants per word.
  int32 num_variants_;

};

#endif
