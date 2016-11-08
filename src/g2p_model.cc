// g2p_model.cc

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


#include "g2p_model.h"

// Constructor for test
G2PModel::G2PModel(int32 ngram_order, int32 num_graphemes, int32 num_phonemes) {
  ngram_order_ = ngram_order;
  bos_ = -3;
  eos_ = -2;
  eps_ = 0; // following the OpenFst convention.
  num_graphemes_ = num_graphemes;
  num_phonemes_ = num_phonemes;

  for (int32 i = 1; i <= num_graphemes_; i++) {
    for (int32 j = 1; j <= num_phonemes_; j++) {
      graphones_.push_back(std::pair<int32, int32>(i, j));
    }
    graphones_.push_back(std::pair<int32, int32>(i, eps_));
  }
  for (int32 j = 1; j <= num_phonemes_; j++) {
    graphones_.push_back(std::pair<int32, int32>(eps_, j));
  }
  graphones_.push_back(std::pair<int32, int32>(eos_, eos_));
  prob_.resize(ngram_order);
  counts_.resize(ngram_order);
}

G2PModel::G2PModel(int32 ngram_order, float discounting_constant_min, 
                   float discounting_constant_max, int32 num_graphemes, int32 num_phonemes,
                   char* words_file, char* prons_file) {
  discounting_constants_.resize(ngram_order);
  for (int32 i = 0; i < ngram_order; i++) {
    discounting_constants_[i] = discounting_constant_min + (discounting_constant_max
      - discounting_constant_min) / static_cast<float>(ngram_order-1) * i;
    std::cout << "discounting_constant for order " << i << " is " << discounting_constants_[i] << std::endl;
  }
  ngram_order_ = ngram_order;
  bos_ = -3;
  eos_ = -2;
  eps_ = 0; // following the OpenFst convention.
  num_graphemes_ = num_graphemes;
  num_phonemes_ = num_phonemes;
  ReadSequences(num_graphemes_, words_file, &words_);
  ReadSequences(num_phonemes_, prons_file, &prons_);

  assert(words_.size() == prons_.size());
  std::cout << "Read " << words_.size() << " word-pronunciation pairs."<< std::endl;
  for (int32 i = 1; i <= num_graphemes_; i++) {
    for (int32 j = 1; j <= num_phonemes_; j++) {
      graphones_.push_back(std::pair<int32, int32>(i, j));
    }
    graphones_.push_back(std::pair<int32, int32>(i, eps_));
  }
  for (int32 j = 1; j <= num_phonemes_; j++) {
    graphones_.push_back(std::pair<int32, int32>(eps_, j));
  }
  graphones_.push_back(std::pair<int32, int32>(eos_, eos_));
  prob_.resize(ngram_order);
  counts_.resize(ngram_order);
}

/// Train the G2P model, assuming we already read in training data into
/// words_ and prons_ in the constructor.
void G2PModel::Train(int32 num_threads) {
  std::vector<std::vector<std::vector<int32> > > word_subsets(num_threads);
  std::vector<std::vector<std::vector<int32> > > pron_subsets(num_threads);
  int32 block_size = words_.size() / num_threads;
  if (block_size * num_threads < words_.size()) block_size += 1;
  std::cout << "running with " << num_threads << 
    " threads , # training examples for each thread is " << block_size << std::endl; 
  for(int32 i = 0; i < num_threads; i++) {
    for(int32 j = i*block_size; j < (i+1)*block_size; j++) {
      if (j >= words_.size()) break;
      word_subsets[i].push_back(words_[j]);
      pron_subsets[i].push_back(prons_[j]);
    }
  }
  for (int32 i = 0; i < graphones_.size(); i++) {
    std::pair<int32, int32> g(graphones_[i]);
    HistType h0;
    prob_[0][h0][g] = 1.0f / static_cast<float>(graphones_.size());
  }
  for (int32 o = 0; o < ngram_order_; o++) {
    float log_like_last = -1.0;
    while (true) {
      float log_like_tot = 0.0;
      std::vector<std::thread> threads(num_threads);
      std::vector<float> log_like(num_threads);
      std::vector<CountType> counts(num_threads);
      counts_[o].clear();
      for(int32 i = 0; i < num_threads; i++) {
        // std::cout << word_subsets[i].size() << std::endl;
        threads[i] = std::thread(ForwardBackwardStatic, this, word_subsets[i], pron_subsets[i], o, &(counts[i]), &(log_like[i]));
      }
      for (int32 i = 0; i < num_threads; i++) {
        threads[i].join();
        log_like_tot += log_like[i];
        MergeCount(counts[i], o);
      }
    // Baseline for debugging (no multi-threading)
    // CountType counts;
    // for(int32 i = 0; i < words_.size(); i++) {
    //   log_like_tot += ForwardBackward(words_[i], prons_[i], o, false, &counts);
    // }
    // MergeCount(counts, o);
      log_like_tot /= static_cast<float>(words_.size());
      std::cout << "order " << o << " log like is " << log_like_tot << std::endl;
      // The stopping condition relies on the change in log-likelihood.
      if (fabs(log_like_tot - log_like_last) < 0.1) break;
      log_like_last = log_like_tot;
      UpdateProb(o);
    }
  }
}

/// Test the G2P model, given a list of test words, and we output the
/// predicted prons to the file "output".
void G2PModel::Test(char* test_words_file, char* output,
                    const int32& num_variants, const int32& max_num_active_nodes) {
  std::vector<std::vector<int32> > test_words;
  ReadSequences(num_graphemes_, test_words_file, &test_words);
  ComputeBoundMatrix();
  std::ofstream os(output);
  max_num_active_nodes_ = max_num_active_nodes;
  num_variants_ = num_variants;
  for (int32 i = 0; i < test_words.size(); i++) {
    std::vector<std::pair<std::vector<int32>, float> > results;
    Decode(test_words[i], &results);
    for (int32 j = 0; j < num_variants_; j++) {
      for (int32 k = 1; k < test_words[i].size()-1; k++) {
        os << test_words[i][k] << " ";
      }
      os << "\t" << results[j].second << "\t";
      for (int32 k = 1; k < results[j].first.size()-1; k++) { // we don't print eos_, bos_.
        os << results[j].first[k] << " ";
       // std::cout << results[j].first[k] << " ";
      }
      os << std::endl;
      // std::cout << std::endl;
    }
  }
}

/// Decode a word. Return the 10-best phone-sequences and normalized posteriors. 
/// It's basically A-star search, with the cost as the the log-prob of the
/// current decoding results (represented in each Node).
void G2PModel::Decode(const std::vector<int32>& word,
                      std::vector<std::pair<std::vector<int32>, float> >* results) { 

  // The vector of heuristics used in A-star search. It's the bound on
  // the cost from each position in the word sequence to the end.
  std::vector<float> heuristics;
  // The number of active Nodes (in the queue) at each position in the word
  // sequence. We want to limit this number by 1000.
  std::vector<int32> num_active(word.size(), 0);
  std::vector<float> beam_width(word.size(), -std::numeric_limits<float>::infinity());
  float sum_heuristics = 0.0f;
  for (int32 i = 0; i < word.size()-1; i++) {
    float logp = Log(bound_matrix_[word[i]][word[i+1]]);
    sum_heuristics += logp;
    heuristics.push_back(logp);
  }
  heuristics.push_back(0.0f);

  // Construct the start node and store related information.
  std::vector<int32> phone_hist0;
  phone_hist0.push_back(bos_);
  State state0(0, phone_hist0);
  Node *node0 = new Node(state0, 0.0f, sum_heuristics);
  nodes_all_.insert(node0);
  
  // We denote the source node_id of the start node as -1.
  SourceInfo source_info0(-1, std::pair<int32, int32>(bos_, bos_), 0.0f);
  best_cost_[state0] = std::pair<Node*, SourceInfo>(node0, source_info0);
  graph_[state0] = std::vector<SourceInfo>();

  ForwardQueueType q;
  q.push(std::pair<float, Node*>(sum_heuristics, node0));
  num_active[0] += 1;

  int32 counts = 0;
  float prob_mass_tot = 0.0f;
  
  BackTraceQueueType q2;

  while (!q.empty()) {
    counts += 1;

    std::pair<float, Node*> node_with_cost = q.top();
    q.pop();
    auto node = node_with_cost.second;
    num_active[node->state.pos] -= 1;
    if (node_with_cost.first != node->fcost + node->bcost) continue;
    // We have updated the cost of the same node in a former en-queue operation.
    // So we are skipping visiting this node for now to avoid duplicated visits.
    if (node->state.phone_hist[node->state.phone_hist.size()-1] == eos_) 
      assert(node->state.pos == word.size()-1);
    if (node->state.pos == word.size()-1) {
      assert(node->state.phone_hist[node->state.phone_hist.size()-1] == eos_);
      q2.push(Hyp(0.0,  std::vector<int32>(), node));
      if (q2.size() == num_variants_)
        break;
      continue;
   
   /// One-best result generation. For debugging.
   //   auto node_next_id = best_cost_[node->state].second.node_id;
   //   std::vector<int32> result;      
   //   result.push_back(best_cost_[node->state].second.graphone.second);
   //   int32 c = 0; 
   //   while (node_next_id != -1) {
   //     c += 1;
   //     auto node_next = nodes_visited_[node_next_id];
   //     int32 pos1 = node_next->state.pos; 
   //     SourceInfo source_info = best_cost_[node_next->state].second;
   //     node_next_id = source_info.node_id;
   //     int32 phoneme = source_info.graphone.second;
   //     if (phoneme != eps_) {
   //       result.insert(result.begin(), phoneme);
   //     }
   //   }
   //   for (auto x: nodes_all_) {
   //     delete x;
   //   }
   //   results->push_back(std::pair<std::vector<int32>, float>(result, 1.0));
   //   nodes_visited_.clear();
   //   nodes_all_.clear();
   //   std::cout << "number of de-queue operations during decoding: " << counts << std::endl;
   //   return;

    }

    int32 hist_len = std::min(ngram_order_-1, std::min(static_cast<int32>(node->state.phone_hist.size()), node->state.pos+1));
    // Construct the history vector.
    HistType h;
    if ((hist_len) > 0) {
      for (int32 i = 0; i < hist_len; i++) {
        assert(node->state.pos-i < word.size());
        assert(node->state.pos-i >= 0);
        h.insert(h.begin(), std::pair<int32, int32>(word[node->state.pos-i], node->state.phone_hist[node->state.phone_hist.size()-1-i]));
      }
    }
    if (std::find_if(nodes_visited_.begin(), nodes_visited_.end(), [node](Node* n){return n == node;}) != nodes_visited_.end()) continue;
    
    nodes_visited_.push_back(node);
    // if (node.cost < best_cost_[node.pos][ngram]) continue;
    if (node->state.pos == word.size()-2) {
      std::pair<int32, int32> g(eos_, eos_);
      Enqueue(*node, g, h, heuristics, &q, &num_active, &beam_width);
    } else {
      std::pair<int32, int32> g(word[node->state.pos+1], eps_);
      Enqueue(*node, g, h, heuristics, &q, &num_active, &beam_width);
      for (int32 p = 1; p <= num_phonemes_; p++) {
        std::pair<int32, int32> g(word[node->state.pos+1], p);
        Enqueue(*node, g, h, heuristics, &q, &num_active, &beam_width);
      }
    }
    for (int32 p = 1; p <= num_phonemes_; p++) {
      std::pair<int32, int32> g(eps_, p);
      Enqueue(*node, g, h, heuristics, &q, &num_active, &beam_width);
    }
  }
  
  while (!q2.empty()) {
    counts += 1;
    Hyp hyp = q2.top();
    q2.pop();
    auto state = hyp.node->state;
    if (graph_[state].size() == 0) {
      assert(best_cost_[state].second.node_id == -1 && state.pos == 0 && best_cost_[state].second.graphone.second == bos_);
      std::vector<int32> result(hyp.phone_seq);
      result.insert(result.begin(), bos_);
      if (std::find_if(results->begin(), results->end(), [result](std::pair<std::vector<int32>, float> r){return IntVectorHasher()(r.first) == IntVectorHasher()(result);}) != results->end()) continue;
      // Rescore the decoding result by doing one pass of forward computation.
      float post = Exp(ForwardBackward(word, result, ngram_order_-1, true));
      results->push_back(std::pair<std::vector<int32>, float>(result, post));
      prob_mass_tot += post;
      if (results->size() == num_variants_ * 2) break;
    } else {
      for (int32 i = 0; i < graph_[state].size(); i++) {
        SourceInfo source_info = graph_[state][i];
        float fcost = hyp.fcost + source_info.arc_cost;
        std::vector<int32> phone_seq(hyp.phone_seq);
        int32 p = source_info.graphone.second;
        if (p != eps_)
          phone_seq.insert(phone_seq.begin(), p);
        q2.push(Hyp(fcost, phone_seq, nodes_visited_[source_info.node_id]));
      }
    }
  }
  for (int32 i = 0; i < results->size(); i++) {
    (*results)[i].second /= prob_mass_tot;
  }
  std::sort(results->begin(), results->end(), CompareResult());
  for (auto x: nodes_all_) {
    delete x;
  }
  nodes_visited_.clear();
  nodes_all_.clear();
  graph_.clear();
  best_cost_.clear();
  return;
}

/// Read sequences of integers (must be separated by spaces) from a file. 
void G2PModel::ReadSequences(int32 vocab_size, char* file, std::vector<std::vector<int32> >* sequences) {
  std::ifstream data_input(file);
  if (!data_input.is_open()) {
    std::cerr << "error opening '" << file
              << "' for reading\n";
    exit(1);
  }
  std::string line;
  while (getline(data_input, line)) {
    std::istringstream iss(line);
    std::vector<int32> symbols;
    symbols.push_back(bos_);
    while (iss.peek(), !iss.eof()) {
      iss >> std::ws; 
      int32 symbol;
      iss >> symbol;
      if (symbol > vocab_size || symbol <= 0) {
        std::cout << "oov symbol " << symbol << " encountered in the " << symbols.size() << "th sequence. Skipping it." << std::endl;
      } else {
        symbols.push_back(symbol);
      }
    }
    symbols.push_back(eos_);
    sequences->push_back(symbols);
    // std::cout << "end reading word" << std::endl;
  }
}

/// Get the probability of a graphone given a requested n-gram order and a history. 
float G2PModel::GetProb(const int32& order, // ngram order. 0 == unigram
                        const HistType& h, // history
                        const std::pair<int32, int32>& g) { //graphone
  assert(h.size() <= order);
  auto it = prob_[order].find(h);
  if (it == prob_[order].end() ||
      it->second.find(g) == it->second.end()) {
    if (order == 0) {
      return 1e-37f;
    } else {
      // back off to an lower order history.
      HistType h_reduced(h);
      if (h.size() == order) {
        h_reduced.erase(h_reduced.begin());
      }
      return GetProb(order-1, h_reduced, g);
    }
  }
  // We always assign a small probability mass to zero-prob events.
  // This helps with the numerical behavior during decoding when we encounter
  // events unseen in training data.
  // TODO: make probs sum up to one considering this?
  if (it->second[g] < 1e-37f) return 1e-37f; 
  return it->second[g];          
}

/// Do forward-backward computation, accumulating n-gram counts, given a pair
/// of word and pronuncation, and the specified n-gram order (length of histories).
float G2PModel::ForwardBackward(const std::vector<int>& word, 
                                const std::vector<int>& pron,
                                const int32& order,
                                const bool& skip_backward,
                                CountType* counts) {
  std::vector<std::vector<HistType> > hist; // history on each node.
  std::vector<std::vector<float> > alpha;
  alpha.resize(word.size(), std::vector<float>(pron.size(), 0.0f));
  std::vector<std::vector<float> > beta;
  beta.resize(word.size(), std::vector<float>(pron.size(), 0.0f));
  HistType h;
  hist.resize(word.size(), std::vector<HistType >(pron.size(), h));
  for (int32 i = 0; i < word.size(); i++) {
    for (int32 j = 0; j < pron.size(); j++) {
      // std::cout << i << " OOO " << j << std::endl;
      HistType h_temp;
      if (i == 0 && j == 0) {
        alpha[i][j] = 0.0f;
        for (int32 k = 0; k < order; k++) {
          h_temp.push_back(std::pair<int32, int32>(bos_, bos_));
        }
        hist[i][j] = h_temp;
      } else {
        if (i > 0 && j > 0) {
          std::pair<int32, int32> g(word[i], pron[j]);
          alpha[i][j] = alpha[i-1][j-1] + Log(GetProb(order, hist[i-1][j-1], g));
          std::pair<int32, int32> g1(eps_, pron[j]);
          alpha[i][j] = LogAdd(alpha[i][j], alpha[i][j-1] + Log(GetProb(order, hist[i][j-1], g1)));
          std::pair<int32, int32> g2(word[i], eps_);
          alpha[i][j] = LogAdd(alpha[i][j], alpha[i-1][j] + Log(GetProb(order, hist[i-1][j], g2)));
          h_temp = hist[i-1][j-1];
          h_temp.push_back(g);
        } else if (j == 0) {
          std::pair<int32, int32> g(word[i], eps_);
          alpha[i][j] = alpha[i-1][j] + Log(GetProb(order, hist[i-1][j], g));
          h_temp = hist[i-1][j];
          h_temp.push_back(std::pair<int32, int32>(word[i], bos_));
        } else { // j > 0 && i == 0
          assert(i == 0);
          std::pair<int32, int32> g(eps_, pron[j]);
          alpha[i][j] = alpha[i][j-1] + Log(GetProb(order, hist[i][j-1], g));
          h_temp = hist[i][j-1];
          h_temp.push_back(std::pair<int32, int32>(bos_, pron[j]));
        }
        int32 start = 0;
        int32 l = h_temp.size(); 
        if (l > order) start = l - order;
        hist[i][j] = HistType(h_temp.begin()+start, h_temp.end()); 
        // std::cout << i << " " << j << " " << alpha[i][j] << std::endl;
      }
    } 
  }
  // std::cout << "last alpha " << alpha[word.size()-1][pron.size()-1] << std::endl; 
  float log_like = alpha[word.size()-1][pron.size()-1];
  if (skip_backward) return log_like;
  for (int32 i = word.size()-1; i >= 0; i--) {
    for (int32 j = pron.size()-1; j >= 0; j--) {
      if (i == word.size()-1 && j == pron.size()-1) {
        beta[i][j] = -alpha[i][j];
      } else {
        HistType h(hist[i][j]);
        beta[i][j] = -std::numeric_limits<float>::infinity();
        if (i < word.size()-1 && j < pron.size()-1) {
          std::pair<int32, int32> g(word[i+1], pron[j+1]);
          float p = GetProb(order, h, g);
          float b = beta[i+1][j+1] + Log(p);
          beta[i][j] = LogAdd(beta[i][j], b);
          AddCount(h, g, Exp(alpha[i][j] + b), counts);
        } 
        if (i < word.size()-1) {
          std::pair<int32, int32> g(word[i+1], eps_);
          float p = GetProb(order, h, g);
          float b = beta[i+1][j] + Log(p);
          beta[i][j] = LogAdd(beta[i][j], b);
          AddCount(h, g, Exp(alpha[i][j] + b), counts);
        } 
        if (j < pron.size()-1) {
          std::pair<int32, int32> g(eps_, pron[j+1]);
          float p = GetProb(order, h, g);
          float b = beta[i][j+1] + Log(p);
          beta[i][j] = LogAdd(beta[i][j], b);
          AddCount(h, g, Exp(alpha[i][j] + b), counts);
        }
      }
    }
  }
  assert(!isnan(beta[0][0]));
  assert(fabs(beta[0][0] - alpha[0][0]) < 1e-3f);
  return log_like;
}

/// Add count obtained in forward-backward computation into counts_.
void G2PModel::AddCount(const HistType& h, // history
                        const std::pair<int32, int32>& g, // graphone
                        const float& value, CountType *counts) {
  if (value > 0.0f)
    if ((*counts).find(h) == (*counts).end() ||
        (*counts)[h].find(g) == (*counts)[h].end())
      (*counts)[h][g] = value; 
    else
      (*counts)[h][g] += value; 
 } 

/// Merge the contents of a CountType counts into counts_
void G2PModel::MergeCount(CountType& counts, const int32& order) {
 for (CountType::iterator it1 = counts.begin();
   it1 != counts.end(); ++it1) {
   HistType h(it1->first);
   for (auto it2 = counts[h].begin();
     it2 != counts[h].end(); ++it2) {
     if (h.size() == order && it2->second > 0.0f)
       AddCount(h, it2->first, it2->second, &(counts_[order]));
    }
  }
} 
 
/// Update the model represented by prob_, using the accumulated n-gram counts.
/// The standard Kneyser-Ney smooting is used. 
void G2PModel::UpdateProb(const int32 order) {
  float tot_counts = 0.0f;
  if (order == 0) {
    HistType h_0; // empty history vector
    for (auto it = counts_[0][h_0].begin();
         it != counts_[0][h_0].end(); ++it) {
      tot_counts += it->second;
    };
    for (auto it = counts_[0][h_0].begin();
         it != counts_[0][h_0].end(); ++it) {
        prob_[0][h_0][it->first] = it->second / tot_counts;
    }
  } else {
    for(CountType::iterator it1 = counts_[order].begin();
        it1 != counts_[order].end(); ++it1) {
      HistType h(it1->first);
      if (h.size() != order) continue;
      float den = 0.0f; // denominator, the total counts of a given history
      float discounts_tot = 0.0f;
      for (unordered_map<std::pair<int32, int32>, float,
           IntPairHasher>::iterator it2 = counts_[order][h].begin();
        it2 != counts_[order][h].end(); ++it2) {
        den += it2->second;
        if (it2->second < discounting_constants_[order]) {
          discounts_tot += it2->second;
          it2->second = 0.0f;
        } else {
          discounts_tot += discounting_constants_[order];
          it2->second -= discounting_constants_[order];
        }
      }
      /// We won't update the model if the total count of a given history
      /// is too small, cause it's not numerically trustworthy.
      if (den < 1e-20f) continue;
      float backoff_prob = discounts_tot / den;
      HistType h_reduced(h.begin()+1, h.end());
      auto it3 = prob_[order-1].find(h_reduced);
      for (auto it2 = counts_[order][h].begin();
           it2 != counts_[order][h].end(); ++it2) {
        // the probability of the lower order model, given the reduced history.
        float p_lower = 0.0f;
        if (it3 != prob_[order-1].end()) {
          auto it4 = it3->second.find(it2->first);
          if (it4 != it3->second.end())
            p_lower = it4->second; 
        }
        prob_[order][h][it2->first] = counts_[order][h][it2->first] / den + backoff_prob * p_lower;
      }
    }
  }
}
 
/// Compute the bound_matrix_ from the model represented by prob_.
/// See comments near the declaration of bound_matrix_.
void G2PModel::ComputeBoundMatrix() {
  std::vector<int32> valid_graphemes;
  for (int32 i = 1; i <= num_graphemes_; i++) {
    valid_graphemes.push_back(i);
  }
  valid_graphemes.push_back(bos_);
  // given a bigram of letters: l2 | l1, we iterate over prob_ to to find the
  // largest possible probability p((l2, *) | (l1, *), (*, *),...) matching this bigram.
  for (int32 order = 0; order < ngram_order_; order++) {
    if (order == 0) {
      for (int32 i = 0; i < valid_graphemes.size(); i++) {
        int32 l1 = valid_graphemes[i];
        HistType h0;
        for (unordered_map<std::pair<int32, int32>, float,
             IntPairHasher>::iterator it2 = prob_[order][h0].begin();
          it2 != prob_[order][h0].end(); ++it2) {
          int32 l2 = it2->first.first;
          float p = it2->second;
          if (bound_matrix_.find(l1) == bound_matrix_.end() ||
              bound_matrix_[l1].find(l2) == bound_matrix_[l1].end() ||
              bound_matrix_[l1][l2] < p)
            bound_matrix_[l1][l2] = p;
        }
      } 
    } else {
      for (CountType::iterator it1 = prob_[order].begin();
        it1 != prob_[order].end(); ++it1) {
        HistType h(it1->first);
        int32 l1 = h[h.size()-1].first;
        for (unordered_map<std::pair<int32, int32>, float,
             IntPairHasher>::iterator it2 = prob_[order][h].begin();
          it2 != prob_[order][h].end(); ++it2) {
          int32 l2 = it2->first.first;
          float p = it2->second;
          if (bound_matrix_.find(l1) == bound_matrix_.end() ||
              bound_matrix_[l1].find(l2) == bound_matrix_[l1].end() ||
              bound_matrix_[l1][l2] < p)
            bound_matrix_[l1][l2] = p;
        }
      }
    }
  }
  for (unordered_map<int32, unordered_map<int32, float> >::iterator it1 = 
         bound_matrix_.begin(); it1 != bound_matrix_.end(); it1++) {
    for (unordered_map<int32, float>::iterator it2 = it1->second.begin();
           it2 != it1->second.end(); it2++) {
      if (it2->second == 0.0f) it2->second = 1e-37f; 
    }
  }
}

/// Enqueue a Node into the queue in A-star search.
void G2PModel::Enqueue(const Node& node, const std::pair<int32, int32>& g,
                       const HistType& h,
                       const std::vector<float>& heuristics,
                       ForwardQueueType* q,
                       std::vector<int32>* num_active,
                       std::vector<float>* beam_width) {
  float logp = Log(GetProb(h.size(), h, g));
  float fcost_new = logp + node.fcost;
  float bcost_new = node.bcost;
  std::vector<int32> phone_hist_new(node.state.phone_hist);
  
  if (g.second == eos_) assert(g.first == eos_);
  if (g.first == eos_) assert(g.second == eos_);
  if (g.second != eos_) assert(phone_hist_new[phone_hist_new.size()-1] != eos_);
  if (g.second != eps_)
    phone_hist_new.push_back(g.second);

  if (phone_hist_new.size() > ngram_order_-1)
    phone_hist_new.erase(phone_hist_new.begin());

  int32 pos_new = node.state.pos;
  if (g.first != eps_) {
    pos_new += 1;
    bcost_new -= heuristics[node.state.pos];
  //  float p = GetProb(h.size(), h, g);
  //  if (!(Log(p)-heuristics[node.pos] < 1e-5f)) {
  //    std::cout << "warning: log prob larger than heuristics: " << Log(p) << " " << heuristics[node.pos] << std::endl;
  //    assert(Log(p)-heuristics[node.pos] < 1e-5f);
  //  }
  }
  // We only enqueue a Node if the cost is better than before with the same
  // (phoneme-sequence, word-position), and the number of Nodes in the queue
  // with the same word-position is smaller than max_num_active_nodes.
  State s(pos_new, phone_hist_new);
  SourceInfo source_info(nodes_visited_.size()-1, g, logp); // backpointer to the source state.
  auto it = best_cost_.find(s);
  if (it == best_cost_.end()) {
    Node *node_new = new Node(s, fcost_new, bcost_new);
    nodes_all_.insert(node_new);
    q->push(std::pair<float, Node*>(fcost_new+bcost_new, node_new));
    best_cost_[s] = std::pair<Node*, SourceInfo>(node_new, source_info); 
  } else if (fcost_new > it->second.first->fcost) {
    auto node_to_update = it->second.first;
    node_to_update->fcost = fcost_new;
    node_to_update->bcost = bcost_new;
    it->second.second = source_info; 
  }
  auto it2 = graph_.find(s);
  if (it2 == graph_.end()) {
    std::vector<SourceInfo> sources;
    sources.push_back(source_info);
    graph_[s] = sources; 
  } else {
    graph_[s].push_back(source_info); 
  }

  //  if ((*num_active)[node_new.pos] < max_num_active_nodes_) {
      
      // std::cout << (*num_active)[node_new.pos] << " " <<  max_num_active_nodes_ << std::endl;
    //  (*num_active)[node_new->state.pos] += 1;
    //  if (cost_new > (*beam_width)[node_new->state.pos])
    //    (*beam_width)[node_new->state.pos] = cost_new;
 //   } else if (cost_new > (*beam_width)[node_new.pos]) {
      // std::cout << " the current node's cost " << cost_new << "is better than the beam width. " << (*beam_width)[node_new.pos] << std::endl;
      // q->push(node_new);
 //   } else {
     // std::cout << " the current node's cost " << cost_new << "is worse than the beam width. " << (*beam_width)[node_new.pos] << std::endl;
 //   }
 // }
}

void G2PModel::WriteProb(std::ostream &os, bool binary) const {
  if (!os.good()) {
    std::cerr << "Failure writing probs to stream.\n";
  }
  if (binary) {
 
  } else { // text mode.
    for (int32 o = 0; o < ngram_order_; o++) {
      // iterate the hashmap of vector of int pairs (history) and dict of graphone vs. prob
      os << "{";
      CountType::const_iterator it = prob_[o].begin();
      for (; it != prob_[o].end(); ++it) {
        // write history
        os << "( ";
        std::vector<std::pair<int32, int32> >::const_iterator it_v = (it->first).begin();
        // empty pair represents empty history
        if (it_v == (it->first).end()) {
          os << "( " << ") ";
        }
        for (; it_v != (it->first).end(); ++it_v) {
          os << "( " << it_v ->first << " " << it_v ->second << " ) ";
        }
        os << ") ";
        // write hashmap of graphone and prob
        os << " {";
        Graphone2ProbType::const_iterator it_gp = (it->second).begin();
        for (; it_gp != (it->second).end(); ++it_gp) {
          // write graphone
          os << " ( ";
          os << "( " << (it_gp->first).first << " " << (it_gp->first).second << " ) ";
          // write prob
          os << it_gp->second << " )";
        }
        os << "}";
      }
      os << "} ";
    }
  }
}

void G2PModel::ReadProb(std::istream &is, bool binary, std::vector<CountType>* prob) {
  if (is.fail()) {
    std::cerr << "Failure reading from input.\n";
  }
  if (binary) {
  
  } else { //text mode.
    (*prob).resize(ngram_order_);
    for (int32 order = 1; order <= ngram_order_; order++) {
      char ch;
      is >> ch;
      assert(ch == '{');
      // std::cout << "symbol should be '{' " << ch << std::endl;
      // std::cout << "current order is " << order << std::endl;
      while(is.peek() != '}') {
        HistType h;
        Graphone2ProbType v;
        ReadHistory(is, binary, &h, order);
        ReadGraphone2ProbMap(is, binary, &v);
        (*prob)[order-1][h] = v;
      }
      is >> ch;
      assert(ch == '}');
      // std::cout << "order " << order << " is finished. current symbol is '}'" << ch<< std::endl;
    }
  } // end of text mode
}

// Test Read and Write functions. 
void G2PModel::TestWriteRead(char* file, bool binary) {
  Write(file, binary);
  std::cout << "print out the original probs: ";
  PrintWrite();

  std::vector<CountType> prob;
  std::ifstream f1;
  f1.open(file);
  ReadProb(f1, binary, &prob);
  PrintRead(&prob);
  f1.close();
}

void G2PModel::Write(char* file, bool binary) const {
  std::ofstream f(file);
  WriteProb(f, binary);
  std::cout << "print out the original probs: ";
  f.close();
}

void G2PModel::Read(char* file, bool binary) {
  std::ifstream f;
  f.open(file);
  ReadProb(f, binary, &prob_);
  f.close();
  std::cout << "finished reading model." << std::endl;
}

void G2PModel::PrintWrite() {
  for (int32 o = 0; o < ngram_order_; o++) {
    // iterate the hashmap of vector of int pairs (history) and dict of graphone vs. prob
    std::cout << "order is " << o + 1 << std::endl;
    CountType::const_iterator it = prob_[o].begin();
    for (; it != prob_[o].end(); it++) {
      // write history
      std::cout << "Write: history is " << std::endl;
      HistType::const_iterator it_v = (it->first).begin();
      for (; it_v != (it->first).end(); it_v++) {
        std::cout << "( " << it_v->first << " " << it_v->second << " )" << std::endl;
      }
      // write hashmap of graphone and prob
      Graphone2ProbType::const_iterator it_gp = (it->second).begin();
      for (; it_gp != (it->second).end(); it_gp++) {
        // write graphone
        std::cout  << "Write: graphone is " << (it_gp->first).first << " " << (it_gp->first).second << std::endl;
        // write prob
        std::cout << "Write: float is " << it_gp->second << std::endl;
      }
    }
  }
}

void G2PModel::PrintRead(std::vector<CountType>* prob) {
  for (int32 o = 0; o < ngram_order_; o++) {
    std::cout << "order is " << o + 1 << std::endl;
    CountType::const_iterator it = (*prob)[o].begin();
    for (; it != (*prob)[o].end(); ++it) {
      std::cout << "Read: history is " << std::endl;
      HistType::const_iterator it_v = (it->first).begin();
      for (; it_v != (it->first).end(); ++it_v) {
        std::cout << "( " << it_v->first << " " << it_v->second << " )" << std::endl;
      }
      Graphone2ProbType::const_iterator it_gp = (it->second).begin();
      for (; it_gp != (it->second).end(); ++it_gp) {
        std::cout  << "Read: graphone is " << (it_gp->first).first << " " << (it_gp->first).second << std::endl;
        std::cout << "Read: float is " << it_gp->second << std::endl;
      }
    }
  }
}

