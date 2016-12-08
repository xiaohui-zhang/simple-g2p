// apply_g2p.cc

// Copyright     2016  Xiaohui Zhang

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

int main(int argc, char **argv) {
  // arguments: 
  // 1:ngram_order, 2:discouting_constant_min, 3:discounting_constant_max, 4:num_graphemes, 
  // 5:num_phonemes, 6:training words, 7:training prons, 8:valid words, 
  // 9:valid prons, 10:num_threads, 11:output_model
  G2PModel mdl(atoi(argv[1]), atof(argv[2]), atof(argv[3]), atoi(argv[4]), atoi(argv[5]), argv[6], argv[7], argv[8], argv[9]); 
  mdl.Train(atoi(argv[10]));
  mdl.Write(argv[11], true);
  return 0;
}
