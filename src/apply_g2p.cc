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
  // arguments: ngram_order, num_graphemes, num_phonemes, input_model, test_words, predicted_prons.
  G2PModel mdl(atoi(argv[1]), atoi(argv[2]), atoi(argv[3])); 
  mdl.Read(argv[4], false);
  mdl.Test(argv[5], argv[6], 1);
  return 0;
}
