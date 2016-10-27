// g2p_utils.cc

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

#include "g2p_utils.h"

// Read a pair of ints from istream.
void ReadPair(std::istream &is, bool binary, std::pair<int32, int32>* graphone) {
  if (binary) {
  
  } else {
    int32 grapheme_id;
    int32 phoneme_id;
    char ch;
    is >> ch;
    assert(ch == '('); // start of a graphone pair
    
    ch = is.peek(); 
    if (ch == ')') {
      is >> ch;
      std::cerr << "fail to get valid graphone pairs.\n"; 
      exit(1);
    } 
    is >> grapheme_id;
    is >> phoneme_id; // read the two integers representing a graphone
    // std::cout << "the first number: "  << grapheme_id << " the second number: "  << phoneme_id << std::endl;
    *graphone = std::make_pair(grapheme_id, phoneme_id);
    is >> ch;
    // std::cout << "expected symbol is ')' " << ch << std::endl;
    assert(ch == ')'); // end of a graphone pair
  }
} 

// Read a history (vector of pair of integers) from stream
void ReadHistory(std::istream &is, bool binary, HistType* hist, int32 order) {
  if (binary) {
  
  } else {
    char ch;
    is >> ch;
    assert(ch == '('); // start of a history
    // std::cout << "history: current symbol should be '('  " << ch << std::endl; 
    // std::cout << "current order should be 1 "<< order << std::endl;
    if (order == 1) { // empty history
      is >> ch;
      assert(ch == '(');
      // std::cout <<"history: current order is " << order << ". symbol should be '(' " << ch << std::endl; 
      is >> ch;
      assert(ch == ')');
    } else {
      (*hist).resize(order - 1);
      for (int32 i = 0; i < order - 1; i++) {
        ReadPair(is, binary, &((*hist)[i]));
      }
    }
    is >> ch;
    assert(ch == ')'); // end of a history
  }
}

// Read a map from graphone (pair of integers) to prob (float) from istream.
void ReadGraphone2ProbMap(std::istream &is, bool binary, Graphone2ProbType* graphone_map) {
  if (binary) {
  
  } else {
    float prob_value;
    char ch;
    is >> ch;
    assert(ch == '{'); // start of a graphone2prob map
    while(is.peek() != '}') {
      is >> ch;
      // std::cout << "ReadGraphone2ProbMap: symbol should be '('" << ch << std::endl;
      assert(ch == '('); // start of an entry of the graphone2prob map
      
      std::pair<int32, int32> g;
      ReadPair(is, binary, &g);
      
      is >> prob_value;
     // std::cout << "the prob is " << prob_value << std::endl;
      (*graphone_map)[g] = prob_value;

      is >> ch;
      assert(ch == ')');
    }
    is >> ch;
    assert(ch == '}'); // end of a graphone2prob map
  }
}

