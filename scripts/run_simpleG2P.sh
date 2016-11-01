#!/bin/bash

# exit on error.
set -e
set -x

# make sure that the 'scripts' directory and the 'src' directory are on the
# path.
rootdir=$(dirname "$0")/..
PATH=$PATH:$rootdir/scripts:$rootdir/src

num_threads=3

# parse options.
for iter in 1; do
  if [ "$1" == "--num_threads" ]; then
    num_threads="$2"
    shift; shift
  fi
done

#if [ $# != 4 ]; then
#  echo "Usage:"
#  echo "  $0 [options] <lexicon> <nonsilence-phones> <ngram-order> <output-dir>"
#  echo "e.g.:  $0 data/local/dict/lexicon.txt data/local/dict/nonsilence_phones.txt 3 exp/"
#  echo
#  echo "This script demontrates how to train and apply a g2p model."
#  echo
#  echo "Options"
#  echo "   --num <true|false>  [default: 3]"
#  echo "      Number of threads used in multithreaded training."
#  exit 1
#fi

lexicon=$1
phones=$2
ngram_order=$3
dir=$4

num_total=1300
num_train=1000
discounting_constant_min=0.1
discounting_constant_max=0.5

# For convenience of testing.
if [ -z $lexicon ]; then lexicon="/export/a11/xzhang/tedlium/s5_r2/data/local/dict/lexicon.txt"; fi
if [ -z $phones ]; then phones="/export/a11/xzhang/tedlium/s5_r2/data/local/dict/nonsilence_phones.txt"; fi
if [ -z $ngram_order ]; then ngram_order=3; fi
if [ -z $dir ]; then dir=exp; fi

mkdir -p $dir
cat $lexicon | sed 's/^<.*>.*$//g' | sed 's/^#.*//g' | sort -R | head -n $num_total > $dir/lexicon_subset

cut -f1 -d' ' $dir/lexicon_subset | grep -o . | sort -u | awk '{print ($1" "NR)}' > $dir/grapheme2int
cat $phones | awk '{print ($1" "NR)}' > $dir/phoneme2int
num_graphemes=`wc -l $dir/grapheme2int | cut -f1 -d' '`
num_phonemes=`wc -l $dir/phoneme2int | cut -f1 -d' '`

# Preapare training data
cat $dir/lexicon_subset | head -n $num_train | cut -f1 -d' ' | sed 's/./& /g' | \
  awk 'NR==FNR{a[$1] = $2; next} {if ($1 in a) printf a[$1]; for (n=2;n<=NF;n++) if ($n in a) printf(" "a[$n]); printf("\n");}' \
  $dir/grapheme2int - > $dir/words

cat $dir/lexicon_subset | head -n $num_train | cut -f2- -d' ' | \
  awk 'NR==FNR{a[$1] = $2; next} {if ($1 in a) printf a[$1]; for (n=2;n<=NF;n++) if ($n in a) printf(" "a[$n]); printf("\n");}' \
  $dir/phoneme2int - > $dir/prons

# Prepare test data
cat $dir/lexicon_subset | head -n-$num_train | cut -f1 -d' ' > $dir/test_words.txt
cat $dir/test_words.txt | sed 's/./& /g' | \
  awk 'NR==FNR{a[$1] = $2; next} {if ($1 in a) printf a[$1]; for (n=2;n<=NF;n++) if ($n in a) printf(" "a[$n]); printf("\n");}' \
  $dir/grapheme2int - > $dir/test_words

cat $dir/lexicon_subset | tail -10 | cut -f2- -d' ' > $dir/test_prons.txt
cat $dir/test_prons.txt | \
  awk 'NR==FNR{a[$1] = $2; next} {if ($1 in a) printf a[$1]; for (n=2;n<=NF;n++) if ($n in a) printf(" "a[$n]); printf("\n");}' \
  $dir/phoneme2int - > $dir/test_prons

# train and evaluate the n-gram graphone model
time train_g2p $ngram_order $discounting_constant_min $discounting_constant_max $num_graphemes \
  $num_phonemes $dir/words $dir/prons $num_threads $dir/final.mdl
time apply_g2p $ngram_order $num_graphemes $num_phonemes $dir/final.mdl $dir/test_words $dir/output 1000
cat $dir/output | cut -f1 -d$'\t' |
  awk 'NR==FNR{a[$2] = $1; next} {if ($1 in a) printf a[$1]; for (n=2;n<=NF;n++) if ($n in a) printf(" "a[$n]); printf("\n");}' \
  $dir/grapheme2int - | sed 's/ //g' > $dir/output_words.txt
cat $dir/output | cut -f3 -d$'\t' |
  awk 'NR==FNR{a[$2] = $1; next} {if ($1 in a) printf a[$1]; for (n=2;n<=NF;n++) if ($n in a) printf(" "a[$n]); printf("\n");}' \
  $dir/phoneme2int - > $dir/output_prons.txt

paste $dir/output_words.txt $dir/output_prons.txt > $dir/output.txt

# time steps/dict/train_g2p.sh $dir/lexicon_subset $dir/sequitur
# time steps/dict/apply_g2p.sh $dir/test_words.txt $dir/sequitur $dir/sequitur/test


#TODO: deal with unseen graphemes
