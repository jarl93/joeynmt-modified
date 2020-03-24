#!/bin/bash

merge_ops=32000
src=en
tgt=vi
lang=en-vi

SAVE_DIR="test/data/iwslt_envi"
mkdir -p ${SAVE_DIR}

cd ${SAVE_DIR}

# train
curl -o train.en   https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.en
curl -o train.vi   https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.vi

# dev
curl -o valid.en https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2012.en
curl -o valid.vi https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2012.vi

# test
curl -o test.en https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.en
curl -o test.vi https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.vi

echo "learning * separate * BPEs..."
codes_file_src="bpe.${merge_ops}.${src}"
codes_file_tgt="bpe.${merge_ops}.${tgt}"

python3 -m subword_nmt.learn_bpe -s "${merge_ops}" -i "train.${src}" -o "${codes_file_src}"
python3 -m subword_nmt.learn_bpe -s "${merge_ops}" -i "train.${tgt}" -o "${codes_file_tgt}"

yes '' | sed 5q

echo "applying BPE..."
for p in train valid test; do
        python3 -m subword_nmt.apply_bpe -c "${codes_file_src}" -i "${p}.${src}" -o "${p}.bpe.${merge_ops}.${src}"
        python3 -m subword_nmt.apply_bpe -c "${codes_file_tgt}" -i "${p}.${tgt}" -o "${p}.bpe.${merge_ops}.${tgt}"
done

#for l in ${src} ${tgt}; do
#    for p in train valid test; do
#        mv ${tmp}/${p}.${l} ${prep}/
#    done
#done

echo "Done pre-processing small corpus."