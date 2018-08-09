#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Main training workflow
"""
from __future__ import print_function
from __future__ import division

import random
import argparse
import os
from glob import glob
import sys
import gzip
import tempfile
import shutil
import logging
import nltk


pos = {"de" : nltk.tag.stanford.StanfordPOSTagger("/home/tom/data/stanford/edu/stanford/nlp/models/pos-tagger/german/german-fast.tagger", "/home/tom/data/st/stanford-postagger-2018-02-27/stanford-postagger-3.9.1.jar", java_options="-mx8000m"),
       "en" : nltk.tag.stanford.StanfordPOSTagger("/home/tom/data/st/stanford-postagger-2018-02-27/models/english-bidirectional-distsim.tagger", "/home/tom/data/st/stanford-postagger-2018-02-27/stanford-postagger-3.9.1.jar", java_options="-mx8000m"),
}

ner = {"de" : nltk.tag.stanford.StanfordNERTagger("/home/tom/data/stanford/edu/stanford/nlp/models/ner/german.conll.germeval2014.hgc_175m_600.crf.ser.gz", "/home/tom/data/sn/stanford-ner-2018-02-27/stanford-ner.jar", java_options="-mx8000m"),
       "en" : nltk.tag.stanford.StanfordNERTagger("/home/tom/data/sn/stanford-ner-2018-02-27/classifiers/english.muc.7class.distsim.crf.ser.gz", "/home/tom/data/sn/stanford-ner-2018-02-27/stanford-ner.jar", java_options="-mx8000m"),
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    parser.add_argument("-c", "--count", dest="count", type=int, default=5000, help="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    domains = ["ted", "wipo"]
    langs = ["en", "de"]

    domain_labels = []
    pos_labels = {"en" : [],
                  "de" : []
    }
    ner_labels = {"en" : [],
                  "de" : []
    }
    all_data = {"en" : [],
                "de" : [],
    }
    for domain in domains:
        data = {}
        with gzip.open(os.path.join(args.input, "{}.train.raw.en.gz".format(domain)), "rt") as ifdEN, gzip.open(os.path.join(args.input, "{}.train.raw.de.gz".format(domain)), "rt") as ifdDE:            
            bitext = list(zip([x.strip().split() for x in ifdEN], [x.strip().split() for x in ifdDE]))
            random.shuffle(bitext)
            data["en"] = [x for x, _ in bitext[0:args.count]]
            data["de"] = [x for _, x in bitext[0:args.count]]
            
        domain_labels += ([domain] * len(data["en"]))
        for lang in langs:
            logging.info("Starting POS tagging...")
            pos_labels[lang] += [[x for _, x in s] for s in pos[lang].tag_sents(data[lang])]
            logging.info("Done POS tagging, starting NER tagging...")
            ner_labels[lang] += [[x for _, x in s] for s in ner[lang].tag_sents(data[lang])]
            logging.info("Done NER tagging.")
            all_data[lang] += data[lang]

    with gzip.open(os.path.join(args.output, "domain.txt.gz"), "wt") as ofd:
        ofd.write("\n".join(domain_labels) + "\n")
    for lang in langs:
        with gzip.open(os.path.join(args.output, "{}.txt.gz".format(lang)), "wt") as ofd:
            ofd.write("\n".join([" ".join(toks) for toks in all_data[lang]]) + "\n")
        with gzip.open(os.path.join(args.output, "{}_pos.txt.gz".format(lang)), "wt") as ofd:
            ofd.write("\n".join([" ".join(toks) for toks in pos_labels[lang]]) + "\n")
        with gzip.open(os.path.join(args.output, "{}_ner.txt.gz".format(lang)), "wt") as ofd:
            ofd.write("\n".join([" ".join(toks) for toks in ner_labels[lang]]) + "\n")                        
