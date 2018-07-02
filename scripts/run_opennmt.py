#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Main training workflow
"""
from __future__ import print_function
from __future__ import division

import random
from torch import cuda
from onmt.inputters.inputter import build_dataset_iter, lazily_load_dataset, _load_fields, _collect_report_features
from onmt.model_builder import build_model
from onmt.utils.optimizers import build_optim
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
import argparse
import os
import glob
import sys
import torch
from onmt.utils.misc import get_logger
import onmt.inputters as inputters
import onmt.opts as opts
import gzip
import tempfile
import shutil
import logging
from vivisect.pytorch import probe
from vivisect.servers import clear, flush


def check_existing_pt_files(opt):
    """ Checking if there are existing .pt files to avoid tampering """
    # We will use glob.glob() to find sharded {train|valid}.[0-9]*.pt
    # when training, so check to avoid tampering with existing pt files
    # or mixing them up.
    for t in ['train', 'valid', 'vocab']:
        pattern = opt.save_data + '.' + t + '*.pt'
        if glob.glob(pattern):
            sys.stderr.write("Please backup existing pt file: %s, "
                             "to avoid tampering!\n" % pattern)
            sys.exit(1)


def build_save_in_shards(src_corpus, tgt_corpus, fields,
                         corpus_type, opt, logger=None):
    """
    Divide the big corpus into shards, and build dataset separately.
    This is currently only for data_type=='text'.

    The reason we do this is to avoid taking up too much memory due
    to sucking in a huge corpus file.

    To tackle this, we only read in part of the corpus file of size
    `max_shard_size`(actually it is multiples of 64 bytes that equals
    or is slightly larger than this size), and process it into dataset,
    then write it to disk along the way. By doing this, we only focus on
    part of the corpus at any moment, thus effectively reducing memory use.
    According to test, this method can reduce memory footprint by ~50%.

    Note! As we process along the shards, previous shards might still
    stay in memory, but since we are done with them, and no more
    reference to them, if there is memory tight situation, the OS could
    easily reclaim these memory.

    If `max_shard_size` is 0 or is larger than the corpus size, it is
    effectively preprocessed into one dataset, i.e. no sharding.

    NOTE! `max_shard_size` is measuring the input corpus size, not the
    output pt file size. So a shard pt file consists of examples of size
    2 * `max_shard_size`(source + target).
    """

    corpus_size = os.path.getsize(src_corpus)
    if corpus_size > 10 * (1024 ** 2) and opt.max_shard_size == 0:
        if logger:
            logger.info("Warning. The corpus %s is larger than 10M bytes, "
                        "you can set '-max_shard_size' to process it by "
                        "small shards to use less memory." % src_corpus)

    if opt.max_shard_size != 0:
        if logger:
            logger.info(' * divide corpus into shards and build dataset '
                        'separately (shard_size = %d bytes).'
                        % opt.max_shard_size)

    ret_list = []
    src_iter = inputters.ShardedTextCorpusIterator(
        src_corpus, opt.src_seq_length_trunc,
        "src", opt.max_shard_size)
    tgt_iter = inputters.ShardedTextCorpusIterator(
        tgt_corpus, opt.tgt_seq_length_trunc,
        "tgt", opt.max_shard_size,
        assoc_iter=src_iter)

    index = 0
    while not src_iter.hit_end():
        index += 1
        dataset = inputters.TextDataset(
            fields, src_iter, tgt_iter,
            src_iter.num_feats, tgt_iter.num_feats,
            src_seq_length=opt.src_seq_length,
            tgt_seq_length=opt.tgt_seq_length,
            dynamic_dict=opt.dynamic_dict)

        # We save fields in vocab.pt separately, so make it empty.
        dataset.fields = []

        pt_file = "{:s}.{:s}.{:d}.pt".format(
            opt.save_data, corpus_type, index)
        if logger:
            logger.info(" * saving %s data shard to %s."
                        % (corpus_type, pt_file))
        torch.save(dataset, pt_file)

        ret_list.append(pt_file)

    return ret_list


def build_save_dataset(corpus_type, fields, opt, logger=None):
    """ Building and saving the dataset """
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src_corpus = opt.train_src
        tgt_corpus = opt.train_tgt
    else:
        src_corpus = opt.valid_src
        tgt_corpus = opt.valid_tgt

    # Currently we only do preprocess sharding for corpus: data_type=='text'.
    if opt.data_type == 'text':
        return build_save_in_shards(
            src_corpus, tgt_corpus, fields,
            corpus_type, opt)

    # For data_type == 'img' or 'audio', currently we don't do
    # preprocess sharding. We only build a monolithic dataset.
    # But since the interfaces are uniform, it would be not hard
    # to do this should users need this feature.
    dataset = inputters.build_dataset(
        fields, opt.data_type,
        src_path=src_corpus,
        tgt_path=tgt_corpus,
        src_dir=opt.src_dir,
        src_seq_length=opt.src_seq_length,
        tgt_seq_length=opt.tgt_seq_length,
        src_seq_length_trunc=opt.src_seq_length_trunc,
        tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
        dynamic_dict=opt.dynamic_dict,
        sample_rate=opt.sample_rate,
        window_size=opt.window_size,
        window_stride=opt.window_stride,
        window=opt.window)

    # We save fields in vocab.pt seperately, so make it empty.
    dataset.fields = []

    pt_file = "{:s}.{:s}.pt".format(opt.save_data, corpus_type)
    if logger:
        logger.info(" * saving %s dataset to %s." % (corpus_type, pt_file))
    torch.save(dataset, pt_file)

    return [pt_file]


def build_save_vocab(train_dataset, fields, opt, logger=None):
    """ Building and saving the vocab """
    fields = inputters.build_vocab(train_dataset, fields, opt.data_type,
                                   opt.share_vocab,
                                   opt.src_vocab,
                                   opt.src_vocab_size,
                                   opt.src_words_min_frequency,
                                   opt.tgt_vocab,
                                   opt.tgt_vocab_size,
                                   opt.tgt_words_min_frequency,
                                   logger)

    # Can't save fields, so remove/reconstruct at training time.
    vocab_file = opt.save_data + '.vocab.pt'
    torch.save(inputters.save_fields_to_vocab(fields), vocab_file)


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    print('encoder: ' + str(enc))
    print('decoder: ' + str(dec))


def training_opt_postprocessing(opt):
    if opt.word_vec_size != -1:
        opt.src_word_vec_size = opt.word_vec_size
        opt.tgt_word_vec_size = opt.word_vec_size

    if opt.layers != -1:
        opt.enc_layers = opt.layers
        opt.dec_layers = opt.layers

    opt.brnn = (opt.encoder_type == "brnn")
    if opt.seed > 0:
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)

    if opt.rnn_type == "SRU" and not opt.gpuid:
        raise AssertionError("Using SRU requires -gpuid set.")

    if torch.cuda.is_available() and not opt.gpuid:
        print("WARNING: You have a CUDA device, should run with -gpuid 0")

    if opt.gpuid:
        cuda.set_device(opt.gpuid[0])
        if opt.seed > 0:
            torch.cuda.manual_seed(opt.seed)

    if len(opt.gpuid) > 1:
        sys.stderr.write("Sorry, multigpu isn't supported yet, coming soon!\n")
        sys.exit(1)
    return opt


def training_main(opt):
    opt = training_opt_postprocessing(opt)

    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        print('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
        # I don't like reassigning attributes of opt: it's not clear.
        opt.start_epoch = checkpoint['epoch'] + 1
    else:
        checkpoint = None
        model_opt = opt

    # Peek the fisrt dataset to determine the data_type.
    # (All datasets have the same data_type).
    first_dataset = next(lazily_load_dataset("train", opt))
    data_type = first_dataset.data_type

    # Load fields generated from preprocess phase.
    fields = _load_fields(first_dataset, data_type, opt, checkpoint)

    # Report src/tgt features.
    _collect_report_features(fields)

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    _tally_parameters(model)
    _check_save_model_path(opt)

    model._vivisect = {"iteration" : 0, "model_name" : "OpenNMT Model", "framework" : "pytorch", "mode" : "train"}
    probe(model, "localhost", 8082)
    
    # Build optimizer.
    optim = build_optim(model, opt, checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    trainer = build_trainer(
        opt, model, fields, optim, data_type, model_saver=model_saver)

    def train_iter_fct():
        model._vivisect["iteration"] += 1
        model._vivisect["mode"] = "train"
        return build_dataset_iter(lazily_load_dataset("train", opt), fields, opt)

    def valid_iter_fct():
        model._vivisect["mode"] = "dev"
        return build_dataset_iter(lazily_load_dataset("valid", opt), fields, opt)

    # Do training.
    trainer.train(train_iter_fct, valid_iter_fct, opt.start_epoch, opt.epochs)

    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()

        
def preprocess_main(opt):
    logger = get_logger(opt.log_file)
    src_nfeats = inputters.get_num_features(
        opt.data_type, opt.train_src, 'src')
    tgt_nfeats = inputters.get_num_features(
        opt.data_type, opt.train_tgt, 'tgt')
    logger.info(" * number of source features: %d." % src_nfeats)
    logger.info(" * number of target features: %d." % tgt_nfeats)

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(opt.data_type, src_nfeats, tgt_nfeats)

    logger.info("Building & saving training data...")
    train_dataset_files = build_save_dataset('train', fields, opt, logger)

    logger.info("Building & saving vocabulary...")
    build_save_vocab(train_dataset_files, fields, opt, logger)

    logger.info("Building & saving validation data...")
    build_save_dataset('valid', fields, opt, logger)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", dest="source")
    parser.add_argument("--target", dest="target")
    parser.add_argument("--epochs", dest="epochs", default=10, type=int)
    args = parser.parse_args()
    
    temp = tempfile.mkdtemp()
    
    source = []
    with gzip.open(args.source, "rt") as ifd:
            for line in ifd:
                source.append(line)
                
    target = []
    with gzip.open(args.target, "rt") as ifd:
            for line in ifd:
                target.append(line)
        
    pairs = list(zip(source, target))[0:50]
    random.shuffle(pairs)
    os.mkdir(os.path.join(temp, "data"))
    
    with open(os.path.join(temp, "data", "train_source.txt"), "wt") as sofd, open(os.path.join(temp, "data", "train_target.txt"), "wt") as tofd:
        for s, t in pairs[0:int(.9 * len(pairs))]:
            sofd.write(s)
            tofd.write(t)
            
    with open(os.path.join(temp, "data", "dev_source.txt"), "wt") as sofd, open(os.path.join(temp, "data", "dev_target.txt"), "wt") as tofd:
        for s, t in pairs[int(.9 * len(pairs)):]:
            sofd.write(s)
            tofd.write(t)
    
    preproc_parser = argparse.ArgumentParser(
        description='vivisect example',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    opts.add_md_help_argument(preproc_parser)
    opts.preprocess_opts(preproc_parser)
    
    preproc_args = preproc_parser.parse_args(args=["-train_src", os.path.join(temp, "data", "train_source.txt"),
                                                   "-train_tgt", os.path.join(temp, "data", "train_target.txt"),
                                                   "-valid_src", os.path.join(temp, "data", "dev_target.txt"),
                                                   "-valid_tgt", os.path.join(temp, "data", "dev_target.txt"),
                                                   "-save_data", os.path.join(temp, "data", "out")])

    train_parser = argparse.ArgumentParser(
        description='vivisect example',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    opts.add_md_help_argument(train_parser)
    opts.model_opts(train_parser)
    opts.train_opts(train_parser)
    train_args = train_parser.parse_args(["-data", os.path.join(temp, "data/out"),
                                          "-epochs", str(args.epochs),
                                          "-save_model", os.path.join(temp, "model")])

    clear("localhost", 8082)
    
    try:
        torch.manual_seed(preproc_args.seed)
        check_existing_pt_files(preproc_args)
        preprocess_main(preproc_args)
        training_main(train_args)
    except Exception as e:
        raise e
    finally:
        flush("localhost", 8082)
        shutil.rmtree(temp)
