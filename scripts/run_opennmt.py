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
from onmt.models import build_model_saver
import argparse
import os
import glob
import sys
import torch
import onmt.inputters as inputters
import onmt.opts as opts
import gzip
import tempfile
import shutil
import logging
import nltk
from vivisect import probe, clear, flush, register_clustering_targets, register_classification_targets, remove
import numpy
import onmt.inputters as inputters
import onmt.utils

from onmt.utils.logging import logger





class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1, n_gpu=1, gpu_rank=1,
                 gpu_verbose_level=0, report_manager=None, model_saver=None):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver = model_saver

        assert grad_accum_count > 0
        if grad_accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def train(self, train_iter_fct, valid_iter_fct, train_steps, valid_steps):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info('Start training...')

        step = self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0


        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)
        
        for epoch in range(train_steps):
            train_iter = train_iter_fct()
            #print(step)
            #step += 1
            reduce_counter = 0
            self.model._v.state = "train"
            self.model._v.epoch += 1
            logging.info("Train epoch %d", model._v.epoch)
            for i, batch in enumerate(train_iter):
                #print(type(batch))
                #print(batch.batch_size)
                #continue
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):
                    if self.gpu_verbose_level > 1:
                        logger.info("GpuRank %d: index: %d accum: %d"
                                    % (self.gpu_rank, i, accum))
                    cur_dataset = train_iter.get_cur_dataset()
                    self.train_loss.cur_dataset = cur_dataset

                    true_batchs.append(batch)

                    if self.norm_method == "tokens":
                        num_tokens = batch.tgt[1:].data.view(-1) \
                                     .ne(self.train_loss.padding_idx).sum()
                        normalization += num_tokens
                    else:
                        normalization += batch.batch_size

                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.gpu_verbose_level > 0:
                            logger.info("GpuRank %d: reduce_counter: %d \
                                        n_minibatch %d"
                                        % (self.gpu_rank, reduce_counter,
                                           len(true_batchs)))
                        if self.n_gpu > 1:
                            normalization = sum(onmt.utils.distributed
                                                .all_gather_list
                                                (normalization))

                        self._gradient_accumulation(
                            true_batchs, normalization, total_stats,
                            report_stats)

                        #report_stats = self._maybe_report_training(
                        #    step, train_steps,
                        #    self.optim.learning_rate,
                        #    report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0

            valid_iter = valid_iter_fct()
            #print(step)
            #step += 1
            #reduce_counter = 0
            self.model._v.state = "dev"
            logging.info("Dev epoch %d", model._v.epoch)
            v = self.validate(valid_iter)
            logging.info("Perplexity: %.3f", v.ppl())
            
        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = onmt.utils.Statistics()

        for batch in valid_iter:
            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.cur_dataset = cur_dataset

            src = inputters.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
            else:
                src_lengths = None

            tgt = inputters.make_features(batch, 'tgt')

            # F-prop through the model.
            outputs, attns, _ = self.model(src, tgt, src_lengths)

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            target_size = batch.tgt.size(0)
            # Truncated BPTT
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            dec_state = None
            src = inputters.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
                report_stats.n_src_words += src_lengths.sum().item()
            else:
                src_lengths = None

            tgt_outer = inputters.make_features(batch, 'tgt')

            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model.zero_grad()
                outputs, attns, dec_state = \
                    self.model(src, tgt, src_lengths, dec_state)

                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss.sharded_compute_loss(
                    batch, outputs, attns, j,
                    trunc_size, self.shard_size, normalization)
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

        # 3.bis Multi GPU gradient gather
        if self.n_gpu > 1:
            grads = [p.grad.data for p in self.model.parameters()
                     if p.requires_grad
                     and p.grad is not None]
            onmt.utils.distributed.all_reduce_and_rescale_tensors(
                grads, float(1))

        # 4. Update the parameters and statistics.
        self.optim.step()

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return onmt.utils.Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)



def build_trainer(opt, model, fields, optim, data_type, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    train_loss = onmt.utils.loss.build_loss_compute(
        model, fields["tgt"].vocab, opt)
    valid_loss = onmt.utils.loss.build_loss_compute(
        model, fields["tgt"].vocab, opt, train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count
    n_gpu = len(opt.gpuid)
    gpu_rank = opt.gpu_rank
    gpu_verbose_level = opt.gpu_verbose_level

    report_manager = onmt.utils.build_report_manager(opt)
    trainer = Trainer(model, train_loss, valid_loss, optim, trunc_size,
                      shard_size, data_type, norm_method,
                      grad_accum_count, n_gpu, gpu_rank,
                      gpu_verbose_level, report_manager,
                      model_saver=model_saver)
    return trainer
            

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
                                   opt.tgt_words_min_frequency)
    
    # Can't save fields, so remove/reconstruct at training time.
    vocab_file = opt.save_data + '.vocab.pt'
    torch.save(inputters.save_fields_to_vocab(fields), vocab_file)


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


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", dest="name", default="OpenNMT", help="Model name")
    parser.add_argument("--host", dest="host", default="0.0.0.0", help="Host name")
    parser.add_argument("--port", dest="port", default=8082, type=int, help="Port number")
    parser.add_argument("--source", dest="source")
    parser.add_argument("--target", dest="target")
    parser.add_argument("--word_task", dest="word_tasks", default=[], action="append")
    parser.add_argument("--sentence_task", dest="sentence_tasks", default=[], action="append")
    parser.add_argument("--epochs", dest="epochs", default=10, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    
    temp = tempfile.mkdtemp()
    
    data = []
    source_lang = os.path.basename(args.source).split(".")[0]
    source_max = 10000
    with gzip.open(args.source, "rt") as ifd:
            for line in ifd:
                toks = line.strip().split()
                source_max = max(source_max, len(toks))
                data.append({"source" : line.strip()})
                
    target_lang = os.path.basename(args.target).split(".")[0]
    target_max = 10000
    with gzip.open(args.target, "rt") as ifd:
            for i, line in enumerate(ifd):
                toks = line.strip().split()
                target_max = max(target_max, len(toks))
                data[i]["target"] = line.strip()

    task_names = set()
    
    for wt in args.word_tasks:

        name = os.path.basename(wt).split(".")[0]
        task_names.add(name)
        lookup = {}
        with gzip.open(wt, "rt") as ifd:
            for i, line in enumerate(ifd):
                data[i][name] = [lookup.setdefault(v, len(lookup) + 1) for v in line.strip().split()]
                assert(len(data[i][name]) in [len(data[i][x].split()) for x in ["source", "target"]])
                if len(data[i][name]) == len(data[i]["target"].split()):
                    data[i][name].append(0)
                
    for st in args.sentence_tasks:
        name = os.path.basename(st).split(".")[0]
        lookup = {}
        task_names.add(name)
        with gzip.open(st, "rt") as ifd:
            for i, line in enumerate(ifd):
                val = line.strip()
                data[i][name] = lookup.setdefault(val, len(lookup) + 1)
                
    random.shuffle(data)
    train_data = data[:int(len(data) * 0.95)]
    dev_data = data[int(len(data) * 0.95):]
    logging.info("Read %d sentence pairs, %d train, %d dev", len(data), len(train_data), len(dev_data))

    for name in task_names:
        logging.info("Registering task '%s' of shape %s", name, len(dev_data))
        register_classification_targets(args.host, args.port, name="Classify {}".format(name), targets=[x[name] for x in dev_data], model_pattern="OpenNMT")
        register_clustering_targets(args.host, args.port, name="Cluster {}".format(name), targets=[x[name] for x in dev_data], model_pattern="OpenNMT")
        #register_classification_targets(args.host, args.port, name="Classify {}".format(name), targets=[x[name] for x in train_data], model_pattern="OpenNMT")
        #register_clustering_targets(args.host, args.port, name="Cluster {}".format(name), targets=[x[name] for x in train_data], model_pattern="OpenNMT")        
        
    os.mkdir(os.path.join(temp, "data"))
    
    with open(os.path.join(temp, "data", "train_source.txt"), "wt") as sofd, open(os.path.join(temp, "data", "train_target.txt"), "wt") as tofd:
        sofd.write("\n".join([x["source"] for x in train_data]) + "\n")
        tofd.write("\n".join([x["target"] for x in train_data]) + "\n")
            
    with open(os.path.join(temp, "data", "dev_source.txt"), "wt") as sofd, open(os.path.join(temp, "data", "dev_target.txt"), "wt") as tofd:
        sofd.write("\n".join([x["source"] for x in dev_data]) + "\n")
        tofd.write("\n".join([x["target"] for x in dev_data]) + "\n")
        
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
    preproc_args.shuffle = 0
    preproc_args.src_seq_length = source_max
    preproc_args.tgt_seq_length = target_max
    
    train_parser = argparse.ArgumentParser(
        description='vivisect example',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    opts.add_md_help_argument(train_parser)
    opts.model_opts(train_parser)
    opts.train_opts(train_parser)
    train_args = train_parser.parse_args(["-data", os.path.join(temp, "data/out"),
                                          "-train_steps", str(args.epochs - 1),
                                          "-save_model", os.path.join(temp, "model"),
                                          "-enc_layers", "3",
                                          "-dec_layers", "3",
                                          "-rnn_size", "50",
                                          "-src_word_vec_size", "25",
                                          "-tgt_word_vec_size", "25",
    ])

    train_args.batch_size = 50
    #print(train_args)
    #sys.exit()
    try:
        torch.manual_seed(preproc_args.seed)
        
        opt = preproc_args
        logger = logging

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

        opt = train_args
        opt = training_opt_postprocessing(opt)

        model_opt = opt
        checkpoint = None
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
        remove(args.host, args.port, "OpenNMT")

        probe(args.name, model, args.host, args.port, when=lambda m, o : m._v.state == "dev", which=lambda m, o : True, #o._v.operation_name in ["encoder", "decoder"],
              parameters=False, forward=True, backward=False, batch_axis=1)

        opt.report_every = 1
        # Build optimizer.
        optim = build_optim(model, opt, checkpoint)

        trainer = build_trainer(opt, model, fields, optim, "text")

        def train_iter_fct():
            return build_dataset_iter(lazily_load_dataset("train", opt), fields, opt)

        def valid_iter_fct():
            return build_dataset_iter(lazily_load_dataset("valid", opt), fields, opt)

        #print(args.epochs)
        # Do training.
        trainer.train(train_iter_fct, valid_iter_fct, args.epochs, 1)

    
    except Exception as e:
        raise e
    finally:
        flush(args.host, args.port)
        shutil.rmtree(temp)
