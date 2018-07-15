#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from sockeye.model import *
from sockeye.train import *
from sockeye.training import *
from sockeye.arguments import *
from vivisect import probe, flush, clear
import warnings
warnings.simplefilter(action='ignore')
import tempfile
import shutil
import gzip
from types import MethodType


# class ModifiedTrainer(training.EarlyStoppingTrainer):
#     def __init__(self, *args, **argdict):
#         super(self, ModifiedTrainer).__init__(*args, **argdict)

#     def fit(self, *args, **argdict):
#         pass







if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", dest="host", default="0.0.0.0", help="Host name")
    parser.add_argument("--port", dest="port", default=8082, type=int, help="Port number")
    parser.add_argument("--source", dest="source")
    parser.add_argument("--target", dest="target")
    parser.add_argument("--clear", dest="clear", action="store_true", default=False, help="Clear the database first")
    parser.add_argument("--epochs", dest="epochs", default=10, type=int)
    args, rest = parser.parse_known_args()    

    if args.clear:
        clear(args.host, args.port)   
    temp = tempfile.mkdtemp()

    try:
        
        source = []
        with gzip.open(args.source, "rt") as ifd:
                for line in ifd:
                    source.append(line)

        target = []
        with gzip.open(args.target, "rt") as ifd:
                for line in ifd:
                    target.append(line)

        pairs = list(zip(source, target))
        random.shuffle(pairs)
        
        with open(os.path.join(temp, "train_source.txt"), "wt") as sofd, open(os.path.join(temp, "train_target.txt"), "wt") as tofd:
            for s, t in pairs[0:int(.9 * len(pairs))]:
                sofd.write(s)
                tofd.write(t)

        with open(os.path.join(temp, "dev_source.txt"), "wt") as sofd, open(os.path.join(temp, "dev_target.txt"), "wt") as tofd:
            for s, t in pairs[int(.9 * len(pairs)):]:
                sofd.write(s)
                tofd.write(t)

        params = arguments.ConfigArgumentParser(description='Train Sockeye sequence-to-sequence models.')
        arguments.add_train_cli_args(params)
        train_args = params.parse_args(args=["-o", os.path.join(temp, "out"),
                                             "--source", os.path.join(temp, "train_source.txt"),
                                             "--target", os.path.join(temp, "train_target.txt"),
                                             "-vs", os.path.join(temp, "dev_source.txt"),
                                             "-vt", os.path.join(temp, "dev_target.txt"),
                                             "--use-cpu",
                                             "--max-num-epochs", str(args.epochs),
                                             "--num-layers", "2:2",
                                             "--encoder", "rnn",
                                             "--decoder", "rnn",
                                             "--rnn-num-hidden", "500",
                                             "--num-embed", "500:500",
            ])
        
        utils.seedRNGs(train_args.seed)

        check_arg_compatibility(train_args)
        output_folder = os.path.abspath(train_args.output)
        resume_training = check_resume(train_args, output_folder)

        global logger
        logger = setup_main_logger(__name__,
                                   file_logging=True,
                                   console=not train_args.quiet, path=os.path.join(output_folder, C.LOG_NAME))
        utils.log_basic_info(train_args)
        arguments.save_args(train_args, os.path.join(output_folder, C.ARGS_STATE_NAME))

        max_seq_len_source, max_seq_len_target = train_args.max_seq_len
        # The maximum length is the length before we add the BOS/EOS symbols
        max_seq_len_source = max_seq_len_source + C.SPACE_FOR_XOS
        max_seq_len_target = max_seq_len_target + C.SPACE_FOR_XOS
        logger.info("Adjusting maximum length to reserve space for a BOS/EOS marker. New maximum length: (%d, %d)",
                    max_seq_len_source, max_seq_len_target)

        with ExitStack() as exit_stack:
            context = determine_context(train_args, exit_stack)

            train_iter, eval_iter, config_data, source_vocabs, target_vocab = create_data_iters_and_vocabs(
                args=train_args,
                max_seq_len_source=max_seq_len_source,
                max_seq_len_target=max_seq_len_target,
                shared_vocab=use_shared_vocab(train_args),
                resume_training=resume_training,
                output_folder=output_folder)
            max_seq_len_source = config_data.max_seq_len_source
            max_seq_len_target = config_data.max_seq_len_target

            # Dump the vocabularies if we're just starting up
            if not resume_training:
                vocab.save_source_vocabs(source_vocabs, output_folder)
                vocab.save_target_vocab(target_vocab, output_folder)

            source_vocab_sizes = [len(v) for v in source_vocabs]
            target_vocab_size = len(target_vocab)
            logger.info('Vocabulary sizes: source=[%s] target=%d',
                        '|'.join([str(size) for size in source_vocab_sizes]),
                        target_vocab_size)

            model_config = create_model_config(args=train_args,
                                               source_vocab_sizes=source_vocab_sizes, target_vocab_size=target_vocab_size,
                                               max_seq_len_source=max_seq_len_source, max_seq_len_target=max_seq_len_target,
                                               config_data=config_data)
            model_config.freeze()

            training_model = create_training_model(config=model_config,
                                                        context=context,
                                                        output_dir=output_folder,
                                                        train_iter=train_iter,
                                                        args=train_args)

            # Handle options that override training settings
            min_updates = train_args.min_updates
            max_updates = train_args.max_updates
            min_samples = train_args.min_samples
            max_samples = train_args.max_samples
            max_num_checkpoint_not_improved = train_args.max_num_checkpoint_not_improved
            min_epochs = train_args.min_num_epochs
            max_epochs = train_args.max_num_epochs
            if min_epochs is not None and max_epochs is not None:
                check_condition(min_epochs <= max_epochs,
                                "Minimum number of epochs must be smaller than maximum number of epochs")
            # Fixed training schedule always runs for a set number of updates
            if train_args.learning_rate_schedule:
                min_updates = None
                max_updates = sum(num_updates for (_, num_updates) in train_args.learning_rate_schedule)
                max_num_checkpoint_not_improved = -1
                min_samples = None
                max_samples = None
                min_epochs = None
                max_epochs = None

            training_model.module._vivisect = {"model_name" : "Sockeye model", "iteration" : 0, "framework" : "mxnet"}
            probe(training_model.module, args.host, args.port)

            trainer = training.EarlyStoppingTrainer(model=training_model,
                                                    optimizer_config=create_optimizer_config(train_args, source_vocab_sizes),
                                                    max_params_files_to_keep=train_args.keep_last_params,
                                                    source_vocabs=source_vocabs,
                                                    target_vocab=target_vocab)

            trainer.__step = trainer._step
            def callback(self, *args, **argdict):
                retval = self.__step(*args, **argdict)
                self.model.module._vivisect["iteration"] = self.state.epoch + 1
                return retval
            
            trainer._step = MethodType(callback, trainer)
            
            trainer.fit(train_iter=train_iter,
                        validation_iter=eval_iter,
                        early_stopping_metric=train_args.optimized_metric,
                        metrics=train_args.metrics,
                        checkpoint_frequency=train_args.checkpoint_frequency,
                        max_num_not_improved=max_num_checkpoint_not_improved,
                        min_samples=min_samples,
                        max_samples=max_samples,
                        min_updates=min_updates,
                        max_updates=max_updates,
                        min_epochs=min_epochs,
                        max_epochs=max_epochs,
                        lr_decay_param_reset=train_args.learning_rate_decay_param_reset,
                        lr_decay_opt_states_reset=train_args.learning_rate_decay_optimizer_states_reset,
                        decoder=create_checkpoint_decoder(train_args, exit_stack, context),
                        mxmonitor_pattern=train_args.monitor_pattern,
                        mxmonitor_stat_func=train_args.monitor_stat_func,
                        allow_missing_parameters=train_args.allow_missing_params or model_config.lhuc,
                        existing_parameters=train_args.params)
            
    except Exception as e:
        raise e
    finally:
        flush(args.host, args.port)
        shutil.rmtree(temp)
