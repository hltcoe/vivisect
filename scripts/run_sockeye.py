import argparse
from sockeye.model import *
from sockeye.train import *
from sockeye.training import *
from sockeye.arguments import *
from vivisect.mxnet import probe

if __name__ == "__main__":

    params = arguments.ConfigArgumentParser(description='Train Sockeye sequence-to-sequence models.')
    arguments.add_train_cli_args(params)
    args = params.parse_args()
    
    if args.dry_run:
        # Modify arguments so that we write to a temporary directory and
        # perform 0 training iterations
        temp_dir = tempfile.TemporaryDirectory()  # Will be automatically removed
        args.output = temp_dir.name
        args.max_updates = 0

    utils.seedRNGs(args.seed)

    check_arg_compatibility(args)
    output_folder = os.path.abspath(args.output)
    resume_training = check_resume(args, output_folder)

    global logger
    logger = setup_main_logger(__name__,
                               file_logging=True,
                               console=not args.quiet, path=os.path.join(output_folder, C.LOG_NAME))
    utils.log_basic_info(args)
    arguments.save_args(args, os.path.join(output_folder, C.ARGS_STATE_NAME))

    max_seq_len_source, max_seq_len_target = args.max_seq_len
    # The maximum length is the length before we add the BOS/EOS symbols
    max_seq_len_source = max_seq_len_source + C.SPACE_FOR_XOS
    max_seq_len_target = max_seq_len_target + C.SPACE_FOR_XOS
    logger.info("Adjusting maximum length to reserve space for a BOS/EOS marker. New maximum length: (%d, %d)",
                max_seq_len_source, max_seq_len_target)

    with ExitStack() as exit_stack:
        context = determine_context(args, exit_stack)

        train_iter, eval_iter, config_data, source_vocabs, target_vocab = create_data_iters_and_vocabs(
            args=args,
            max_seq_len_source=max_seq_len_source,
            max_seq_len_target=max_seq_len_target,
            shared_vocab=use_shared_vocab(args),
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

        model_config = create_model_config(args=args,
                                           source_vocab_sizes=source_vocab_sizes, target_vocab_size=target_vocab_size,
                                           max_seq_len_source=max_seq_len_source, max_seq_len_target=max_seq_len_target,
                                           config_data=config_data)
        model_config.freeze()

        training_model = create_training_model(config=model_config,
                                                    context=context,
                                                    output_dir=output_folder,
                                                    train_iter=train_iter,
                                                    args=args)

        # Handle options that override training settings
        min_updates = args.min_updates
        max_updates = args.max_updates
        min_samples = args.min_samples
        max_samples = args.max_samples
        max_num_checkpoint_not_improved = args.max_num_checkpoint_not_improved
        min_epochs = args.min_num_epochs
        max_epochs = args.max_num_epochs
        if min_epochs is not None and max_epochs is not None:
            check_condition(min_epochs <= max_epochs,
                            "Minimum number of epochs must be smaller than maximum number of epochs")
        # Fixed training schedule always runs for a set number of updates
        if args.learning_rate_schedule:
            min_updates = None
            max_updates = sum(num_updates for (_, num_updates) in args.learning_rate_schedule)
            max_num_checkpoint_not_improved = -1
            min_samples = None
            max_samples = None
            min_epochs = None
            max_epochs = None

        training_model.module._vivisect = {"model_id" : "Sockeye model", "iteration" : 0}
        probe(training_model.module, "localhost", 8082)
        
        trainer = training.EarlyStoppingTrainer(model=training_model,
                                                optimizer_config=create_optimizer_config(args, source_vocab_sizes),
                                                max_params_files_to_keep=args.keep_last_params,
                                                source_vocabs=source_vocabs,
                                                target_vocab=target_vocab)
        
        trainer.fit(train_iter=train_iter,
                    validation_iter=eval_iter,
                    early_stopping_metric=args.optimized_metric,
                    metrics=args.metrics,
                    checkpoint_frequency=args.checkpoint_frequency,
                    max_num_not_improved=max_num_checkpoint_not_improved,
                    min_samples=min_samples,
                    max_samples=max_samples,
                    min_updates=min_updates,
                    max_updates=max_updates,
                    min_epochs=min_epochs,
                    max_epochs=max_epochs,
                    lr_decay_param_reset=args.learning_rate_decay_param_reset,
                    lr_decay_opt_states_reset=args.learning_rate_decay_optimizer_states_reset,
                    decoder=create_checkpoint_decoder(args, exit_stack, context),
                    mxmonitor_pattern=args.monitor_pattern,
                    mxmonitor_stat_func=args.monitor_stat_func,
                    allow_missing_parameters=args.allow_missing_params or model_config.lhuc,
                    existing_parameters=args.params)
