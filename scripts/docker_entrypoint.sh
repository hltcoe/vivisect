#!/bin/bash

if [ "$1" == "aggregator" ]
then
    FLASK_APP="src/servers/aggregator.py:create_server()" flask run --port 8080 --reload
elif [ "$1" == "evaluator" ]
then
    FLASK_APP="src/servers/evaluator.py:create_server()" flask run --port 8080 --reload
elif [ "$1" == "frontend" ]
then
    FLASK_APP="src/servers/frontend.py:create_server('test.sql')" flask run --port 8080 --reload
elif [ "$1" == "simple_examples" ]
then
    python scripts/run_examples.py --host aggregator --port 8080 --epochs 50
elif [ "$1" == "sockeye_example" ]
then
    python scripts/run_sockeye.py -o out --source temp/corpus.tc.de.small --target temp/corpus.tc.en.small -vs temp/corpus.tc.de.small -vt temp/corpus.tc.en.small --use-cpu --num-embed 32:32 --rnn-num-hidden 32 --cnn-num-hidden 32 --num-layers 2 --transformer-model-size 32 --max-num-epochs 1 --batch-size 100
elif [ "$1" == "opennmt_example" ]
then
    python scripts/run_opennmt.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo
else
    echo "Available commands: (frontend|evaluator|aggregator|simple_examples|sockeye_example|opennmt_example)"
fi
