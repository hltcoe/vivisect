import argparse
import logging
from vivisect.servers import Aggregator

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", dest="host", default="0.0.0.0", help="Host name")
    parser.add_argument("--port", dest="port", default=8082, type=int, help="Port number")
    parser.add_argument("--eval_host", dest="eval_host", default="0.0.0.0", help="Host name")
    parser.add_argument("--eval_port", dest="eval_port", default=8081, type=int, help="Port number")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    server = Aggregator((args.host, args.port), args.eval_host, args.eval_port)
    logging.info("Starting aggregator on {}:{}".format(args.host, args.port))
    server.serve_forever()
