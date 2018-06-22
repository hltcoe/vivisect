import argparse
import logging
from vivisect.servers import Evaluator

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", dest="host", default="0.0.0.0", help="Host name")
    parser.add_argument("--port", dest="port", default=8081, type=int, help="Port number")
    parser.add_argument("--frontend_host", dest="frontend_host", default="0.0.0.0", help="Host name")
    parser.add_argument("--frontend_port", dest="frontend_port", default=8080, type=int, help="Port number")    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    server = Evaluator((args.host, args.port), args.frontend_host, args.frontend_port)
    logging.info("Starting evaluator on {}:{}".format(args.host, args.port))
    server.serve_forever()
