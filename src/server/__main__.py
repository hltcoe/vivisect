import argparse
import logging
from . import VivisectServer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", dest="host", default="0.0.0.0", help="Host name")
    parser.add_argument("--port", dest="port", default=39628, type=int, help="Port number")
    parser.add_argument("--database", dest="database", default=":memory:", help="Database file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    server = VivisectServer((args.host, args.port), args.database)
    logging.info("Starting server on {}:{}".format(args.host, args.port))
    server.serve_forever()
