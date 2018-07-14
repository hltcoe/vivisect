import argparse
from .frontend import Frontend, create_server as create_frontend_server
from .evaluator import Evaluator, create_server as create_evaluator_server
from .aggregator import Aggregator, create_server as create_aggregator_server

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    
    frontend_parser = subparsers.add_parser("frontend")
    frontend_parser.add_argument("--host", dest="host", default="localhost", help="Host name to serve from")
    frontend_parser.add_argument("--port", dest="port", default=8080, type=int, help="Port number to serve from")
    frontend_parser.add_argument("--database_file", dest="database_file", default=":memory:", help="Sqlite file for storing metrics")
    frontend_parser.set_defaults(func=create_frontend_server)
    
    evaluator_parser = subparsers.add_parser("evaluator")
    evaluator_parser.add_argument("--host", dest="host", default="localhost", help="Host name to serve from")
    evaluator_parser.add_argument("--port", dest="port", default=8081, type=int, help="Port number to serve from")
    evaluator_parser.add_argument("--frontend_host", dest="frontend_host", default="localhost", help="Host name of frontend server")    
    evaluator_parser.add_argument("--frontend_port", dest="frontend_port", default=8080, type=int, help="Port number of frontend server")
    evaluator_parser.set_defaults(func=create_evaluator_server)
        
    aggregator_parser = subparsers.add_parser("aggregator")
    aggregator_parser.add_argument("--host", dest="host", default="localhost", help="Host name to serve from")
    aggregator_parser.add_argument("--port", dest="port", default=8080, type=int, help="Port number to serve from")
    aggregator_parser.add_argument("--evaluator_host", dest="evaluator_host", default="localhost", help="Host name of evaluator server")
    aggregator_parser.add_argument("--evaluator_port", dest="evaluator_port", default=8080, type=int, help="Port number of evaluator server")
    aggregator_parser.set_defaults(func=create_aggregator_server)

    
    args = parser.parse_args()

    f = args.func(**{n : getattr(args, n) for n in dir(args) if not n.startswith("_") and n not in ["host", "port", "func"]})
    f.run(args.host, args.port)
