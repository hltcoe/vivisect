import sqlite3
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import TCPServer
import json
import logging
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen

class AggregatorHandler(BaseHTTPRequestHandler):

    # self.server
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_HEAD(self):
        logging.info("Processing HEAD request")
        self._set_headers()        
    
    def do_GET(self):
        logging.info("Processing GET request")
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        print(self.headers)
        self.wfile.write("<html><body><h1>Vivisect evaluator</h1></body></html>".encode())        

    def do_POST(self):
        logging.info("Processing POST request")
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode()        
        j = json.loads(post_data)
        if j.get("command") == "flush":
            logging.info("Flushing tables")
            for model_id, iteration in self.server.state.items():
                for op_name, vals in self.server.data[model_id].items():
                    self.server.state[model_id]["op_name"] = op_name                    
                    r = Request("http://{}:{}".format(self.server.eval_host, self.server.eval_port), method="POST", data=json.dumps({"data" : vals, "metadata" : self.server.state[model_id]}).encode())
                    urlopen(r)
            self.server.state = {}
            self.server.data = {}
        else:
            logging.info("POST metadata: %s", j["metadata"])
            model_id = j["metadata"]["model_id"]
            op_name = j["metadata"]["op_name"]
            iteration = j["metadata"]["iteration"]
            if model_id not in self.server.state:
                self.server.state[model_id] = j["metadata"]
                self.server.data[model_id] = {}
            elif self.server.state[model_id]["iteration"] != iteration:
                for op_name, vals in self.server.data[model_id].items():
                    self.server.state[model_id]["op_name"] = op_name
                    r = Request("http://{}:{}".format(self.server.eval_host, self.server.eval_port), method="POST", data=json.dumps({"data" : vals, "metadata" : self.server.state[model_id]}).encode())
                    urlopen(r)
                self.server.state[model_id] = j["metadata"]
                self.server.data[model_id] = {}
            self.server.data[model_id] = self.server.data.get(model_id, {})
            self.server.data[model_id][op_name] = self.server.data[model_id].get(op_name, [])
            self.server.data[model_id][op_name].append((j["inputs"], j["outputs"]))
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

            
class Aggregator(HTTPServer):
    def __init__(self, address, eval_host, eval_port):
        super(Aggregator, self).__init__(address, AggregatorHandler)
        self.eval_host = eval_host
        self.eval_port = eval_port
        self.data = {}
        self.state = {}
