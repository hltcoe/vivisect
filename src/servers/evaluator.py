import sqlite3
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import TCPServer
import json
import logging
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen


class EvaluatorHandler(BaseHTTPRequestHandler):

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
        logging.info("POST metadata: %s", j["metadata"])
        val = {"metadata" : {k : v for k, v in j["metadata"].items()}}
        val["metric_name"] = "constant"
        val["metric_value"] = 1.0
        r = Request("http://{}:{}".format(self.server.frontend_host, self.server.frontend_port), method="POST", data=json.dumps(val).encode())
        urlopen(r)
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

            
class Evaluator(HTTPServer):
    def __init__(self, address, frontend_host, frontend_port):
        super(Evaluator, self).__init__(address, EvaluatorHandler)
        self.frontend_host = frontend_host
        self.frontend_port = frontend_port

