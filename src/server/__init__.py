import sqlite3
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import TCPServer
import json
import logging
from urllib.parse import urlparse, parse_qs
import plotnine


class VivisectHandler(BaseHTTPRequestHandler):

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
        #query = parse_qs(urlparse(self.path).query)
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        print(self.headers)
        self.wfile.write("<html><body><h1>Vivisect server</h1></body></html>".encode())        

    def do_POST(self):
        logging.info("Processing POST request")
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode()
        j = json.loads(post_data)
        #print(j)
        if j["type"] == "LAYER":
            logging.info("Payload: {}".format(j["metadata"]))
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
        elif j["type"] == "PLOT":
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"<html><body><h1>POST</h1></body></html>")

            
class VivisectServer(HTTPServer):
    def __init__(self, address, db_file):
        super(VivisectServer, self).__init__(address, VivisectHandler)
        self.conn_ = sqlite3.connect(db_file)
        self.cur_ = self.conn_.cursor()
