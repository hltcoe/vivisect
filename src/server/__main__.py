import argparse
import sqlite3
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import TCPServer
import json
import logging


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
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"<html><body><h1>GET</h1></body></html>")

    def do_POST(self):
        logging.info("Processing POST request")
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode()
        j = json.loads(post_data)
        logging.info("Payload: {}".format(j["metadata"]))
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"<html><body><h1>POST</h1></body></html>")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", dest="host", default="0.0.0.0", help="Host name")
    parser.add_argument("--port", dest="port", default=39628, type=int, help="Port number")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    server = HTTPServer((args.host, args.port), VivisectHandler)
    logging.info("Starting server on {}:{}".format(args.host, args.port))
    server.serve_forever()
