import sqlite3
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import TCPServer
import json
import logging
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
from plotly.offline import plot
from plotly.graph_objs import Scatter


class FrontendHandler(BaseHTTPRequestHandler):

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
        self.path = self.path.replace('%20', " ")
        path_elems = self.path[1:].split("/")
        if self.path == "/":
            model_ids = set([m[0] for m in self.server.cur_.execute('''SELECT model_id from metrics''')])        
            models = "\n".join(["<h2><a href=\"/{}\">{}</a></h2>".format(m, m) for m in model_ids])
            html = "<html><body><h1>Vivisect server</h1>{}</body></html>".format(models).encode()
        elif len(path_elems) == 1:
            model_id = path_elems[0]
            print(model_id)
            metric_names = set([m[0] for m in self.server.cur_.execute('SELECT metric_name from metrics where model_id=?', (model_id,))])
            metrics = "\n".join(["<h2><a href=\"/{}/{}\">{}</a></h2>".format(model_id, m, m) for m in metric_names])
            html = "<html><body><h1>Vivisect server</h1><h2>Model {}</h2>{}</body></html>".format(model_id, metrics).encode()
        elif len(path_elems) == 2:
            model_id, metric_name = path_elems
            vals = sorted([x for x in self.server.cur_.execute('SELECT iteration,op_name,metric_value from metrics where model_id=? and metric_name=?', (model_id,metric_name))])
            op_names = sorted(set([o for _, o, m in vals]))
            ops = "\n".join(["<h2><a href=\"/{}/{}/{}\">{}</a></h2>".format(model_id, metric_name, m, m) for m in op_names])
            html = "<html><body><h1>Vivisect server</h1><h2>Model {}, Metric {}</h2>{}</body></html>".format(model_id, metric_name, ops).encode()            
        elif len(path_elems) == 3:
            model_id, metric_name, op_name = path_elems
            vals = sorted([x for x in self.server.cur_.execute('SELECT iteration,metric_value from metrics where model_id=? and metric_name=? and op_name=?', (model_id,metric_name,op_name))])
            my_plot = plot([Scatter(x=[x[0] for x in vals], y=[x[1] for x in vals])], output_type='div')
            html = "<html><body><h1>Vivisect server</h1><h2>Model {}, Metric {}, Layer {}</h2>{}</body></html>".format(model_id, metric_name, op_name, my_plot).encode()
            pass
        else:
            html = "".encode()
        self.wfile.write(html)            

    def do_POST(self):
        logging.info("Processing POST request")
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode()
        j = json.loads(post_data)
        model_id = j["metadata"]["model_id"]
        op_name = j["metadata"]["op_name"]
        metric_name = j["metric_name"]
        metric_value = j["metric_value"]        
        logging.info("POST metadata: %s %s %s", metric_name, metric_value, j["metadata"])
        iteration = j["metadata"]["iteration"]
        self.server.cur_.execute('''INSERT INTO metrics VALUES (?, ?, ?, ?, ?)''', (metric_name, metric_value, iteration, model_id, op_name))
        self.server.conn_.commit()
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

            
class Frontend(HTTPServer):
    def __init__(self, address, db_file):
        super(Frontend, self).__init__(address, FrontendHandler)
        self.conn_ = sqlite3.connect(db_file)
        self.cur_ = self.conn_.cursor()
        try:
            self.cur_.execute('''CREATE TABLE metrics (metric_name text, metric_value real, iteration int, model_id text, op_name text)''')
        except:
            pass
