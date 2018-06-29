import sqlite3
import logging
from plotly.offline import plot
from plotly.graph_objs import Scatter, Figure
from flask import Flask, request

            
class Frontend(Flask):
    
    def __init__(self, database_file):
        super(Frontend, self).__init__("Frontend")
        self.database_file = database_file
        self.reset_db()

        @self.route("/clear", methods=["POST"])
        def clear():
            self.reset_db(force=True)
            return "OK"
        
        @self.route("/", methods=["GET", "POST"])
        def models():
            if request.method == "GET":
                model_names = set([m[0] for m in self._cur.execute('''SELECT model_name from metrics''')])        
                model_html = "\n".join(["<h2><a href=\"/{}\">{}</a></h2>".format(m, m) for m in model_names])
                html = "<html><body><h1>Vivisect server</h1>{}</body></html>".format(model_html).encode()                
                return html
            elif request.method == "POST":
                j = request.get_json()
                model_name = j["metadata"]["model_name"]
                op_name = j["metadata"]["op_name"]
                metric_name = j["metric_name"]
                metric_value = j["metric_value"]        
                iteration = j["metadata"]["iteration"]
                self._cur.execute('''INSERT INTO metrics VALUES (?, ?, ?, ?, ?)''', (model_name, metric_name, op_name, metric_value, iteration))
                self._conn.commit()
                return "OK"
            
        @self.route("/<string:model_name>", methods=["GET", "POST"])
        def model_metrics(model_name):
            metric_names = set([m[0] for m in self._cur.execute('SELECT metric_name from metrics where model_name=?', (model_name,))])
            metric_html = "\n".join(["<h2><a href=\"/{}/{}\">{}</a></h2>".format(model_name, m, m) for m in metric_names])
            html = "<html><body><h2><a href='/'>Vivisect server</a>|{}</h2>{}</body></html>".format(model_name, metric_html).encode()
            return html

        @self.route("/<string:model_name>/<string:metric_name>", methods=["GET"])
        def model_metric_ops(model_name, metric_name):
            vals = sorted([x for x in self._cur.execute('SELECT iteration,op_name,metric_value from metrics where model_name=? and metric_name=?', (model_name, metric_name))])
            op_names = sorted(set([o for _, o, _ in vals]))
            plots = []
            for op_name in op_names:
                vals = sorted([x for x in self._cur.execute('SELECT iteration,metric_value from metrics where model_name=? and metric_name=? and op_name=?', (model_name, metric_name, op_name))])
                plots.append(Scatter(x=[x[0] for x in vals], y=[x[1] for x in vals], mode="lines", name=op_name))
            plots_html = plot(plots, output_type="div")
            html = "<html><body><h2><a href='/'>Vivisect server</a>|<a href='/{0}'>{0}</a>|{1}</h2>{2}</body></html>".format(model_name,
                                                                                                                             metric_name,
                                                                                                                             plots_html).encode()
            return html
    
    def reset_db(self, force=False):
        self._conn = sqlite3.connect(self.database_file, check_same_thread=False)
        self._cur = self._conn.cursor()
        try:
            self._cur.execute('''CREATE TABLE metrics (model_name, metric_name text, op_name text, metric_value real, iteration int)''')
        except:            
            pass


def create_server(database_file=":memory:"):
    return Frontend(database_file)
