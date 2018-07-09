import sqlite3
import logging
from plotly.offline import plot
from plotly.graph_objs import Scatter, Figure
from flask import Flask, request

            
class Frontend(Flask):
    
    def __init__(self, database_file, level):
        super(Frontend, self).__init__("Frontend")
        self.logger.setLevel(level)
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
                group_name = j["metadata"]["group_name"]
                slot_name = j["metadata"]["slot_name"]                
                metric_name = j["metric_name"]
                metric_value = j["metric_value"]        
                epoch = j["metadata"]["epoch"]
                self._cur.execute('''INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?)''', (model_name, metric_name, op_name, group_name, slot_name, metric_value, epoch))
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
            vals = sorted([x for x in self._cur.execute('SELECT epoch,op_name,group_name,slot_name,metric_value from metrics where model_name=? and metric_name=?', (model_name, metric_name))])
            line_tuples = sorted(set([(o, g, s) for _, o, g, s, _ in vals]))
            plots = []
            for op_name, group_name, slot_name in line_tuples:
                vals = sorted([x for x in self._cur.execute('SELECT epoch,metric_value from metrics where model_name=? and metric_name=? and op_name=? and slot_name=? and group_name=?', (model_name, metric_name, op_name, slot_name, group_name))])
                #total = sum([x[1] for x in vals])
                plots.append(Scatter(x=[x[0] for x in vals], y=[x[1] for x in vals], mode="lines", name="{}_{}_{}".format(op_name, group_name, slot_name)))
            plots_html = plot(plots, output_type="div")
            html = "<html><body><h2><a href='/'>Vivisect server</a>|<a href='/{0}'>{0}</a>|{1}</h2>{2}</body></html>".format(model_name,
                                                                                                                             metric_name,
                                                                                                                             plots_html).encode()
            return html
    
    def reset_db(self, force=False):
        self._conn = sqlite3.connect(self.database_file, check_same_thread=False)
        self._cur = self._conn.cursor()
        if force:
            self._cur.execute('''DROP TABLE IF EXISTS metrics''')
        try:
            self._cur.execute('''CREATE TABLE metrics (model_name, metric_name text, op_name text, group_name text, slot_name text, metric_value real, epoch int)''')
        except:
            pass
            


def create_server(database_file=":memory:"):
    return Frontend(database_file, logging.INFO)
