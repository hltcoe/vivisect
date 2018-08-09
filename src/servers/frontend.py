import sqlite3
import logging
from plotly.offline import plot
from plotly.graph_objs import Scatter, Figure, Layout
from flask import Flask, request, send_file, render_template_string
import pkg_resources as pr
import jinja2


model_template = jinja2.Template("""
<html>
  <head>
    <style>
      .click td { display : block }
    </style>
  </head>
  <body>
    <a href='/'>
      <img height=100px src='/logo'/>
    </a>
    <table>
      <tr>
        <th>Metric</th>
        {% for array_type, array_type_name in array_types %}
        <th> {{ array_type_name }} </th>
        {% endfor %}
        <th/>
      </tr>
      {% for model_name in model_names %}
      <tr><th colspan="7" align="right">{{ model_name }}</th></tr>
        {% for metric_name in metric_names %}
          {% if (model_name, metric_name) in metric_presence %}
          <tr>
          <th>{{ metric_name }}</th>
          {% for array_type, array_type_name in array_types %}            
            {% if (model_name,metric_name,array_type) in plots %}
            <td bgcolor="green" onclick="document.location = '/{{ model_name }}/{{ metric_name }}/{{ array_type }}'" />
            {% else %}
            <td bgcolor="grey" />
            {% endif %}
          {% endfor %}
          </tr>
          {% endif %}
        {% endfor %}
      {% endfor %}
    </table>
  </body>
</html>
""")


class Frontend(Flask):
    
    def __init__(self, database_file, level):
        super(Frontend, self).__init__("Frontend")
        self.logger.setLevel(level)
        self.database_file = database_file
        self._conn = sqlite3.connect(self.database_file, check_same_thread=False)
        self._cur = self._conn.cursor()
        self._create_db()
        self.logo = pr.resource_filename("vivisect", "logo.png")
        
        @self.route("/logo", methods=["GET"])
        def logo():
            return send_file(self.logo, mimetype="image/png")
        
        @self.route("/clear", methods=["POST"])
        def clear():
            self._create_db(overwrite=True)
            return "OK"

        @self.route("/remove", methods=["POST"])
        def remove():
            j = request.get_json()
            self.logger.info("Removing '%s'", j["model_name"])
            self._cur.execute('''DELETE FROM metrics WHERE model_name=?''', (j["model_name"],))
            return "OK"
        
        @self.route("/", methods=["GET", "POST"])
        def top_level():
            if request.method == "GET":
                plots = set([m for m in self._cur.execute('SELECT DISTINCT model_name,metric_name,array_type from metrics')])
                model_names = sorted(set([x[0] for x in plots]))
                metric_names = sorted(set([x[1] for x in plots]))
                array_types = [("parameter", "Parameter"),
                               ("activation_input", "Activation input"),
                               ("activation_output", "Activation output"),
                               ("gradient_input", "Gradient input"),
                               ("gradient_output", "Gradient output")]
                metric_presence = set()
                for model, metric, _ in plots:
                    metric_presence.add((model, metric))
                return model_template.render(plots=plots, metric_presence=metric_presence, model_names=model_names, metric_names=metric_names, array_types=array_types)
            elif request.method == "POST":
                j = request.get_json()
                self.logger.info("%s", j)
                for field in [k for k, v in j["metadata"].items() if isinstance(v, (str, int, float, bool))]:
                    if field not in self._fields:
                        self._cur.execute("ALTER TABLE metrics ADD COLUMN {}".format(field))
                        self._conn.commit()
                        self._fields.append(field)
                self._cur.execute('''INSERT INTO metrics VALUES ({})'''.format(", ".join(["?"] * len(self._fields))),
                                  [j["metadata"].get(k, "") for k in self._fields])
                if j["metadata"]["metric_type"] == "classification":
                    print(j["metadata"]["metric_name"], j["metadata"]["epoch"])
                self._conn.commit()
                return "OK"
            
        # @self.route("/<string:model_name>", methods=["GET", "POST"])
        # def model_metrics(model_name):
        #     plots = {}
        #     for m, a in set([m for m in self._cur.execute('SELECT metric_name,array_type from metrics where model_name=?', (model_name,))]):
        #         plots[m] = plots.get(m, set())
        #         plots[m].add(a)
        #     return chart_template.render(model_name=model_name, plots=plots)
        
        @self.route("/<string:model_name>/<string:metric_name>/<string:array_type>", methods=["GET"])
        def model_metric_ops(model_name, metric_name, array_type):
            group_by = ["operation_name", "array_name", "array_type"]
            x_axis = "epoch"
            y_axis = "metric_value"
            vals = [{k : v if v else "" for k, v in zip(self._fields, x)} for x in self._cur.execute('SELECT * from metrics where model_name=? and metric_name=? and array_type=?', (model_name, metric_name, array_type))]
            line_tuples = sorted(set([tuple([v[k] for k in group_by]) for v in vals]))
            plots = []
            for line in line_tuples:
                pairs = [(k, v) for k, v in zip(group_by, line) if v != ""]
                points = sorted([x for x in self._cur.execute("SELECT {},{} from metrics where model_name=? and metric_name=? and {}".format(x_axis, y_axis, " and ".join(["{}=?".format(g) for g, _ in pairs])), (model_name, metric_name, *[v for _, v in pairs]))])
                plots.append(Scatter(x=[x[0] for x in points], y=[x[1] for x in points], mode="lines", name="_".join([v for _, v in pairs])))
            plots_html = plot({"data" : plots, "layout" : Layout(title="<a href='/{0}'>{0}</a> - {1} for {2}s".format(model_name, metric_name, array_type), titlefont={"size" : 30})}, output_type="div")
            html = "<html><body><a href='/'><img height=100px src='/logo'/></a>{0}</body></html>".format(plots_html).encode()
            return html
        
    def _create_db(self, overwrite=False):
        """
        Create the "metrics" database with fields "metric_name", "metric_type", and "metric_value".
        If a "metrics" table already exists, only drop it if overwrite==True.
        """
        current = list(self._cur.execute("select * from sqlite_master where type='table' and name='metrics'"))
        if overwrite and len(current) >= 1:
            self._cur.execute('''DROP TABLE IF EXISTS metrics''')
            self._conn.commit()
        elif len(current) >= 1:
            self._fields = [x[1] for x in sorted(self._cur.execute('''PRAGMA table_info(metrics)'''))]
            return None
        self._cur.execute('''CREATE TABLE metrics (model_name text, operation_name text, metric_name text, metric_type text, metric_value real)''')
        self._fields = ["model_name", "operation_name", "metric_name", "metric_type", "metric_value"]
        self._conn.commit()
            

def create_server(database_file=":memory:"):
    return Frontend(database_file, logging.INFO)
