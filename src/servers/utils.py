from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
import json

def flush(host, port):
    r = Request("http://{}:{}".format(host, port), method="POST", data=json.dumps({"command" : "flush"}).encode())
    urlopen(r)
