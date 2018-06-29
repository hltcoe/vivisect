from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
import json

def flush(host, port):
    r = Request("http://{}:{}/flush".format(host, port), method="POST")
    urlopen(r)

def clear(host, port):
    r = Request("http://{}:{}/clear".format(host, port), method="POST")
    urlopen(r)    
