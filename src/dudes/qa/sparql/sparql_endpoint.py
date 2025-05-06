import logging
import random
import threading
import time
import traceback
import urllib
from typing import Optional, Dict, Any

import rdflib
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLWrapper2
from aiohttp.web_exceptions import HTTPError
from rdflib import Graph
from rdflib.namespace import NamespaceManager

from dudes import consts
from lemon import namespaces


class SPARQLEndpoint:
    def __init__(self, endpoint: str = consts.sparql_endpoint, default_graph=None, cache: Optional[Dict[str, Dict[str, Any]]] = None, check_endpoint=True):
        self.endpoint = endpoint
        self.cache = cache
        self.lock = threading.Lock()
        self.nsmanager = NamespaceManager(Graph())
        for name, ns in namespaces.default_namespaces:
            self.nsmanager.bind(name, ns, override=True, replace=True)

        self.default_graph = default_graph

        if not endpoint.startswith("http"):
            self.endpoint_graph = rdflib.Graph().parse(endpoint)
        else:
            self.endpoint_graph = None
            if check_endpoint:
                endpoint_up = urllib.request.urlopen(self.endpoint).getcode()
                if endpoint_up != 200:
                    raise RuntimeError(f"SPARQL endpoint {self.endpoint} is not up")

    @staticmethod
    def sanitize_sparql_result(res):
        if res is None:
            return None
        elif True not in res and False not in res:
            return set([item["value"] for r in res for item in list(r.values())])
        else:
            return set(res)


    def get_results_query(self, query: str, raw=False, check_query=True):
        #for i in range(10):
        if not isinstance(query, str):
            logging.error(f"Query is not a string: {query}")
            return set()
        query = query.removeprefix("```sparql").removesuffix("```")

        if check_query:
            try:  # check for syntax errors
                qresult = rdflib.Graph(namespace_manager=self.nsmanager).query(query)
            except Exception as e:
                logging.error(f"Query is not valid: {query} {e}")
                return set()

        if self.endpoint_graph is not None:
            if raw:
                logging.error("Raw results not supported for local SPARQL endpoints")
            with self.lock:
                if self.cache is not None and self.endpoint in self.cache.keys() and query in self.cache[self.endpoint].keys():
                    return self.cache[self.endpoint][query]

                sani_result = None
                try:
                    qresult = self.endpoint_graph.query(query)
                    if qresult.type == "ASK":
                        sani_result = {bool(qresult)}
                    elif qresult.type == "SELECT":
                        qresult = list(qresult)
                        sani_result = {str(v) for r in qresult for v in r}
                    else:
                        raise RuntimeError(f"Query type {qresult.type} not supported")
                except Exception as e:
                    logging.error(f"Error in SPARQL endpoint: {e}")
                    raise e

                if self.cache is not None and sani_result is not None:
                    if self.endpoint not in self.cache.keys():
                        self.cache[self.endpoint] = dict()
                    self.cache[self.endpoint][query] = sani_result

                return sani_result
        else:
            if "PREFIX" not in query:
                prefixes = ""
                for prefix, ns in self.nsmanager.namespaces():
                    prefixes += f"PREFIX {prefix}: <{ns}>\n"
                query = prefixes + query

            #logging.debug(query)
            fails = 0
            while True:
                try:
                    with self.lock:
                        if self.cache is not None and self.endpoint in self.cache.keys() and query in self.cache[self.endpoint].keys():
                            return self.cache[self.endpoint][query]

                    sparql = SPARQLWrapper(self.endpoint, defaultGraph=self.default_graph)
                    sparql.setQuery(query)
                    sparql.setReturnFormat(JSON)

                    ret = None
                    # try:
                    ret = sparql.queryAndConvert()
                    if not hasattr(ret, "keys"):
                        logging.error(f"SPARQL endpoint returned invalid result: {ret} {type(ret)}")
                        return set()
                    elif "results" in ret.keys() and "bindings" in ret["results"].keys():
                        ret = ret["results"]["bindings"]
                    elif "boolean" in ret.keys():
                        ret = [ret["boolean"]]

                    if not raw:
                        ret = self.sanitize_sparql_result(ret)

                    with self.lock:
                        if self.cache is not None and ret is not None:
                            if self.endpoint not in self.cache.keys():
                                self.cache[self.endpoint] = dict()
                            self.cache[self.endpoint][query] = ret

                    # print(ret)

                    if ret is None:
                        ret = set()

                    # for r in ret["results"]["bindings"]:
                    #    print(r)
                    return ret
                    # except Exception as e:
                    #    print(e)
                except HTTPError as e:
                    if e.status == 429:
                        secs = random.uniform(1.0, 5.0)
                        logging.error(f"429 error, sleeping for {secs} seconds")
                        time.sleep(secs)
                        continue
                    else:
                        secs = random.uniform(1.0, 5.0)
                        logging.error(f"{e.status} error, sleeping for {secs} seconds")
                        time.sleep(secs)
                        if fails > 10:
                            raise e
                        else:
                            fails += 1
                            continue
                except OSError as e:
                    secs = random.uniform(1.0, 5.0)
                    logging.error(f"OSError {e} error, sleeping for {secs} seconds")
                    time.sleep(secs)
                    if fails > 10:
                        raise e
                    else:
                        fails += 1
                        continue
                except Exception as e:
                   logging.debug(f"Error in SPARQL endpoint: {traceback.format_exc()}")
                   logging.error(f"Error in SPARQL endpoint: {e}")
                   raise e
            raise RuntimeError("Failed to get results from SPARQL endpoint, too many 429 errors!")

    def ask_triple(self, t):
        prefixes = ""
        for prefix, ns in self.nsmanager.namespaces():
            prefixes += f"PREFIX {prefix}: <{ns}>\n"
        if isinstance(t, str):
            return True in self.get_results_query(prefixes + f"ASK WHERE {{ {t} }}")
        else:
            return True in self.get_results_query(prefixes + f"ASK WHERE {{ {t.get_text()} }}")
