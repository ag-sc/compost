import argparse
import copy
import csv
import glob
import io
import itertools
import logging
import os
import random
import sys
import json
import traceback
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool, Lock
from multiprocessing.pool import ThreadPool
from pathlib import Path
from pprint import pprint
from queue import Queue
from threading import Thread
from typing import Dict, Any, Optional, List, Set, Tuple

import more_itertools
import networkx as nx
import zstandard
from SPARQLWrapper.SPARQLExceptions import EndPointInternalError, QueryBadFormed
from jsonlines import jsonlines
from matplotlib import pyplot as plt
from more_itertools import random_product
from networkx.classes import DiGraph
from pyparsing import ParseException
from sklearn.model_selection import train_test_split  # type: ignore

import lemon
from dudes import utils, consts
from dudes.qa.sparql.sparql_endpoint import SPARQLEndpoint
from lemon.lemon import SubjOfProp, Reference, PartOfSpeech
from lemon.lemon_parser import LEMONParser
from lemon.lexicon import Lexicon

logging.basicConfig(level=logging.DEBUG, force=True)

class DatasetGenerator:
    def __init__(self, faulty_path: Optional[str] = None):
        if faulty_path is None:
            faulty_path = os.path.join(
                os.path.dirname(sys.modules["lemon"].__file__),
                "resources",
                "faulty_entries.csv"
            )

        self.fieldnames = ["question1", "query1", "question2", "query2", "question3", "query3"]

        lp = LEMONParser.from_ttl_dir()
        lex_entries = lp.parse_nodes(lp.entry_nodes)
        self.lex: Lexicon = Lexicon(lex_entries)


        with open(faulty_path, newline='') as csvfile:
            faulty = set([(r[0], utils.safe_expand_curie(r[1], self.lex.nsmanager)) for r in csv.reader(csvfile)])

        self.entries = [
            e for e in self.lex.entries
            if e is not None
            and len(e.syn_behavior) > 0
            and not isinstance(e.syn_behavior[0], str)
            and e.syn_behavior[0] is not None
            and "lexinfo:NounPPFrame" in e.syn_behavior[0].type
            and e.canonical_form is not None
            and e.canonical_form.written_rep is not None
            and "_" not in e.canonical_form.written_rep
            and e.sense[0] is not None
            and e.syn_behavior[0].prepositional_adjunct is not None
            and e.syn_behavior[0].prepositional_adjunct.marker is not None
            and e.syn_behavior[0].prepositional_adjunct.marker.canonical_form is not None
            and isinstance(e.syn_behavior[0].prepositional_adjunct.marker.canonical_form.written_rep, str)
            and len(e.syn_behavior[0].prepositional_adjunct.marker.canonical_form.written_rep) > 0
            and (e.canonical_form.written_rep, utils.safe_expand_curie(e.sense[0].reference[0], self.lex.nsmanager)) not in faulty
            and utils.safe_expand_curie(e.sense[0].reference[0], self.lex.nsmanager) not in [utils.safe_expand_curie(p, self.lex.nsmanager) for p in consts.compositionality_bad_props]
            and not any([bw in e.canonical_form.written_rep for bw in consts.compositionality_bad_words])
        ]

        self.predicative_type_entries = [
            e for e in self.lex.entries
            if e is not None
            and e.part_of_speech == PartOfSpeech.NOUN
            and e.canonical_form is not None
            and e.canonical_form.written_rep is not None
            and "_" not in e.canonical_form.written_rep
            and e.sense[0] is not None
            and isinstance(e.sense[0].reference[0], Reference)
            and utils.safe_expand_curie(e.sense[0].reference[0].on_property) == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
            and isinstance(utils.safe_expand_curie(e.sense[0].reference[0].has_value, self.lex.nsmanager), str)
            and (e.canonical_form.written_rep, utils.safe_expand_curie(e.sense[0].reference[0].has_value, self.lex.nsmanager)) not in faulty
            and utils.safe_expand_curie(e.sense[0].reference[0].has_value, self.lex.nsmanager) not in [utils.safe_expand_curie(p, self.lex.nsmanager) for p in consts.compositionality_bad_props]
            and not any([bw in e.canonical_form.written_rep for bw in consts.compositionality_bad_words_predicative])
        ]

        self.predicative_nontype_entries = [
            e for e in self.lex.entries
            if e is not None
            and e.canonical_form is not None
            and e.canonical_form.written_rep is not None
            and "_" not in e.canonical_form.written_rep
            and e.sense[0] is not None
            and isinstance(e.sense[0].reference[0], Reference)
            and (e.part_of_speech, utils.safe_expand_curie(e.sense[0].reference[0].on_property)) in consts.compositionality_nontype_props
            and isinstance(utils.safe_expand_curie(e.sense[0].reference[0].has_value, self.lex.nsmanager), str)
            and (e.canonical_form.written_rep, utils.safe_expand_curie(e.sense[0].reference[0].has_value, self.lex.nsmanager)) not in faulty
            and utils.safe_expand_curie(e.sense[0].reference[0].has_value, self.lex.nsmanager) not in [utils.safe_expand_curie(p, self.lex.nsmanager) for p in consts.compositionality_bad_props]
            and not any([bw in e.canonical_form.written_rep for bw in consts.compositionality_bad_words_predicative])
        ]
        pass

        # self.entries = []
        #
        # for e in entries:
        #     try:
        #         e.sense[0].reference[0] = "<"+self.lex.nsmanager.expand_curie(e.sense[0].reference[0])+">"
        #     except:
        #         e.sense[0].reference[0] = "<"+e.sense[0].reference[0]+">"
        #     self.entries.append(e)

    def generate(self, out_path, num_samples, num_threads):
        raise NotImplementedError()

    @staticmethod
    def extend_entry(entry):
        if "_" in entry.canonical_form.written_rep:
            raise ValueError("Entry has an underscore in its canonical form")
        return f"the {entry.canonical_form.written_rep} {entry.syn_behavior[0].prepositional_adjunct.marker.canonical_form.written_rep}"

    @staticmethod
    def is_english(s):
        try:
            s.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        else:
            return True

    @staticmethod
    def initializer():
        global se
        global bad_triples
        global all_triples
        se = SPARQLEndpoint(endpoint="http://localhost:8890/sparql", default_graph="http://dbpedia.org")#"https://dbpedia.org/sparql")
        bad_triples = set()
        all_triples = set()


    def write_csv(self, path, data):
        with open(path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow({
                    "question1": row[0][0],
                    "query1": row[0][1],
                    "question2": row[1][0],
                    "query2": row[1][1],
                    "question3": row[2][0],
                    "query3": row[2][1]
                })
            csvfile.flush()

    @staticmethod
    def _fix_query_vars(query):
        return query.replace("?b", "?a").replace("?c", "?a")

    @staticmethod
    def _randomize_triples(question, query):
        head = "WHERE {"
        if query.find(head) == -1:
            pass
        tail = "}"
        querybody = query[query.index(head) + len(head):query.rindex(tail)]
        triples = [s.strip() for s in querybody.split(". ") if len(s.strip()) > 0]
        random.shuffle(triples)
        return question, query[:query.index(head) + len(head)] + " " + ". ".join(triples) + ". " + query[query.rindex(tail):]

    def deduplicate(self, path, path_out=None, sample_limit=None, randomize_triples=True, check_sparql=True):
        with open(path) as csv_file:
            data = list(csv.DictReader(csv_file, delimiter=','))
            dups = defaultdict(set)
            for d in data:
                data_tuple = ((d["question1"], self._fix_query_vars(d["query1"])), (d["question2"], self._fix_query_vars(d["query2"])), (d["question3"], self._fix_query_vars(d["query3"])))
                dups[d["question1"]].add(data_tuple)
                dups[d["question2"]].add(data_tuple)
                dups[d["question3"]].add(data_tuple)

            max_sets = []

            qs = list(dups.keys())
            done = set()
            for q in qs:
                if q in done:
                    continue
                curr_max_set = set()
                curr_dup = dups[q]
                curr_max_set.update(curr_dup)
                curr_qs = set([d[0][0] for d in curr_dup] + [d[1][0] for d in curr_dup] + [d[2][0] for d in curr_dup])
                done.update(curr_qs)
                q = Queue()
                curr_dup_qs = list(curr_qs - {q})
                for c in curr_dup_qs:
                    q.put(c)
                while not q.empty():
                    curr = q.get()
                    curr_dup = dups[curr]
                    curr_max_set.update(curr_dup)
                    curr_dup_qs = set(
                        [d[0][0] for d in curr_dup] + [d[1][0] for d in curr_dup] + [d[2][0] for d in curr_dup])
                    for c in curr_dup_qs:
                        if c not in done:
                            done.add(c)
                            q.put(c)
                max_sets.append(curr_max_set)

            final_data = []
            qs_used = set()

            for s in max_sets:
                if len(s) == 1:
                    present_qs = set([d[0][0] for d in s] + [d[1][0] for d in s] + [d[2][0] for d in s])
                    if len(present_qs) != 3:
                        continue
                    if len(present_qs & qs_used) == 0:
                        final_data.append(list(s)[0])
                        qs_used.update(present_qs)
                else:
                    q_freq = defaultdict(int)
                    for d in s:
                        q_freq[d[0][0]] += 1
                        q_freq[d[1][0]] += 1
                        q_freq[d[2][0]] += 1
                    sl = list(s)
                    sl.sort(key=lambda x: q_freq[x[0][0]] + q_freq[x[1][0]] + q_freq[x[2][0]])
                    for d in sl:
                        present_qs = {d[0][0], d[1][0], d[2][0]}
                        if len(present_qs) != 3:
                            continue
                        if len(present_qs & qs_used) == 0:
                            final_data.append(d)
                            qs_used.update(present_qs)

            logging.debug(f"{len(final_data)} unique data points")
            if sample_limit is not None:
                final_data = random.sample(final_data, sample_limit)
                logging.debug(f"Sampled {sample_limit} data points")

            if randomize_triples:
                rand_data = []
                for fd in final_data:
                    qs1, qs2, qs3 = fd
                    rand_data.append((
                        self._randomize_triples(qs1[0], qs1[1]),
                        self._randomize_triples(qs2[0], qs2[1]),
                        self._randomize_triples(qs3[0], qs3[1])
                    ))
                final_data = rand_data

            if check_sparql:
                endpoint = SPARQLEndpoint(endpoint="http://localhost:8890/sparql", default_graph="http://dbpedia.org")
                for fd in final_data:
                    #logging.debug(fd)
                    for qs in fd:
                        qres = endpoint.get_results_query(qs[1])
                        if len(qres) == 0:
                            logging.debug(f"Bad query: {qs[1]}")
                            raise ValueError(f"Bad query: {qs[1]}")
                        # else:
                        #     logging.debug(f"Good query: {qs[1]}")

            if path_out is not None:
                self.write_csv(path_out, final_data)

            return final_data

    def split(self, final_data, path_out_train=None, path_out_val=None, path_out_test=None):
        if path_out_train is None:
            path_out_train = os.path.join(
                os.path.dirname(sys.modules["lemon"].__file__),
                "resources",
                "compositionality_train.csv"
            )

        if path_out_val is None:
            path_out_val = os.path.join(
                os.path.dirname(sys.modules["lemon"].__file__),
                "resources",
                "compositionality_val.csv"
            )

        if path_out_test is None:
            path_out_test = os.path.join(
                os.path.dirname(sys.modules["lemon"].__file__),
                "resources",
                "compositionality_test.csv"
            )

        random.shuffle(final_data)
        train_val, test = train_test_split(final_data, test_size=0.2)
        train, valid = train_test_split(train_val, test_size=0.125)

        self.write_csv(path_out_train, train)
        self.write_csv(path_out_val, valid)
        self.write_csv(path_out_test, test)

    def load_and_split(self, path, path_out_train=None, path_out_val=None, path_out_test=None):
        final_data = []
        with open(path) as csv_file:
            data = list(csv.DictReader(csv_file, delimiter=','))
            for d in data:
                data_tuple = ((d["question1"], self._fix_query_vars(d["query1"])), (d["question2"], self._fix_query_vars(d["query2"])), (d["question3"], self._fix_query_vars(d["query3"])))
                final_data.append(data_tuple)

        self.split(final_data, path_out_train, path_out_val, path_out_test)

    def deduplicate_and_split(self, path, path_out=None, path_out_train=None, path_out_val=None, path_out_test=None, sample_limit=None, randomize_triples=True, check_sparql=True):
        final_data = self.deduplicate(path, path_out, sample_limit, randomize_triples, check_sparql)
        self.split(final_data, path_out_train, path_out_val, path_out_test)


    def convert_to_litgpt(self, path, outpath, valpath=None, testpath=None):
        raise NotImplementedError("Has to be updated for subgraph patterns!")
        # # best optimized proppt for gpt-4o-mini zeroshot
        # #prompt = '**Instruction**: Given a natural language `question`, generate a corresponding `sparql_query` that can be executed against DBpedia to retrieve relevant information. The task involves identifying key entities and their attributes mentioned in the question, such as names, titles, or other identifiers. Use predefined patterns and RDF properties from DBpedia to construct the query accurately.\n\nSteps:\n1. Parse the input question to identify the main entity (e.g., a person\'s name, book title) and the attribute of interest (e.g., birth name, author).\n2. Map these elements to corresponding DBpedia classes and properties.\n3. Construct a SPARQL query using appropriate prefixes and patterns that reflect the identified entities and attributes.\n4. Ensure the query is syntactically correct and adheres to standard SPARQL practices.\n\nExample:\n- For the question \"Give me the birth name of Joyce Ching,\" identify \"Joyce Ching\" as the person and \"birth name\" as the attribute, resulting in a query that retrieves `dbo:birthName` for the entity associated with \"Joyce Ching.\"\n- For \"Give me the author of Jono and Ben,\" recognize \"Jono and Ben\" as a work and \"author\" as the attribute, leading to a query that fetches `dbo:author` for the resource labeled \"Jono and Ben.\"\n\nBy following this structured approach, ensure each question is translated into an executable SPARQL query that accurately reflects its intent.'
        # prompt = 'Generate a single SPARQL query for DBpedia for the following Question. Return only that query as the output.'
        #
        # if outpath is None:
        #     outpath = str(Path(path).with_suffix('.json'))
        #
        # res_data = []
        # with open(path) as csv_file:
        #     data = list(csv.DictReader(csv_file, delimiter=','))
        #     for row in data:
        #         print("Train question:", row, flush=True)
        #         res_data.extend([
        #             {
        #                 "instruction": prompt,
        #                 "input": row["question1"],
        #                 "output": self._fix_query_vars(row["query1"])
        #             },
        #             {
        #                 "instruction": prompt,
        #                 "input": row["question2"],
        #                 "output": self._fix_query_vars(row["query2"])
        #             },
        #             {
        #                 "instruction": prompt,
        #                 "input": row["question3"],
        #                 "output": self._fix_query_vars(row["query3"])
        #             },
        #         ])
        #
        # if valpath is not None:
        #     with open(valpath) as csv_file:
        #         data = list(csv.DictReader(csv_file, delimiter=','))
        #         for row in data:
        #             print("Val question:", row, flush=True)
        #             res_data.extend([
        #                 {
        #                     "instruction": prompt,
        #                     "input": row["question1"],
        #                     "output": self._fix_query_vars(row["query1"])
        #                 },
        #                 {
        #                     "instruction": prompt,
        #                     "input": row["question2"],
        #                     "output": self._fix_query_vars(row["query2"])
        #                 },
        #             ])
        #
        # if testpath is not None:
        #     with open(testpath) as csv_file:
        #         data = list(csv.DictReader(csv_file, delimiter=','))
        #         for row in data:
        #             print("Val question:", row, flush=True)
        #             res_data.extend([
        #                 {
        #                     "instruction": prompt,
        #                     "input": row["question1"],
        #                     "output": self._fix_query_vars(row["query1"])
        #                 },
        #                 {
        #                     "instruction": prompt,
        #                     "input": row["question2"],
        #                     "output": self._fix_query_vars(row["query2"])
        #                 },
        #             ])
        #
        # with open(outpath, "w") as f:
        #     f.write(json.dumps(res_data))


    def convert_to_openai(self, path, outpath, valpath=None, testpath=None):
        # best optimized proppt for gpt-4o-mini zeroshot
        prompt = 'Given a natural language question about a specific entity, generate a SPARQL query that retrieves relevant information from a knowledge base like DBpedia. The output should be structured in the format of a SPARQL SELECT statement, ensuring that the query accurately reflects the information requested in the question. For example, if the question is \"Who is the author of Jono and Ben?\", the output should be a query that selects the author of the specified entity.'

        if outpath is None:
            outpath = str(Path(path).with_suffix('.jsonl'))

        with open(path) as csv_file:
            with open(outpath, "w") as f:
                data = list(csv.DictReader(csv_file, delimiter=','))
                for row in data:
                    print("Train question:", row, flush=True)
                    msg = {"messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": row["question1"]},
                        {"role": "assistant", "content": self._fix_query_vars(row["query1"])}
                    ]}
                    f.write(f"{json.dumps(msg)}\n")
                    msg = {"messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": row["question2"]},
                        {"role": "assistant", "content": self._fix_query_vars(row["query2"])}
                    ]}
                    f.write(f"{json.dumps(msg)}\n")
                    msg = {"messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": row["question3"]},
                        {"role": "assistant", "content": self._fix_query_vars(row["query3"])}
                    ]}
                    f.write(f"{json.dumps(msg)}\n")
                if valpath is not None:
                    with open(valpath) as csv_file:
                        data = list(csv.DictReader(csv_file, delimiter=','))
                        for row in data:
                            print("Val question:", row, flush=True)
                            msg = {"messages": [
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": row["question1"]},
                                {"role": "assistant", "content": self._fix_query_vars(row["query1"])}
                            ]}
                            f.write(f"{json.dumps(msg)}\n")
                            msg = {"messages": [
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": row["question2"]},
                                {"role": "assistant", "content": self._fix_query_vars(row["query2"])}
                            ]}
                            f.write(f"{json.dumps(msg)}\n")
                if testpath is not None:
                    with open(testpath) as csv_file:
                        data = list(csv.DictReader(csv_file, delimiter=','))
                        for row in data:
                            print("Test question:", row, flush=True)
                            msg = {"messages": [
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": row["question1"]},
                                {"role": "assistant", "content": self._fix_query_vars(row["query1"])}
                            ]}
                            f.write(f"{json.dumps(msg)}\n")
                            msg = {"messages": [
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": row["question2"]},
                                {"role": "assistant", "content": self._fix_query_vars(row["query2"])}
                            ]}
                            f.write(f"{json.dumps(msg)}\n")

@dataclass
class Subgraph:
    breadth: int
    depth: int
    graph: DiGraph
    edgelist: frozenset[Tuple[str, str]]
    gid: int

class SubgraphDatasetGenerator(DatasetGenerator):
    def __init__(self, faulty_path: Optional[str] = None, depth=1, breadth=1, endpoint="http://localhost:8890/sparql"):
        super().__init__(faulty_path=faulty_path)
        self.depth = depth
        self.breadth = breadth
        self.prop_to_entry = defaultdict(set)
        self.prop_to_rentry = defaultdict(set)
        self.type_to_entry = defaultdict(set)
        self.nontype_to_entry = defaultdict(lambda: defaultdict(set))
        self.endpoint = endpoint

        nsmanager = utils.create_namespace_manager()
        for en in self.entries:
            try:
                # if not isinstance(e.sense[0].subj_of_prop, SubjOfProp):
                #     logging.warning("Only subj of prop supported")
                #     continue
                uri = utils.safe_expand_curie(en.sense[0].reference[0], nsmanager=nsmanager)

                if not isinstance(en.sense[0].subj_of_prop, SubjOfProp):
                    self.prop_to_rentry[uri].add(self.extend_entry(en))
                else:
                    self.prop_to_entry[uri].add(self.extend_entry(en))
            except Exception as e:
                logging.error(f"Error: {str(e)}")
                logging.error(traceback.format_exc())

        for en in self.predicative_type_entries:
            try:
                uri = utils.safe_expand_curie(en.sense[0].reference[0].has_value, nsmanager=nsmanager)

                self.type_to_entry[uri].add(self.extend_entry(en))
            except Exception as e:
                logging.error(f"Error: {str(e)}")
                logging.error(traceback.format_exc())

        for en in self.predicative_nontype_entries:
            try:
                uri = utils.safe_expand_curie(en.sense[0].reference[0].has_value, nsmanager=nsmanager)
                onprop = utils.safe_expand_curie(en.sense[0].reference[0].on_property, nsmanager=nsmanager)

                self.nontype_to_entry[onprop][uri].add(self.extend_entry(en))
            except Exception as e:
                logging.error(f"Error: {str(e)}")
                logging.error(traceback.format_exc())

        print("Entries:", len([v2 for k, v in self.prop_to_entry.items() for v2 in v]))
        pprint(self.prop_to_entry, width=200)
        print("Reverse entries:", len([v2 for k, v in self.prop_to_rentry.items() for v2 in v]))
        pprint(self.prop_to_rentry, width=200)
        print("Type entries:", len([v2 for k, v in self.type_to_entry.items() for v2 in v]))
        pprint(self.type_to_entry, width=200)
        print("Nontype entries:", len([v3 for k, v in self.nontype_to_entry.items() for k2, v2 in v.items() for v3 in v2]))
        pprint(self.nontype_to_entry, width=200)


    @staticmethod
    def draw_graph(G, res_map=None, bfs_layout=False):
        #plt.figure(figsize=(2560 / 200, 1440 / 200), dpi=200)
        G = G.copy()
        if res_map is not None:
            for e in G.edges:
                if G.edges[e]["label"] in res_map:
                    G.edges[e]["label"] = res_map[G.edges[e]["label"]] + " " + str(G.edges[e]["reverse"])
            for n in G.nodes:
                if "label" in G.nodes[n] and G.nodes[n]["label"] in res_map:
                    G.nodes[n]["label"] = res_map[G.nodes[n]["label"]]

        # nodes_by_descendants = []
        # for n in G.nodes:
        #     nodes_by_descendants.append((n, nx.descendants(G, n)))
        # nodes_by_descendants.sort(key=lambda x: x[1], reverse=True)

        for layer, nodes in enumerate(nx.topological_generations(G)):
            for node in nodes:
                G.nodes[node]["layer"] = layer

        # Draw the graph
        if not bfs_layout:
            pos = nx.multipartite_layout(G, subset_key="layer")
        else:
            pos = nx.bfs_layout(G, scale=1, start=sorted(G.nodes, key=lambda n: nx.descendants(G, n), reverse=True)[0])

        # Draw nodes and edges
        nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=[
            len(v['label']) * 250 if "label" in v else 250 for v in G.nodes().values()
        ])

        # Draw node labels
        node_labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        try:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            logging.error(traceback.format_exc())
        # Show the plot
        plt.show()

    @staticmethod
    def gen_single_pattern_query(basenum, depth, breadth, random_reverse_prob=0.0):
        # TODO: UNIONS fail for 3x3 and larger because of >10000 SQL lines
        # Alternative: independent variables and filters: ?v1 ?p1 ?v2 . ?v3 ?p2 ?v4 . FILTER(?v2 == ?v3 || ?v1 == ?v3 || ?v2 == ?v4 || ?v1 == ?v4)
        # But gets complicated as you chain them further, because not three triples must depend on the same single value if we want to keep the sequences
        assert depth > 0
        assert breadth > 0
        bzeros = str(len(str(breadth)))
        dzeros = str(len(str(depth)))
        resultvar = f"?result{basenum}"

        all_vars = [resultvar]

        labelvars = []
        branch_preds = []

        G = nx.DiGraph()
        G.add_node(resultvar, label=resultvar)

        pattern: str = ""#f"{resultvar} rdfs:label ?resultLabel. "
        clean_pattern: str = ""

        for idx in range(breadth):
            preds = []
            lvs = []

            subj = ("?o{}qb{:0" + bzeros + "d}d{:0" + dzeros + "d}").format(basenum, idx, 0)
            subjl = ("?l{}qb{:0" + bzeros + "d}d{:0" + dzeros + "d}").format(basenum, idx, 0)

            G.add_node(subj, label=subj)
            G.add_node(subjl, label=subjl)
            G.add_edge(subj, subjl, label="rdfs:label", reverse=False)

            all_vars.append(subj)
            all_vars.append(subjl)
            lvs.append((subj, subjl))

            for depthnum in range(1, depth):
                pred = ("?p{}qb{:0" + bzeros + "d}d{:0" + dzeros + "d}").format(basenum, idx, depthnum)
                preds.append(pred)
                all_vars.append(pred)
                obj = ("?o{}qb{:0" + bzeros + "d}d{:0" + dzeros + "d}").format(basenum, idx, depthnum)
                objl = ("?l{}qb{:0" + bzeros + "d}d{:0" + dzeros + "d}").format(basenum, idx, depthnum)

                G.add_node(obj, label=obj)
                G.add_node(objl, label=objl)
                G.add_edge(obj, objl, label="rdfs:label", reverse=False)

                all_vars.append(obj)
                all_vars.append(objl)
                lvs.append((obj, objl))

                if random.random() < random_reverse_prob:
                    G.add_edge(obj, subj, label=pred, reverse=True)
                    #G.add_edge(subj, obj, label=pred, reverse=True)
                    pattern += f"{obj} {pred} {subj}. "
                    clean_pattern += f"{obj} {pred} {subj}. "
                else:
                    #G.add_edge(subj, obj, label=pred, reverse=False)
                    G.add_edge(obj, subj, label=pred, reverse=False)
                    pattern += f"{subj} {pred} {obj}. "
                    clean_pattern += f"{subj} {pred} {obj}. "

                subj = obj

            pred = ("?p{}qb{:0" + bzeros + "d}d{:0" + dzeros + "d}").format(basenum, idx, depth)
            all_vars.append(pred)
            preds.append(pred)

            if random.random() < random_reverse_prob:
                G.add_edge(resultvar, subj, label=pred, reverse=True)
                #G.add_edge(subj, resultvar, label=pred, reverse=True)
                pattern += f"{resultvar} {pred} {subj}. "
                clean_pattern += f"{resultvar} {pred} {subj}. "
            else:
                #G.add_edge(subj, resultvar, label=pred, reverse=False)
                G.add_edge(resultvar, subj, label=pred, reverse=False)
                pattern += f"{subj} {pred} {resultvar}. "
                clean_pattern += f"{subj} {pred} {resultvar}. "

            branch_preds.append(preds)
            labelvars.append(lvs)

        #SubgraphDatasetGenerator.draw_graph(G)
        return pattern, clean_pattern, labelvars, branch_preds, all_vars, G

    @staticmethod
    def _shorten_uri(uri):
        if uri.startswith("http://dbpedia.org/resource/"):
            return uri[len("http://dbpedia.org/resource/"):]
        elif uri.startswith("http://dbpedia.org/ontology/"):
            return uri[len("http://dbpedia.org/ontology/"):]
        elif uri.startswith("http://dbpedia.org/property/"):
            return uri[len("http://dbpedia.org/property/"):]
        return uri

    @staticmethod
    def _escape(s):
        map = {
            '"': '\\"',
            '\r': '\\r',
            '\n': '\\n',
            '\t': '\\t',
            '\b': '\\b',
            '\f': '\\f'
        }
        s = s.replace('\\', '\\u005C\\u005C')
        for key, value in map.items():
            s = s.replace(key, value)
        return '<' + s + '>'

    @staticmethod
    def _verbalize_and(strs):
        return ' and '.join([
            SubgraphDatasetGenerator._adapt_after_and(b) if i > 0 and (
                    (strs[i-1].startswith("the ") and b.startswith("the "))
                    or (strs[i-1].startswith("is ") and b.startswith("is "))
            )
            else b
            for i, b in enumerate(strs)
        ])

    @staticmethod
    def _verbalize_with_types(prop_strs, type_strs):
        if isinstance(prop_strs, str):
            prop_strs = [prop_strs]

        # for ts in type_strs:# avoid sth like "movie that is a movie"
        #     cleants = ' '.join(ts[1].strip().split(' ')[2:])
        #     if len(cleants) > 0 and any([cleants in p for p in prop_strs]):
        #         logging.warning(f"Type {ts} {cleants} already in prop_strs {prop_strs}")
        #         return prop_strs

        if len(type_strs) == 0:
            return prop_strs
        res = []
        for prop_str in prop_strs:
            prop_str_split = prop_str.strip().split(" ")
            prop_str_split = [p for p in prop_str_split if p != ""]
            offset = -1
            if prop_str_split[-2] == "famous":
                offset = -2

            prop_str = f"{' '.join(prop_str_split[:offset])}, that {SubgraphDatasetGenerator._verbalize_and([ts[1] for ts in type_strs])}, {' '.join(prop_str_split[offset:])}"
            res.append(prop_str)
        return res


    @staticmethod
    def verbalize_graph(G: DiGraph, res_map: Dict, prop_to_entry, prop_to_rentry, gid=None, type_candidates=None, is_person=None, seed=None):
        #self.draw_graph(G, res_map=res_map)

        candidates = 1
        if type_candidates is None:
            type_candidates = defaultdict(dict)
        if is_person is None:
            is_person = defaultdict(bool)
        root_node = sorted(G.nodes, key=lambda n: nx.descendants(G, n), reverse=True)[0]

        rnd = random.Random(seed)

        question_root = "Give me "
        question_end = "."
        if rnd.random() < 0.5:
            if is_person[root_node]:
                question_root = "Who is "
                question_end = "?"
            else:
                pass
                # question_root = "What is "
                # question_end = "?"

        pte = sorted([(k, v) for k, v in prop_to_entry.items()], key=lambda t: (t[0], frozenset(t[1])))
        ptre = sorted([(k, v) for k, v in prop_to_rentry.items()], key=lambda t: (t[0], frozenset(t[1])))
        tte = sorted([(k, frozenset([(k2, v2) for k2, v2 in v.items()])) for k, v in type_candidates.items()])

        prop_to_entry = {k: [rnd.choice(list(v))] for k, v in pte}
        prop_to_rentry = {k: [rnd.choice(list(v))] for k, v in ptre}

        type_candidates = defaultdict(list)
        #TODO: type_candidates.update({k: rnd.sample(list(v), rnd.randint(0, len(v))) for k, v in tte})
        type_candidates.update({k: rnd.sample(list(v), rnd.randint(0, len(v))) for k, v in tte})

        var_map = dict()
        var_counter = 0
        for n in G.nodes:
            if n == root_node:
                var_map[n] = "?result"
                continue
            nodename = str(n)
            if "label" in G.nodes[n]:
                nodename = G.nodes[n]["label"]
            var_map[nodename] = f"?v{var_counter}"
            var_counter += 1
        # for e in G.edges:
        #     edgename = str(e)
        #     if "label" in G.edges[e]:
        #         edgename = G.edges[e]["label"]
        #     var_map[edgename] = f"?v{var_counter}"
        #     var_counter += 1

        #nx.edge_dfs(G, root_node)
        end_nodes = [n for n in G.nodes if G.out_degree(n) == 0]
        branches = list(nx.all_simple_edge_paths(G, source=root_node, target=end_nodes))
        branches.sort(key=lambda b: end_nodes.index(b[-1][1]))#ensure end var order matches branch order

        branch_preds = [
            [G.edges[e]["label"] for e in branch]
            for branch in branches
        ]

        branch_reverse = [
            [G.edges[e]["reverse"] for e in branch]
            for branch in branches
        ]

        assert len(branches) == len(end_nodes)

        branch_entry_strs = []

        for pred, lv, rev, branch in zip(branch_preds, end_nodes, branch_reverse, branches):
            branch_cands = []
            for es in itertools.product(*[
                SubgraphDatasetGenerator._verbalize_with_types(prop_to_rentry[res_map[p][1:-1]], type_candidates[b[0]]) if r else # reverse not represented by actual edges but "reverse" property, [0] is always the target var
                SubgraphDatasetGenerator._verbalize_with_types(prop_to_entry[res_map[p][1:-1]], type_candidates[b[0]])
                for p, r, b in zip(pred, rev, branch)
            ]):#, [type_to_entry[p] for p in pred]):
                branch_cands.append(" ".join(es) + " " + res_map["?l"+lv[2:]])
            branch_entry_strs.append(branch_cands)

        triples = []

        for branch in branches:
            for b in branch:
                for t in type_candidates[b[0]]:
                    if "nationality" in t[0][0]:
                        pass
                    triples.append(f"{var_map[b[0]]} <{t[0][0]}> <{t[0][1]}>.")

        triples = list(set(triples))

        print("Type triples:", triples)

        for e in G.edges:
            if "label" in G.edges[e]:
                obj = var_map[e[0]]
                if e[1] in end_nodes:
                    subj = res_map[e[1]]
                    #triples.append(f"{res_map[e[1]]} {res_map[G.edges[e]['label']]} {obj} .")
                else:
                    subj = var_map[e[1]]
                    #triples.append(f"{var_map[e[1]]} {res_map[G.edges[e]['label']]} {obj} .")
                if G.edges[e]["reverse"]:
                    triples.append(f"{obj} {res_map[G.edges[e]['label']]} {subj} .")
                else:
                    triples.append(f"{subj} {res_map[G.edges[e]['label']]} {obj} .")

        num_triples = len(triples)

        triples = list(more_itertools.random_permutation(triples))

        query = f"SELECT ?result WHERE {{ {' '.join(triples)} }}"
        tres = []

        branch_cands = list(more_itertools.random_product(*branch_entry_strs, repeat=candidates))
        assert all([len(b) == 1 for b in branch_entry_strs])
        seq_len = len(branch_entry_strs)
        for i in range(candidates):
            curr_branch_cands = branch_cands[i * seq_len:(i * seq_len) + seq_len]
            perm_branch_cands = more_itertools.random_permutation(curr_branch_cands)
            tres.append({
                f"question": f"{question_root}{SubgraphDatasetGenerator._verbalize_and(perm_branch_cands)}{question_end}",
                f"query": query,
                "num_edges": G.number_of_edges(),
                "num_edges_full": num_triples,
                "num_nodes": G.number_of_nodes(),
                "depth":  nx.dag_longest_path_length(G),
                "breadth": int(max(G.degree)[1]),
                "result": f"{res_map[root_node]}" if root_node in res_map else None,
            } | ({"gid": gid} if gid is not None else {}))
        return tres

    @staticmethod
    def extend_entry(entry):
        if "_" in entry.canonical_form.written_rep:
            raise ValueError("Entry has an underscore in its canonical form")
        if (entry is not None
                and len(entry.syn_behavior) > 0
                and not isinstance(entry.syn_behavior[0], str)
                and entry.syn_behavior[0] is not None
                and "lexinfo:NounPPFrame" in entry.syn_behavior[0].type):
            basic_extension = f"{entry.canonical_form.written_rep} {entry.syn_behavior[0].prepositional_adjunct.marker.canonical_form.written_rep}"
            if basic_extension == "famous for":
                return f"the person {basic_extension}"
            else:
                return f"the {basic_extension}"
        elif (entry is not None
              #and  == PartOfSpeech.NOUN
              and entry.sense[0] is not None
              and isinstance(entry.sense[0].reference[0], Reference)):
            onprop = utils.safe_expand_curie(entry.sense[0].reference[0].on_property)
            if onprop == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" and entry.part_of_speech == PartOfSpeech.NOUN:
                if entry.canonical_form.written_rep.lower()[0] in ["a", "e", "i", "o", "u"] and not any([entry.canonical_form.written_rep.lower().startswith(s) for s in ["uni", "eu", "one"]]):
                    return f"is an {entry.canonical_form.written_rep}"
                else:
                    return f"is a {entry.canonical_form.written_rep}"
            elif onprop == 'http://dbpedia.org/ontology/industry' and entry.part_of_speech == PartOfSpeech.NOUN:
                return f"works in the {entry.canonical_form.written_rep} industry"
            elif (entry.part_of_speech, onprop) in consts.compositionality_nontype_props:
                return f"is {entry.canonical_form.written_rep}"

    @staticmethod
    def _adapt_after_and(entry_str):
        # if entry_str.startswith("the person "):
        #     return "person " + entry_str[8:]
        # elif entry_str.startswith("the "):
        #     return entry_str[4:]
        if entry_str.startswith("the "):
            return entry_str[4:]
        elif entry_str.startswith("is "):
            return entry_str[3:]
        else:
            return entry_str

    @staticmethod
    def _fetch_query(limit, depth, breadth, random_reverse_prob, entity=None):
        while True:
            try:
                pattern, clean_pattern, labelvars, branch_preds, all_vars, G = SubgraphDatasetGenerator.gen_single_pattern_query(1, depth=depth, breadth=breadth, random_reverse_prob=random_reverse_prob)
                filters = []
                for i in range(len(labelvars)):
                    for j in range(i + 1, len(labelvars)):
                        filters.append(f"{labelvars[i][-1][0]} != {labelvars[j][-1][0]}")

                # for branch in branch_preds:
                #     #filters.append(f"{branch[-1]} != rdfs:label")
                #     for p in branch:
                #         filters.append(f"{p} != rdfs:label")
                        # filters.append(f"{p} != rdf:type")
                        # filters.append(f"{p} != rdfs:subClassOf")
                filter_str = "FILTER(" + " && ".join(filters) + ")"

                #prop_restr = ' '.join([f"VALUES {p} {{ <http://dbpedia.org/ontology/director> <http://dbpedia.org/ontology/knownFor> <http://dbpedia.org/property/starring> }}." for branch in branch_preds for p in branch])
                #prop_restr = "VALUES ?result1 { <http://dbpedia.org/resource/Varthamana_Kalam> } VALUES ?o1qb0d0 { <http://dbpedia.org/resource/Kuruppinte_Kanakku_Pustakom> } VALUES ?o1qb1d0 { <http://dbpedia.org/resource/1921_(1988_film)> } VALUES ?p1qb0d1 { <http://dbpedia.org/ontology/director> }. VALUES ?p1qb0d2 { <http://dbpedia.org/property/starring> }. VALUES ?p1qb1d1 { <http://dbpedia.org/ontology/director> <http://dbpedia.org/ontology/knownFor> <http://dbpedia.org/property/starring> }. VALUES ?p1qb1d2 { <http://dbpedia.org/ontology/director> }. "

                if entity is not None:
                    base_query = f"SELECT DISTINCT {' '.join([n for n in all_vars if not n.startswith('?l') and not n.startswith('?result')])} WHERE {{ {pattern.replace('?result1', '<'+entity+'>')} {filter_str} }}"  # {prop_restr}
                else:
                    base_query = f"SELECT DISTINCT {' '.join([n for n in all_vars if not n.startswith('?l')])} WHERE {{ {pattern} {filter_str} }}"#{prop_restr}
                logging.debug(f"Query: {base_query} LIMIT {limit} OFFSET {last_offsets[base_query]}")
                res = se_data.get_results_query(base_query + f" LIMIT {limit} OFFSET {last_offsets[base_query]}", raw=True, check_query=False)
                last_offsets[base_query] += limit
                return res, G
            except Exception as e:
                logging.error(f"Exception for query: {type(e)} {str(e)}")
                logging.error(traceback.format_exc())
                continue
                #return []

    @staticmethod
    def _fetch_entities(limit, breadth, depth):
        while True:
            try:
                base_query = f"SELECT DISTINCT ?s WHERE {{ ?s [] [] }} ORDER BY RAND()"  # {prop_restr}
                #base_query = "SELECT DISTINCT ?s WHERE { { SELECT DISTINCT ?s (COUNT(DISTINCT ?p) AS ?c) (COUNT(?s) AS ?d) WHERE {?s ?p []} ORDER BY DESC(?c) DESC(?d) } ?s rdfs:label [] }"  # {prop_restr}

                triples = ["_:a1 [] ?s .", "?s [] [] ."]
                for i in range(1, depth):
                    triples.append(f"_:a{i + 1} [] _:a{i} .")

                query = f"select distinct ?s where {{ {' '.join(triples)} }}"
                if breadth > 1:
                    query += f" GROUP BY ?s HAVING(COUNT(distinct _:a1) >= {breadth})"

                logging.debug(f"Query: {base_query} LIMIT {limit} OFFSET {last_offsets[base_query]}")
                res = se_data.get_results_query(base_query + f" LIMIT {limit} OFFSET {last_offsets[base_query]}", check_query=False)
                #res = se_full.get_results_query(base_query + f" LIMIT {limit}", check_query=False)
                last_offsets[base_query] += limit
                return res
            except Exception as e:
                logging.error(f"Exception for query: {type(e)} {str(e)}")
                logging.error(traceback.format_exc())
                continue
                # return []

    @staticmethod
    def _thread_initializer(endpoint="http://localhost:8890/sparql"):
        global last_offsets
        global se_full
        global se_data
        global se_types
        last_offsets = defaultdict(int)
        se_full = SPARQLEndpoint(endpoint=endpoint, default_graph="http://dbpedia.org", check_endpoint=False, cache=dict())
        se_data = SPARQLEndpoint(endpoint=endpoint, default_graph="http://dbpedia.org/data", check_endpoint=False, cache=dict())
        se_types = SPARQLEndpoint(endpoint=endpoint, default_graph="http://dbpedia.org/types", check_endpoint=False, cache=dict())

    @staticmethod
    def _process_initializer(
            all_vars_,
            force_labels_,
            random_reverse_prob_,
            subgraphs_,
            valid_subgraphs_,
            depth_,
            breadth_,
            prop_to_entry_,
            prop_to_rentry_,
            type_to_entry_,
            nontype_to_entry_,
            endpoint_="http://localhost:8890/sparql",
            limit_=10000
    ):
        SubgraphDatasetGenerator._thread_initializer(endpoint=endpoint_)
        global pool
        global endpoint
        global all_vars
        global force_labels
        global random_reverse_prob
        global subgraphs
        global valid_subgraphs
        global depth
        global breadth
        global prop_to_entry
        global prop_to_rentry
        global type_to_entry
        global nontype_to_entry
        global limit
        pool = ThreadPool(processes=4, initializer=SubgraphDatasetGenerator._thread_initializer, initargs=(endpoint_,))
        endpoint = endpoint_
        all_vars = all_vars_
        force_labels = force_labels_
        random_reverse_prob = random_reverse_prob_
        subgraphs = subgraphs_
        valid_subgraphs = valid_subgraphs_
        depth = depth_
        breadth = breadth_
        prop_to_entry = prop_to_entry_
        prop_to_rentry = prop_to_rentry_
        type_to_entry = type_to_entry_
        nontype_to_entry = nontype_to_entry_
        limit = limit_





    def gen_data_rows(self,
                      limit=100,
                      breadth_limit=None,
                      depth_limit=None,
                      force_labels=False,
                      random_reverse_prob=0.5,
                      entity_list=None,
                      entity_list_skip=None,
                      num_threads=None):
        assert self.depth > 0
        assert self.breadth > 0
        assert self.depth + self.breadth > 2 # cannot separate into two parts otherwise
        if limit is None:
            limit = 100
        if entity_list_skip is None:
            entity_list_skip = set()

        logging.debug(f"Skipping {len(entity_list_skip)} entities")


        all_vars, powerset_graph, subgraphs, valid_subgraphs = self.gen_powerset_graph(breadth_limit, depth_limit)
        yield {"powersetgraph": json.dumps(powerset_graph, default=nx.node_link_data)}

        pool = ThreadPool(processes=1, initializer=SubgraphDatasetGenerator._thread_initializer, initargs=(self.endpoint,))

        if entity_list is None:
            async_result_entities = pool.apply_async(self._fetch_entities, (limit, self.breadth, self.depth))
            ents = list(async_result_entities.get())
            random.shuffle(ents)
            async_result_entities = pool.apply_async(self._fetch_entities, (limit, self.breadth, self.depth))
        else:
            if entity_list.endswith(".zst"):
                with open(entity_list, "rb") as f:
                    dctx = zstandard.ZstdDecompressor()
                    stream_reader = dctx.stream_reader(f)
                    text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
                    reader = csv.DictReader(text_stream)
                    ents = [item["?s"] for item in reader]
            else:
                with open(entity_list) as f:
                    reader = csv.DictReader(f)
                    ents = [item["?s"] for item in reader]

        logging.debug(f"Entities before skipping: {len(ents)}")
        ents = [e for e in ents if e not in entity_list_skip]
        logging.debug(f"Entities after skipping: {len(ents)}")
        random.shuffle(ents)
        finished_num = len(entity_list_skip)
        with Pool(
                processes=num_threads,
                initializer=SubgraphDatasetGenerator._process_initializer,
                initargs=(all_vars, force_labels, random_reverse_prob, subgraphs, valid_subgraphs, self.depth,
                          self.breadth, self.prop_to_entry, self.prop_to_rentry, self.type_to_entry,
                          self.nontype_to_entry, self.endpoint, limit)
        ) as ppool:
            while True:


                for l in ppool.imap_unordered(SubgraphDatasetGenerator.gen_row_for_ent, ents):
                    if l is not None:
                        yield from l
                    logging.debug(f"Finished {finished_num} of {len(ents)}")
                    finished_num += 1
                if len(ents) == limit and entity_list is None:
                    ents = list(async_result_entities.get())
                    ents = [e for e in ents if e not in entity_list_skip]
                    random.shuffle(ents)
                    async_result_entities = pool.apply_async(self._fetch_entities, (limit, self.breadth, self.depth))
                else:
                    break

    @staticmethod
    def gen_row_for_ent(ent):
        pool = ThreadPool(processes=4, initializer=SubgraphDatasetGenerator._thread_initializer, initargs=(endpoint,))
        async_result = None
        for _ in range(10):# 10 tries to get valid subgraph
            # logging.debug(f"Query: {base_query} LIMIT {limit} OFFSET {offset}")
            logging.debug("Fetching new batch")
            if async_result is not None:
                res, G_verb = async_result.get()
                async_result = pool.apply_async(SubgraphDatasetGenerator._fetch_query, (limit, depth, breadth, random_reverse_prob, ent))
            else:
                async_result = pool.apply_async(SubgraphDatasetGenerator._fetch_query, (limit, depth, breadth, random_reverse_prob, ent))
                res, G_verb = async_result.get()
                async_result = pool.apply_async(SubgraphDatasetGenerator._fetch_query, (limit, depth, breadth, random_reverse_prob, ent))

            # res = se.get_results_query(base_query + f" LIMIT {limit} OFFSET {offset}", raw=True)
            logging.debug(f"Results: {len(res)}")
            logging.debug("Fetching new batch finished")
            if len(res) == 0:
                if random_reverse_prob > 0.0001:
                    continue
                else:
                    #break
                    return None
            else:
                logging.debug(f"Found {len(res)} results for entity {ent}")

            random.shuffle(res)

            skeleton_graph = G_verb.subgraph([n for n in all_vars if not n.startswith("?l")])

            rowres = []

            for r in res:
                try:
                    #r_entities = frozenset({v["value"] for k, v in r.items() if "?" + k in end_nodes})
                    r_props = frozenset({v["value"] for k, v in r.items() if k.startswith("p")})

                    if 'http://www.w3.org/2000/01/rdf-schema#label' in r_props:
                        logging.error("rdfs:label in properties, skipping")
                        continue

                    diff_prop_portion = len(r_props) / len([v["value"] for k, v in r.items() if k.startswith("p")])

                    if diff_prop_portion < 0.5:
                        logging.debug(f"Skipping due to not diverse enough property set {diff_prop_portion} {r_props} of {len([v['value'] for k, v in r.items() if k.startswith('p')])}")
                        continue

                    res_map_full = dict()  # {"?" + k: v["value"] if v["type"] != "uri" else v["value"] for k, v in r.items()}
                    type_map = defaultdict(dict)
                    is_person = defaultdict(bool)
                    for k, v in r.items():
                        if v["type"] != "uri":
                            if 'datatype' in v:
                                res_map_full["?" + k] = '"' + v["value"] + '"^^<' + v['datatype'] + '>'
                            elif 'xml:lang' in v:
                                res_map_full["?" + k] = '"' + v["value"] + '"@' + v['xml:lang']
                            else:
                                res_map_full["?" + k] = '"' + v["value"] + '"'
                            if k.startswith("o"):
                                res_map_full["?l" + k[1:]] = v["value"]
                        else:
                            res_map_full["?" + k] = "<" + v["value"] + ">"
                            # res_map_full["?" + k] = "<"+self._shorten_uri(v["value"])+">"

                            if not k.startswith("l"):  # probably unnecessary as labels will never be type uri
                                person_res = se_types.ask_triple(f"<{v['value']}> a <http://xmlns.com/foaf/0.1/Person>")
                                is_person["?" + k] = person_res

                                type_map["?" + k] = dict()
                                typeres = se_types.get_results_query(f"SELECT ?type WHERE {{ <{v['value']}> a ?type }}")
                                if len(typeres) > 0:
                                    for t in typeres:
                                        uri = utils.safe_expand_curie(t)
                                        if uri in type_to_entry:
                                            if person_res is not se_types.ask_triple(f"<{uri}> (rdfs:subClassOf|owl:equivalentClass)* <http://xmlns.com/foaf/0.1/Person>"):
                                                continue

                                            type_map["?" + k][("http://www.w3.org/1999/02/22-rdf-syntax-ns#type", uri)] = random.choice(list(type_to_entry[uri]))
                                            print("Type found", v['value'], " ".join(type_to_entry[uri]), flush=True)

                                for pos, ntprop in consts.compositionality_nontype_props:
                                    typeres = se_data.get_results_query(f"SELECT ?type WHERE {{ <{v['value']}> <{ntprop}> ?type }}")
                                    if len(typeres) > 0:
                                        for t in typeres:
                                            uri = utils.safe_expand_curie(t)
                                            if uri in nontype_to_entry[ntprop]:
                                                type_map["?" + k][(ntprop, uri)] = random.choice(list(nontype_to_entry[ntprop][uri]))
                                                print(f"Type {ntprop} found", v['value'], " ".join(nontype_to_entry[ntprop][uri]), flush=True)

                            if k.startswith("o") and "?l" + k[1:] not in res_map_full:
                                labelres = se_types.get_results_query(f"SELECT ?label WHERE {{ <{v['value']}> rdfs:label ?label }}")
                                if len(labelres) > 0:
                                    res_map_full["?l" + k[1:]] = list(labelres)[0]
                                else:
                                    if force_labels:
                                        raise ValueError(f"No label found for {v['value']}")
                                    else:
                                        logging.warning(f"No label found for {v['value']}")
                                    res_map_full["?l" + k[1:]] = SubgraphDatasetGenerator._shorten_uri(v["value"]).replace("_", " ")

                    # if any([not DatasetGenerator.is_english(x) for x in res_map_full.values()]):
                    #     raise ValueError("Non-ASCII value found")

                    # self.draw_graph(skeleton_graph, res_map_full, bfs_layout=True)

                    # SubgraphDatasetGenerator.draw_graph(G, res_map_full)
                    seed = random.randint(-sys.maxsize, sys.maxsize)
                    comb_res = SubgraphDatasetGenerator.verbalize_graph(
                        skeleton_graph,
                        res_map_full,
                        prop_to_entry=prop_to_entry,
                        prop_to_rentry=prop_to_rentry,
                        gid=subgraphs,
                        type_candidates=type_map,
                        is_person=is_person,
                        seed=seed)
                    part_res = [
                        SubgraphDatasetGenerator.verbalize_graph(
                            G_verb.subgraph([n for n in g.graph.nodes]),
                            res_map_full,
                            prop_to_entry=prop_to_entry,
                            prop_to_rentry=prop_to_rentry,
                            gid=g.gid,
                            type_candidates=type_map,
                            is_person=is_person,
                            seed=seed
                        )
                        for part_graphs in valid_subgraphs.values() for g in part_graphs
                    ]

                    for c in comb_res:
                        # if not se_full.ask_triple(c["query"][len("SELECT ?result WHERE { "):-len(" }")]):
                        #     pass
                        assert se_full.ask_triple(c["query"][len("SELECT ?result WHERE { "):-len(" }")])

                    for p in part_res:
                        for c in p:
                            # if not se_full.ask_triple(c["query"][len("SELECT ?result WHERE { "):-len(" }")]):
                            #     pass
                            assert se_full.ask_triple(c["query"][len("SELECT ?result WHERE { "):-len(" }")])

                    rowres.append({
                        "combined": comb_res,
                        "parts": part_res,
                        "diff_prop_portion": diff_prop_portion,
                        "result": ent,
                    })
                    # yield {
                    #     "combined": comb_res,
                    #     "parts": part_res,
                    #     "diff_prop_portion": diff_prop_portion,
                    #     "result": ent,
                    # }
                except ValueError as e:
                    logging.debug(str(e))
                    logging.debug(traceback.format_exc())
                    continue
                except Exception as e:
                    logging.error(str(type(e)))
                    logging.error(str(e))
                    logging.error(traceback.format_exc())
                    continue

                if len(rowres) >= 10: # 10 random examples are enough
                    break

            return rowres

        return []

    def gen_powerset_graph(self, breadth_limit, depth_limit):
        pattern, clean_pattern, labelvars, branch_preds, all_vars, G = SubgraphDatasetGenerator.gen_single_pattern_query(
            1, depth=self.depth, breadth=self.breadth)
        skeleton_graph = G.subgraph([n for n in all_vars if not n.startswith("?l")])
        # self.draw_graph(skeleton_graph)
        # end_nodes = [skeleton_graph.nodes[n]["label"] for n in skeleton_graph.nodes if
        #              skeleton_graph.out_degree(n) == 0 and "label" in skeleton_graph.nodes[n]]
        valid_subgraphs = defaultdict(list)
        powerset_graph = nx.DiGraph()
        subgraphs = 0
        for edgelist in more_itertools.powerset_of_sets(skeleton_graph.edges):
            if len(edgelist) == 0 or len(edgelist) == skeleton_graph.number_of_edges():
                continue
            G2: DiGraph = skeleton_graph.edge_subgraph(edgelist)
            if nx.is_weakly_connected(G2):
                breadth = int(max(G2.degree)[1])
                depth = nx.dag_longest_path_length(G2)
                if breadth_limit is not None and depth_limit is not None and all(
                        [breadth > bl or depth > dl for bl, dl in zip(breadth_limit, depth_limit)]):
                    continue
                elif breadth_limit is not None and all([breadth > bl for bl in breadth_limit]):
                    continue
                elif depth_limit is not None and all([depth > dl for dl in depth_limit]):
                    continue
                print(f"{subgraphs} Depth: {depth}, Breadth: {breadth}")
                edgelist_set = frozenset(edgelist)
                valid_subgraphs[len(edgelist)].append(
                    Subgraph(breadth=breadth, depth=depth, graph=G2, edgelist=edgelist_set, gid=subgraphs))
                # label = str(edgelist_set)#f"B: {breadth} D: {depth} N: {G2.number_of_nodes()} E: {len(edgelist)}"
                label = f"B: {breadth} D: {depth}\nN: {G2.number_of_nodes()} E: {len(edgelist)}"
                powerset_graph.add_node(subgraphs,
                                        label=label,
                                        breadth=breadth,
                                        depth=depth,
                                        num_edges=len(edgelist),
                                        num_nodes=G2.number_of_nodes(),
                                        graphjson=json.dumps(G2, default=nx.node_link_data),
                                        edgelist=list(edgelist),
                                        gid=subgraphs
                                        )
                for n in powerset_graph.nodes:
                    if n == subgraphs:
                        continue
                    if edgelist_set.issubset(
                            powerset_graph.nodes[n]["edgelist"]):  # probably never happens due to powerset order
                        powerset_graph.add_edge(n, subgraphs)
                    if edgelist_set.issuperset(powerset_graph.nodes[n]["edgelist"]):
                        powerset_graph.add_edge(subgraphs, n)
                subgraphs += 1
                # res = SubgraphDatasetGenerator.verbalize_graph(G2, res_map_full)
                # pprint(res)
                # SubgraphDatasetGenerator.draw_graph(G2)
        # label = str(list(skeleton_graph.edges))#f"B: {self.breadth} D: {self.depth} N: {skeleton_graph.number_of_nodes()} E: {len(skeleton_graph.edges)}"
        label = f"B: {self.breadth} D: {self.depth}\nN: {skeleton_graph.number_of_nodes()} E: {len(skeleton_graph.edges)}"
        powerset_graph.add_node(subgraphs,
                                label=label,
                                breadth=self.breadth,
                                depth=self.depth,
                                num_edges=len(skeleton_graph.edges),
                                num_nodes=skeleton_graph.number_of_nodes(),
                                graphjson=json.dumps(skeleton_graph, default=nx.node_link_data),
                                edgelist=list(skeleton_graph.edges),
                                gid=subgraphs
                                )
        for n in powerset_graph.nodes:
            if n == subgraphs:
                continue
            powerset_graph.add_edge(subgraphs, n)
        logging.debug(f"Subgraphs: {subgraphs}")
        return all_vars, powerset_graph, subgraphs, valid_subgraphs

    def generate(
            self,
            out_path: Optional[str] = None,
            num_samples=None,
            num_threads=1,
            limit=100,
            #num_parts=2,
            breadth_limit=None,
            depth_limit=None,
            #max_part_intersections=0.0,
            #candidates=1,
            force_labels=False,
            random_reverse_prob=0.5,
            entity_list=None,
    ):
        if out_path is None:
            out_path = os.path.join(
                os.path.dirname(sys.modules["lemon"].__file__),
                "resources",
                f"compositionality_samples_subgraph_{self.breadth}_{self.depth}_{utils.slugify(str(breadth_limit))}_{utils.slugify(str(depth_limit))}.jsonl"
            )

        if breadth_limit is None:
            breadth_limit = [self.breadth]
        if depth_limit is None:
            depth_limit = [self.depth]

        #assert all([self.depth * self.breadth <= num_parts * bl * dl for bl, dl in zip(breadth_limit, depth_limit)])
        # self.initializer(self.prop_to_entry)
        # for r in self.gen_basic_cands():
        #     self.gen_data_row(r)

        row_counter = 0
        mode = "w"
        entity_list_skip = set()

        if os.path.exists(out_path):
            with jsonlines.open(out_path) as reader:
                prev_out_data = list(reader)

            row_counter = len(prev_out_data)

            entity_list_skip = {d["result"] for d in prev_out_data if "powersetgraph" not in d}

            # with open(out_path, "r") as f:
            #     row_counter = sum(1 for _ in f)
            mode = "a+"

        logging.debug(f"Starting at {row_counter}")

        with open(out_path, mode) as f:
            for row in self.gen_data_rows(limit=limit, breadth_limit=breadth_limit, depth_limit=depth_limit,
                                          force_labels=force_labels, random_reverse_prob=random_reverse_prob,
                                          entity_list=entity_list, entity_list_skip=entity_list_skip,
                                          num_threads=num_threads):
                if "powersetgraph" in row and row_counter > 0:
                    continue

                row["id"] = row_counter
                f.write(f"{json.dumps(row)}\n")
                logging.debug(row)

                if num_samples is not None and row_counter >= num_samples:
                    f.flush()
                    return

                row_counter += 1
                f.flush()

            logging.debug("Finished")

    @staticmethod
    def bulk_fetch(
            breadth,
            depth,
            out_path=None,
            endpoint="http://localhost:8890/sparql",
    ):
        if out_path is None:
            out_path = os.path.join(
                os.path.dirname(sys.modules["lemon"].__file__),
                "resources",
                f"{breadth}_{depth}_query_result.csv"
            )

        se = SPARQLEndpoint(endpoint=endpoint, default_graph="http://dbpedia.org/data", check_endpoint=False, cache=dict())

        # triples = ["?s ?p _:a1 ."]
        # for i in range(1, depth):
        #     triples.append(f"_:a{i} [] _:a{i+1} .")
        #
        # query = f"select distinct ?s where {{ {' '.join(triples)} }}"
        # if breadth > 1:
        #     query += f" GROUP BY ?s HAVING(COUNT(distinct ?p) >= {breadth})"

        # triples = ["?s [] _:a1 ."]
        # for i in range(1, depth):
        #     triples.append(f"_:a{i} [] _:a{i + 1} .")
        #
        # query = f"select distinct ?s where {{ {' '.join(triples)} }}"
        # if breadth > 1:
        #     query += f" GROUP BY ?s HAVING(COUNT(distinct _:a1) >= {breadth})"

        triples = ["_:a1 [] ?s .", "?s [] [] ."]
        for i in range(1, depth):
            triples.append(f"_:a{i + 1} [] _:a{i} .")

        query = f"select distinct ?s where {{ {' '.join(triples)} }}"
        if breadth > 1:
            query += f" GROUP BY ?s HAVING(COUNT(distinct _:a1) >= {breadth})"

        logging.debug(query)

        with open(out_path, 'a+', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["?s"])
            writer.writeheader()
            limit = 1000000
            offset = 0

            while True:
                logging.debug(query + f" LIMIT {limit} OFFSET {offset}")
                data = se.get_results_query(query=query + f" LIMIT {limit} OFFSET {offset}", check_query=False)

                for row in data:
                    writer.writerow({
                        "?s": row
                    })
                csvfile.flush()

                if len(data) < limit:
                    break
                offset += limit

            logging.debug(f"Finished {out_path}")

    @staticmethod
    def split_jsonl(
            paths,
            existpaths=None,
            path_out_base=None,
            #split_level="hard",
            sample_size=10,
            shots=5,
            val_test_random=None
    ):
        if path_out_base is None:
            path_out_base = os.path.join(
                os.path.dirname(sys.modules["lemon"].__file__),
                "resources"
            )

        split_levels = ["easy", "medium", "hard", "max"]
        #split_levels = ["max"]
        skipids = set()

        if existpaths is not None:
            for p in existpaths:
                with open(p) as f:
                    olddata = json.load(f)
                    for d in olddata["train"] + olddata["val"] + olddata["test"]:
                        skipids.add((d["id"], d["base_depth"], d["base_breadth"]))

        print("Skipping", skipids)

        train = defaultdict(list)
        val = defaultdict(list)
        test = defaultdict(list)
        powersetgraphs = []

        for path in paths:

            if path.endswith(".zst"):
                with open(path, "rb") as f:
                    dctx = zstandard.ZstdDecompressor()
                    stream_reader = dctx.stream_reader(f)
                    text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
                    with jsonlines.Reader(text_stream) as reader:
                        data = list(reader)
            else:
                if not path.endswith(".jsonl"):
                    logging.debug("Not a jsonl file probably!")
                with jsonlines.open(path) as reader:
                    data = list(reader)

            assert len(data) > 1 and "powersetgraph" in data[0]

            print("Original data length", len(data))

            data = data[:1] + [d for d in data[1:] if (d["id"], d["combined"][0]["depth"], d["combined"][0]["breadth"]) not in skipids]
            print("Filtered data length", len(data))

            G = nx.node_link_graph(json.loads(data[0]["powersetgraph"]), edges="links")
            max_edges = data[1]["combined"][0]["num_edges"]
            max_nodes = data[1]["combined"][0]["num_nodes"]
            max_depth = data[1]["combined"][0]["depth"]
            max_breadth = data[1]["combined"][0]["breadth"]
            subgraphs = data[1]["combined"][0]["gid"]

            powersetgraphs.append({
                "max_edges": max_edges,
                "max_nodes": max_nodes,
                "max_depth": max_depth,
                "max_breadth": max_breadth,
                "subgraphs": subgraphs
            } | data[0])

            qdata = defaultdict(list)
            for d in data[1:]:
                qdata[d["result"]].append(d)

            d_split = dict()
            adapted_sample_size = sample_size if int(len(list(qdata.keys()))/len(split_levels)) >= sample_size else int(len(list(qdata.keys()))/len(split_levels))
            logging.debug(f"Sample size {adapted_sample_size}")

            for sl in split_levels:
                sldata = []
                chosen_results = random.sample(list(qdata.keys()), min(adapted_sample_size, len(qdata)))
                for r in chosen_results:
                    chosen_elem = random.choice(qdata[r])
                    sldata.append(chosen_elem)
                    #qdata[r].remove(chosen_elem)
                    #if len(qdata[r]) == 0:
                        # qdata.pop(r, None)
                    qdata.pop(r, None)

                d_split[sl] = sldata

            # qdata = data[1:]
            # qdata.sort(key=lambda x: (-x["diff_prop_portion"], x["min_used_entity_portion"], x["max_used_entity_portion"]))
            # qdata = qdata[:sample_size*len(split_levels)]
            # random.shuffle(qdata)
            #
            # d_split = {
            #     sl: qdata[sample_size*i:sample_size*(i+1)]
            #     # "easy": qdata[:sample_size],
            #     # "medium": qdata[sample_size:sample_size*2],
            #     # "hard": qdata[sample_size*2:sample_size*3],
            #     # "max": qdata[sample_size*3:],
            #     for i, sl in enumerate(split_levels)
            # }

            for split_level in d_split.keys():
                d_all = d_split[split_level]

                train_edgelimit = None

                if split_level == "easy":
                    train_edgelimit = int(0.75 * max_edges)
                elif split_level == "medium":
                    train_edgelimit = int(0.5 * max_edges)
                elif split_level == "hard":
                    train_edgelimit = int(0.25 * max_edges)
                elif split_level == "max":
                    train_edgelimit = int(0.0 * max_edges) # auto-set to minimum
                else:
                    raise ValueError(f"Unknown split level: {split_level}")

                if train_edgelimit < 2 and (max_breadth > 1 or max_depth > 1):
                    logging.warning("Too few edges for training and depth, setting to two!")
                    if max_edges == 2:
                        train_edgelimit = 1
                    else:
                        train_edgelimit = 2


                for d in d_all:
                    did = d["id"]
                    additional_infos = {"id": did, "base_depth": max_depth, "base_breadth": max_breadth, "subgraphs": subgraphs}
                    train_local = []
                    rest = []

                    for part_cands in d["parts"]:
                        for part in part_cands:
                            if part["num_edges"] <= train_edgelimit:
                                train_local.append(part | additional_infos)
                            else:
                                rest.append(part | additional_infos)

                    train[split_level].extend(train_local)

                    for cg in d["combined"]:
                        rest.append(cg | additional_infos)

                    rest.sort(key=lambda x: x["num_edges"])

                    if val_test_random is not None:
                        assert len(rest) > 0
                        test[split_level].extend(rest)
                    else:
                        if len(rest) > 1:
                            val_rest = rest[:len(rest)//2]
                            if len(val_rest) > len(train_local):
                                random.shuffle(val_rest)
                                val[split_level].extend(val_rest[:len(train_local)])
                                test[split_level].extend(val_rest[len(train_local):])
                                test[split_level].extend(rest[len(rest) // 2:])
                            else:
                                val[split_level].extend(val_rest)
                                test[split_level].extend(rest[len(rest) // 2:])
                        else:
                            if len(test[split_level]) > len(val[split_level]):
                                val[split_level].extend(rest)
                            else:
                                test[split_level].extend(rest)

        if val_test_random is not None:
            for split_level in train.keys():
                new_train, new_val = train_test_split(train[split_level], test_size=val_test_random)
                train[split_level] = new_train
                val[split_level] = new_val

        for split_level in train.keys():
            with open(os.path.join(path_out_base, f"compositionality_subgraph_{split_level}.json"), "w") as f:
                json.dump({
                    "powersetgraphs": powersetgraphs,
                    "train": train[split_level],
                    "val": val[split_level],
                    "test": test[split_level]
                }, f)
            jsonoutpath = SubgraphDatasetGenerator.extract_prompt_experiments(
                path=os.path.join(path_out_base, f"compositionality_subgraph_{split_level}.json"),
                out_path=None,
                shots=shots,
                limit=None,
                fill_shots=True if split_level != "max" else False,
                min_only=True if split_level == "max" else False
            )
            SubgraphDatasetGenerator.convert_json_to_openai(jsonoutpath)
            #SubgraphDatasetGenerator.convert_json_to_openai(os.path.join(path_out_base, f"compositionality_subgraph_{split_level}.json"))


    @staticmethod
    def convert_json_to_openai(path):
        # best optimized proppt for gpt-4o-mini zeroshot
        prompt = consts.openai_subgraph_prompt

        outpath = str(Path(path).with_suffix('.jsonl'))

        mode = "r"
        if path.endswith(".zst"):
            mode = "rb"
        with open(path, mode) as json_file:
            if path.endswith(".zst"):
                dctx = zstandard.ZstdDecompressor()
                stream_reader = dctx.stream_reader(json_file)
                json_file = io.TextIOWrapper(stream_reader, encoding='utf-8')
                path = path[:-4]
                outpath = str(Path(path).with_suffix('.jsonl'))

            jsondata = json.load(json_file)

            for split in ["train", "val", "test"]:
                data = jsondata[split]
                with open(outpath[:-6].replace("_fixed", "")+"_"+split+".jsonl", "w") as f:
                    for row in data:
                        msg = {"messages": [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": row["question"]},
                            {"role": "assistant", "content": row["query"]}
                        ]}
                        f.write(f"{json.dumps(msg)}\n")
            if "fixed_shots" in jsondata:
                for split in ["train", "test"]:
                    data = jsondata["fixed_shots"][split]
                    with open(outpath[:-6] + "_" + split + ("_fixed.jsonl" if "fixed" not in outpath[:-6] else ".jsonl"), "w") as f:
                        for row in data:
                            msglist = [
                                {"role": "system", "content": prompt},
                            ] + sum([
                                [
                                    {"role": "user", "content": shot["question"]},
                                    {"role": "assistant", "content": shot["query"], "weight": 0}
                                ] for shot in row["shots"]
                            ], []) + [
                                {"role": "user", "content": row["task"]["question"]},
                                {"role": "assistant", "content": row["task"]["query"], "weight": 1}
                            ]
                            # if len(set([m["content"] for m in msglist])) != len([m["content"] for m in msglist]):
                            #     pass
                            #assert len(set([m["content"] for m in msglist])) == len([m["content"] for m in msglist])
                            assert row["task"]["question"] not in {shot["question"] for shot in row["shots"]}
                            assert row["task"]["query"] not in {shot["query"] for shot in row["shots"]}
                            msg = {"messages": msglist}
                            f.write(f"{json.dumps(msg)}\n")

    @staticmethod
    def greedy_edge_cover(target_edges: Set[Tuple[str, str]], subgraph_gids: List, pg: DiGraph, breadth, depth):
        covered_edges = set()
        selected_subgraphs = set()
        max_depth = 0
        max_breadth = 0
        #subgraph_gids.sort(key=lambda x: (int(pg.nodes[x]["num_edges"]), min(int(pg.nodes[x]["breadth"]), int(pg.nodes[x]["depth"]))), reverse=True)#int(x["num_edges"])
        subgraph_gids.sort(key=lambda x: (len({(e[0], e[1]) for e in pg.nodes[x]["edgelist"]}.difference(covered_edges)), min(int(pg.nodes[x]["breadth"]), int(pg.nodes[x]["depth"]))), reverse=True)#int(x["num_edges"])
        for sg in subgraph_gids:
            edge_set = {(e[0], e[1]) for e in pg.nodes[sg]["edgelist"]}
            if len(covered_edges.intersection(edge_set)) == 0 and len(edge_set.difference(target_edges)) == 0:#len(covered_edges.intersection(edge_set)) == 0   len(edge_set.difference(covered_edges)) > 0
                selected_subgraphs.add(sg)
                covered_edges.update(edge_set)
                max_depth = max(max_depth, pg.nodes[sg]["depth"])
                max_breadth = max(max_breadth, pg.nodes[sg]["breadth"])
            if len(target_edges.difference(covered_edges)) == 0:
                break

        if breadth > 1 and max_breadth <= 1 and depth > 1 and max_depth <= 1:
            cands = [
                x for x in subgraph_gids
                if x not in selected_subgraphs
                   and int(pg.nodes[x]["breadth"]) > 1
                   and int(pg.nodes[x]["depth"]) > 1
                   and len({(e[0], e[1]) for e in pg.nodes[x]["edgelist"]}.difference(target_edges)) == 0
            ]
            if len(cands) > 0:
                selected_subgraphs.add(random.choice(cands))
            else:
                cands = [
                    x for x in subgraph_gids
                    if x not in selected_subgraphs
                       and int(pg.nodes[x]["breadth"]) > 1
                       and len({(e[0], e[1]) for e in pg.nodes[x]["edgelist"]}.difference(target_edges)) == 0
                ]
                if len(cands) == 0:
                    cands = [
                        x for x in subgraph_gids
                        if x not in selected_subgraphs
                           and int(pg.nodes[x]["breadth"]) > 1
                    ]
                assert len(cands) > 0
                selected_subgraphs.add(random.choice(cands))
                cands = [
                    x for x in subgraph_gids
                    if x not in selected_subgraphs
                       and int(pg.nodes[x]["depth"]) > 1
                       and len({(e[0], e[1]) for e in pg.nodes[x]["edgelist"]}.difference(target_edges)) == 0
                ]
                if len(cands) == 0:
                    cands = [
                        x for x in subgraph_gids
                        if x not in selected_subgraphs
                           and int(pg.nodes[x]["depth"]) > 1
                    ]
                assert len(cands) > 0
                selected_subgraphs.add(random.choice(cands))
        elif breadth > 1 and max_breadth <= 1:
            cands = [
                x for x in subgraph_gids
                if x not in selected_subgraphs
                   and int(pg.nodes[x]["breadth"]) > 1
                   and len({(e[0], e[1]) for e in pg.nodes[x]["edgelist"]}.difference(target_edges)) == 0
            ]
            if len(cands) == 0:
                cands = [
                    x for x in subgraph_gids
                    if x not in selected_subgraphs
                       and int(pg.nodes[x]["breadth"]) > 1
                ]
            assert len(cands) > 0
            selected_subgraphs.add(random.choice(cands))
        elif depth > 1 and max_depth <= 1:
            cands = [
                x for x in subgraph_gids
                if x not in selected_subgraphs
                   and int(pg.nodes[x]["depth"]) > 1
                   and len({(e[0], e[1]) for e in pg.nodes[x]["edgelist"]}.difference(target_edges)) == 0
            ]
            if len(cands) == 0:
                cands = [
                    x for x in subgraph_gids
                    if x not in selected_subgraphs
                       and int(pg.nodes[x]["depth"]) > 1
                ]
            assert len(cands) > 0
            selected_subgraphs.add(random.choice(cands))
        #assert max_depth > 1 and max_breadth > 1
        return selected_subgraphs

    @staticmethod
    def _gen_task(item, task, taskdata, shots, pg, fill_shots):
        target_edges = {(e[0], e[1]) for e in pg.nodes[int(item["gid"])]["edgelist"]}
        train_ids = {td["gid"] for td in taskdata}

        # if len(train_ids) < shots:
        #     raise ValueError(f"Skipping task with too few train ids {task} {item}")

        try:
            selected_subgraphs = SubgraphDatasetGenerator.greedy_edge_cover(
                target_edges, list(train_ids),
                pg,
                breadth=pg.nodes[int(item["gid"])]["breadth"],
                depth=pg.nodes[int(item["gid"])]["depth"]
            )
        except AssertionError:
            raise ValueError(f"Skipping task with no valid edge cover {task} {item}")
        if len(selected_subgraphs) > shots:
            raise RuntimeError("Minimal edge cover larger than target shot number!")
        elif len(selected_subgraphs) == 0:
            raise ValueError(f"Skipping task with no edge cover {task} {item}")
        elif fill_shots and len(selected_subgraphs) < shots and len(list(train_ids.difference(selected_subgraphs))) > 0:
            if shots - len(selected_subgraphs) < 0 or shots - len(selected_subgraphs) > len(list(train_ids.difference(selected_subgraphs))):
                pass
            add_items = random.sample(list(train_ids.difference(selected_subgraphs)), shots - len(selected_subgraphs))
            selected_subgraphs.update(add_items)

        res = {
            "task": item,
            "shots": [td for td in taskdata if td["gid"] in selected_subgraphs],
        }
        assert item["question"] not in [td["question"] for td in res["shots"]]
        return res

    @staticmethod
    def extract_prompt_experiments(path, out_path, shots=5, limit=None, fill_shots=True, min_only=False):
        with open(path) as json_file:
            jsondata = json.load(json_file)

        pgraphs = dict()
        pgraphs_data = dict()
        for gdata in jsondata["powersetgraphs"]:
            pgraphs[(gdata["max_breadth"], gdata["max_depth"])] = nx.node_link_graph(json.loads(gdata["powersetgraph"]), edges="links")
            pgraphs_data[(gdata["max_breadth"], gdata["max_depth"])] = gdata

        benchmark_items_grouped = defaultdict(lambda: defaultdict(list))
        for split in ["train", "val", "test"]:
            for item in jsondata[split]:
                benchmark_items_grouped[(item["base_breadth"], item["base_depth"], item["id"])][split].append(item)

        train = []
        test = []

        min_edges_val = min([td["num_edges"] for task, taskdata in benchmark_items_grouped.items() for td in taskdata["val"]])

        for task, taskdata in benchmark_items_grouped.items():
            pg = pgraphs[task[:2]]
            #min_edges_val = min([td["num_edges"] for td in taskdata["val"]])
            print(min_edges_val)
            val_data = taskdata["val"]
            if min_only:
                val_data += taskdata["train"]
            for item in val_data:
                try:
                    restask = SubgraphDatasetGenerator._gen_task(item, task, [td for td in taskdata["train"] if td["question"] != item["question"] and td["query"] != item["query"]], shots, pg, fill_shots)
                    if min_only:
                        if item["num_edges"] <= min_edges_val:
                            #assert restask["task"]["num_edges"] < 4
                            train.append(restask)
                        else:
                            test.append(restask)
                    else:
                        train.append(restask)
                except ValueError as e:
                    logging.debug(str(e))
                    logging.debug(traceback.format_exc())
                    continue
            for item in taskdata["test"]:
                try:
                    restask = SubgraphDatasetGenerator._gen_task(item, task, taskdata["train"], shots, pg, fill_shots) # has to be taskdata["train"] as all shots come from train!
                    test.append(restask)
                except ValueError as e:
                    logging.debug(str(e))
                    logging.debug(traceback.format_exc())
                    continue

                #print(len(selected_subgraphs), task)
                #pass
        print(path, "Train:", len(train), "Test:", len(test))
        # if len(train) > limit:
        #     train = random.sample(train, limit)
        if limit is not None and len(test) > limit:
            test = random.sample(test, limit)
        print(path, "Train:", len(train), "Test:", len(test))

        jsondata["fixed_shots"] = {
            "train": train,
            "test": test,
        }

        print(list(jsondata.keys()))

        if out_path is None:
            out_path = path[:-5] + f"_{shots}{'_'+str(limit) if limit is not None else ''}_fixed.json"

        with open(out_path, "w") as f:
            json.dump(jsondata, f)

        return out_path
        #return train, test

        # subgraph_count = defaultdict(list)
        #
        # for task, taskdata in benchmark_items_grouped.items():
        #     pg = pgraphs[task[:2]]
        #     for item in taskdata["val"] + taskdata["test"]:
        #         target_edges = {(e[0], e[1]) for e in pg.nodes[int(item["gid"])]["edgelist"]}
        #         selected_subgraphs = SubgraphDatasetGenerator.greedy_edge_cover(target_edges, [td["gid"] for td in taskdata["train"]], pg)
        #         subgraph_count[len(selected_subgraphs)].append((task, selected_subgraphs))
        #         print(len(selected_subgraphs), task)
        #         pass
        #
        # for k, v in subgraph_count.items():
        #     print(k, len(v))
        pass


@dataclass
class PathCollection:
    path: str = None
    outpath: str = None
    trainoutpath: str = None
    valoutpath: str = None
    testoutpath: str = None
    valpath: str = None
    testpath: str = None

    @classmethod
    def from_args(cls, arguments):
        return cls(arguments.path,
                   arguments.outpath,
                   arguments.trainoutpath,
                   arguments.valoutpath,
                   arguments.testoutpath,
                   arguments.valpath,
                   arguments.testpath)


if __name__ == "__main__":
    argparser = ArgumentParser()
    #argparser.add_argument("--mode", type=str, default="single", choices=['single', 'multi', 'multi-mixed', "multi-and"])
    argparser.add_argument("--path", type=str, default=None)
    argparser.add_argument("--paths", type=str, nargs="+", default=None)
    argparser.add_argument("--existpaths", type=str, nargs="+", default=None)
    argparser.add_argument("--outpath", type=str, default=None)
    argparser.add_argument("--samples", type=int, default=None)
    argparser.add_argument("--threads", type=int, default=8)
    argparser.add_argument('--dedup', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--split', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--autofind', action=argparse.BooleanOptionalAction)
    argparser.add_argument("--limit", type=int, default=None)
    argparser.add_argument("--trainoutpath", type=str, default=None)
    argparser.add_argument("--valoutpath", type=str, default=None)
    argparser.add_argument("--testoutpath", type=str, default=None)
    argparser.add_argument("--endpoint", type=str, default=None)
    argparser.add_argument("--breadth", type=int, default=1)
    argparser.add_argument("--depth", type=int, default=1)

    argparser.add_argument('--subgraphs', action=argparse.BooleanOptionalAction)
    argparser.add_argument("--numparts", type=int, default=2)
    argparser.add_argument("--breadthlimit", nargs="+", type=int, default=None)
    argparser.add_argument("--depthlimit", nargs="+", type=int, default=None)
    argparser.add_argument("--maxpartintersections", type=float, default=0.0)
    #argparser.add_argument("--candidates", type=int, default=1)
    argparser.add_argument("--forcelabels", action=argparse.BooleanOptionalAction)

    argparser.add_argument('--toopenai', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--tolitgpt', action=argparse.BooleanOptionalAction)
    argparser.add_argument("--valpath", type=str, default=None)
    argparser.add_argument("--testpath", type=str, default=None)


    argparser.add_argument('--valtestrand', type=float, default=None)
    argparser.add_argument('--prompting', action=argparse.BooleanOptionalAction)
    argparser.add_argument("--shots", type=int, default=5)

    argparser.add_argument('--randrevprob', type=float, default=0.5)
    argparser.add_argument("--entitylist", type=str, default=None)
    #argparser.add_argument("--entitylistskip", type=int, default=0)
    argparser.add_argument('--bulkfetch', action=argparse.BooleanOptionalAction)

    arguments = argparser.parse_args()

    # threads = []
    # for breadth in range(1, 5):
    #     for depth in range(1, 6):
    #         if depth == 1 and breadth == 1:
    #             continue
    #         thread = Thread(target=SubgraphDatasetGenerator.bulk_fetch, args=(breadth, depth))
    #         thread.start()
    #         threads.append(thread)
    #
    # for t in threads:
    #     t.join()
    #     print("Thread finished")
    #
    # # SubgraphDatasetGenerator.bulk_fetch(4, 5)
    # exit(0)

    # SubgraphDatasetGenerator()
    # res = SubgraphDatasetGenerator.gen_single_pattern_query(1, depth=5, breadth=4, random_reverse_prob=0.5)
    # print(res)
    # exit(0)

    # SubgraphDatasetGenerator.convert_json_to_openai(os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "compositionality_subgraph_easy_prompt_5.json.zst"))
    # SubgraphDatasetGenerator.convert_json_to_openai(os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "compositionality_subgraph_medium_prompt_5.json.zst"))
    # SubgraphDatasetGenerator.convert_json_to_openai(os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "compositionality_subgraph_hard_prompt_5.json.zst"))
    # exit(0)

    
    #train, test = SubgraphDatasetGenerator.extract_prompt_experiments(os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "compositionality_subgraph_hard.json"))

    # res = SubgraphDatasetGenerator.gen_single_pattern_query(1, depth=3, breadth=3)
    # G = res[-1].subgraph([n for n in res[-2] if not n.startswith("?l")])
    # SubgraphDatasetGenerator.draw_graph(G, bfs_layout=True)
    # exit(0)


    if arguments.endpoint is not None:
        consts.sparql_endpoint = arguments.endpoint

    if arguments.prompting:
        SubgraphDatasetGenerator.extract_prompt_experiments(arguments.path, arguments.outpath, shots=arguments.shots, limit=arguments.limit)
        exit(0)
    elif arguments.bulkfetch:
        threads = []
        for breadth in range(1, 5):
            for depth in range(1, 6):
                if depth == 1 and breadth == 1:
                    continue
                thread = Thread(target=SubgraphDatasetGenerator.bulk_fetch, args=(breadth, depth))
                thread.start()
                threads.append(thread)

        for t in threads:
            t.join()
            print("Thread finished")

        # SubgraphDatasetGenerator.bulk_fetch(4, 5)
        exit(0)
    elif arguments.subgraphs:
        if arguments.split:
            paths = arguments.paths
            if paths is None:
                paths = list(glob.glob(os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "compositionality_samples_subgraph_*.jsonl.zst")))
            SubgraphDatasetGenerator.split_jsonl(
                paths=paths,
                existpaths=arguments.existpaths,
                path_out_base=arguments.outpath,
                sample_size=arguments.samples,
                shots=arguments.shots,
                val_test_random=arguments.valtestrand,
            )
        else:
            sgen = SubgraphDatasetGenerator(depth=arguments.depth, breadth=arguments.breadth, endpoint=arguments.endpoint)
            sgen.generate(
                out_path=arguments.outpath,
                num_samples=arguments.samples,
                limit=arguments.limit,
                #num_parts=arguments.numparts,
                breadth_limit=arguments.breadthlimit,
                depth_limit=arguments.depthlimit,
                #max_part_intersections=arguments.maxpartintersections,
                force_labels=arguments.forcelabels,
                num_threads=arguments.threads,
                random_reverse_prob=arguments.randrevprob,
                entity_list=arguments.entitylist,
                #entity_list_skip=arguments.entitylistskip,
            )
    elif arguments.toopenai or arguments.tolitgpt:
        gen = DatasetGenerator()
        paths = []
        if arguments.autofind:
            if arguments.path is None:
                basepath = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "comp")
            else:
                basepath = arguments.path

            prefix = ""
            suffix = ""
            if arguments.toopenai:
                prefix = "openai_"
                suffix = ".jsonl"
            elif arguments.tolitgpt:
                prefix = "litgpt_"
                suffix = ".json"

            for path in glob.glob(os.path.join(basepath, "dedup_compositionality_samples_multi_and_*.csv")):
                trainpath = os.path.join(basepath, "train_" + os.path.basename(path)[6:])
                valpath = os.path.join(basepath, "val_" + os.path.basename(path)[6:])
                testpath = os.path.join(basepath, "test_" + os.path.basename(path)[6:])

                train = os.path.join(basepath, prefix+"train_" + os.path.basename(path)[6:-4] + suffix)
                trainext = os.path.join(basepath, prefix+"ext_train_" + os.path.basename(path)[6:-4] + suffix)
                val = os.path.join(basepath, prefix+"val_" + os.path.basename(path)[6:-4] + suffix)
                test = os.path.join(basepath, prefix+"test_" + os.path.basename(path)[6:-4] + suffix)
                paths.append(PathCollection(path=trainpath, outpath=train))
                paths.append(PathCollection(path=valpath, outpath=val))
                paths.append(PathCollection(path=testpath, outpath=test))
                paths.append(PathCollection(path=trainpath, outpath=trainext, valpath=valpath, testpath=testpath))
        else:
            paths.append(PathCollection.from_args(arguments))
        for args in paths:
            if arguments.toopenai:
                gen.convert_to_openai(args.path, args.outpath, args.valpath, args.testpath)
            elif arguments.tolitgpt:
                gen.convert_to_litgpt(args.path, args.outpath, args.valpath, args.testpath)
    else:
        paths = []
        if arguments.autofind:
            if arguments.path is None:
                basepath = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "comp")
            else:
                basepath = arguments.path

            for path in glob.glob(os.path.join(basepath, "compositionality_samples_multi_and_*.csv")):
                outpath = os.path.join(basepath, "dedup_" + os.path.basename(path))
                trainoutpath = os.path.join(basepath, "train_" + os.path.basename(path))
                valoutpath = os.path.join(basepath, "val_" + os.path.basename(path))
                testoutpath = os.path.join(basepath, "test_" + os.path.basename(path))
                paths.append(PathCollection(path, outpath, trainoutpath, valoutpath, testoutpath))
        else:
            assert arguments.path is not None
            paths.append(PathCollection.from_args(arguments))

        gen = DatasetGenerator()
        for arg in paths:
            print(f"Processing {arg.path}")
            if arguments.dedup and arguments.split:
                gen.deduplicate_and_split(arg.path, arg.outpath, arg.trainoutpath, arg.valoutpath, arg.testoutpath, sample_limit=arguments.limit)
            elif arguments.dedup:
                gen.deduplicate(arg.path, arg.outpath, sample_limit=arguments.limit)
            elif arguments.split:
                gen.load_and_split(arg.path, arg.trainoutpath, arg.valoutpath, arg.testoutpath)
            else:
                raise ValueError("Should never happen")

