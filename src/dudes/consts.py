from lemon.lemon import PartOfSpeech

#spacy_model = "en_core_web_lg"
spacy_model = "en_core_web_trf"

fp_ratio = 10.0

#dbpedia_spotlight_endpoint = None
dbpedia_spotlight_endpoint = 'http://localhost:2222/rest' # https://api.dbpedia-spotlight.org

#sparql_endpoint = "http://dbpedia.org/sparql"
sparql_endpoint = "http://localhost:8890/sparql"
#sparql_endpoint = http://client.linkeddatafragments.org/#datasources=http%3A%2F%2Ffragments.dbpedia.org%2F2016-04%2Fen

generaltype = "rdf:type/(rdfs:subClassOf|owl:equivalentClass)*"

compositionality_bad_props = ["rdf:type", "dbp:title", "dbo:type", "dbp:name", "dbo:firstAscentPerson", "dbo:foundedBy",
                              "dbo:goldMedalist", "dbo:ground", "dbo:instrument", "dbo:league", "dbo:literaryGenre",
                              "dbo:locatedInArea", "dbo:locationCity", "dbo:numberOfEmployees", "dbo:numberOfEpisodes",
                              "dbo:numberOfVisitors", "dbo:occupation", "dbo:programme", "dbp:borders", "dbp:calories",
                              "dbp:carbs", "dbp:crewMembers", "dbp:employees", "dbp:entranceCount", "dbp:fifaMin",
                              "dbp:industry", "dbp:numEmployees", "dbp:satelliteOf", "dbp:satellites", "dbp:studio",
                              "dbp:trailheads", "dbp:incumbent", "dbo:activeYearsEndDate"]
compositionality_bad_words = ["cave", "volcano", "king", "president", "common root", "wife", "husband", "novelist", "name of the university"]
compositionality_bad_words_predicative = ["cave", "people", "everyone", "mensch", "eukaryote", "species", "person", "city", "agent", "place", "wife", "husband"] # person is too general, not useful as a specification
compositionality_nontype_props = [
    (PartOfSpeech.ADJECTIVE, 'http://dbpedia.org/ontology/country'),
    (PartOfSpeech.NOUN, 'http://dbpedia.org/ontology/industry'),
    (PartOfSpeech.ADJECTIVE, 'http://dbpedia.org/ontology/nationality'),
]


rpc_host = "localhost"
rpc_port = 8042
rpc_threads = 20

question_words = {"who", "which", "when", "where", "what", "why", "whom", "whose", "how"}

some_relation_words = {"with", "have"}


count_keywords = [["how", "many"], ["how", "often"]]

vague_temporal_keywords = [["recently"], ["just"], ["some", "time", "ago"], ["long", "time", "ago"]]

ask_keywords = ["is", "are", "does", "do", "did", "was", "were", "has", "have", "had", "can", "could", "will", "would",
                "shall", "should", "may", "might", "must"]

comp_gt_keywords = ["more", "greater", "later"]
comp_gt_keywords_no_than = ["after"]
comp_lt_keywords = ["less", "fewer", "earlier"]
comp_lt_keywords_no_than = ["before"]

comp_keywords = comp_gt_keywords + comp_gt_keywords_no_than + comp_lt_keywords + comp_lt_keywords_no_than

top_strong_keywords = ["most", "first", "1st"]
top_weak_keywords = ["least", "last"]

conj_keywords = ["and", "or"]

than_keywords = ["than"]

pronouns = ["i", "me", "my", "mine", "you", "your", "yours", "he", "him", "his", "she", "her", "hers", "it", "its",
            "that", "those", "these", "this", "they", "them", "their", "theirs", "who", "whom", "whose", "which",
            "what", "where", "when", "why", "how"]

special_words = set.union(*(
    question_words,
    some_relation_words,
    sum(count_keywords, []),
    sum(vague_temporal_keywords, []),
    ask_keywords,
    comp_gt_keywords,
    comp_lt_keywords,
    comp_gt_keywords_no_than,
    comp_lt_keywords_no_than,
    top_strong_keywords,
    top_weak_keywords,
    conj_keywords,
    than_keywords,
    pronouns
))

uri_attrs = ["reference", "bound_to", "has_value", "degree", "on_property", "property_domain", "property_range"]

#question_words.union(some_relation_words).union(["most", "least", "many"])
openai_subgraph_prompt = 'Given a natural language question about a specific entity, generate a SPARQL query that retrieves relevant information from DBpedia.'
openai_subgraph_prompt_nonft = 'Given a natural language question about a specific entity, generate a SPARQL query that retrieves relevant information from DBpedia. Answer solely with the corresponding correct SPARQL query.'
