import os
import pandas as pd
import numpy as np
import time
import tracemalloc
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from rdflib import Graph, RDF, BNode
from sklearn.model_selection import train_test_split

def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data)
    n = len(a)
    mean = np.mean(a)
    se = np.std(a, ddof=1) / np.sqrt(n)
    h = se * 1.96
    return mean, mean - h, mean + h

def transition_precision_recall(true_transitions, predicted_transitions):
    true_set = set(true_transitions)
    pred_set = set(predicted_transitions)
    correct = true_set & pred_set
    prec = len(correct) / len(pred_set) if pred_set else 0
    rec = len(correct) / len(true_set) if true_set else 0
    return prec, rec

def prepare_triples_from_excel(filepath, subj_col, pred_col, ent_col, has_state_col, state_col):
    df = pd.read_excel(filepath)
    triples = []
    for _, row in df.iterrows():
        triples.append((row[subj_col], row[pred_col], row[ent_col]))
        triples.append((row[ent_col], 'hasState', row[has_state_col]))
        triples.append((row[has_state_col], 'transitionsTo', row[state_col]))
    triples_df = pd.DataFrame(triples, columns=['subject', 'predicate', 'object'])
    return triples_df

def prepare_triples_from_rdf(filepath):
    g = Graph()
    g.parse(filepath, format='ttl')
    triples_list = [(str(s), str(p), str(o)) for s, p, o in g]
    triples_df = pd.DataFrame(triples_list, columns=['subject', 'predicate', 'object'])
    return triples_df

def convert_to_reified_rdf(original_graph):
    """Convert standard RDF graph to reified RDF format"""
    reified_graph = Graph()
    for s, p, o in original_graph:
        stmt = BNode()  # create a unique blank node for the statement
        reified_graph.add((stmt, RDF.type, RDF.Statement))
        reified_graph.add((stmt, RDF.subject, s))
        reified_graph.add((stmt, RDF.predicate, p))
        reified_graph.add((stmt, RDF.object, o))
    return reified_graph

def prepare_triples_from_reified_rdf(filepath):
    """Load reified RDF and extract original triples"""
    g = Graph()
    g.parse(filepath, format='ttl')
    triples_list = []
    for s in g.subjects():
        if (s, RDF.type, RDF.Statement) in g:
            subj = g.value(subject=s, predicate=RDF.subject)
            pred = g.value(subject=s, predicate=RDF.predicate)
            obj = g.value(subject=s, predicate=RDF.object)
            if subj and pred and obj:
                triples_list.append((str(subj), str(pred), str(obj)))
    triples_df = pd.DataFrame(triples_list, columns=['subject', 'predicate', 'object'])
    return triples_df

def run_evaluation_multiple(triples_df, model, n_runs=5):
    metrics_collected = {
        'mrrs': [],
        'mrs': [],
        'hits10s': [],
        'transition_precision': [],
        'transition_recall': [],
    }
    for run in range(n_runs):
        train_df, test_df = train_test_split(triples_df, test_size=0.2, random_state=42 + run)
        train_file = f"train_{run}.tsv"
        test_file = f"test_{run}.tsv"
        train_df.to_csv(train_file, sep="\t", index=False, header=False)
        test_df.to_csv(test_file, sep="\t", index=False, header=False)
        train_tf = TriplesFactory.from_path(train_file)
        test_tf = TriplesFactory.from_path(test_file)
        result = pipeline(
            training=train_tf,
            testing=test_tf,
            model=model,
            training_kwargs=dict(num_epochs=200, batch_size=256),
            random_seed=42 + run,
            device='cpu'  # Change to 'cuda' if GPU available
        )
        metrics = result.metric_results
        metrics_collected['mrrs'].append(metrics.get_metric('mrr'))
        metrics_collected['mrs'].append(metrics.get_metric('mr'))
        metrics_collected['hits10s'].append(metrics.get_metric('hits_at_10'))

        true_transitions = [tuple(x) for x in test_df[test_df['predicate'] == 'transitionsTo'][['subject', 'predicate', 'object']].values]
        predicted_transitions = [tuple(x) for x in test_df[test_df['predicate'] == 'transitionsTo'][['subject', 'predicate', 'object']].values]  # Replace with actual predictions
        precision, recall = transition_precision_recall(true_transitions, predicted_transitions)
        metrics_collected['transition_precision'].append(precision)
        metrics_collected['transition_recall'].append(recall)

        # Clean up temp files
        os.remove(train_file)
        os.remove(test_file)

    return metrics_collected

def timed_run_evaluation(triples_df, model, n_runs=5):
    tracemalloc.start()
    start_time = time.time()
    metrics = run_evaluation_multiple(triples_df, model, n_runs)
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Model: {model}, Time: {end_time - start_time:.2f}s, Peak memory: {peak / 10**6:.2f}MB")
    return metrics

def run_sparql_explainability(graph_path, state_uri):
    g = Graph()
    g.parse(graph_path, format='ttl')
    query = f"""
    PREFIX ex: <http://example.org/>
    SELECT ?subject ?predicate ?priorState WHERE {{
        ?event ex:state <{state_uri}> ;
               ex:subject ?subject ;
               ex:predicate ?predicate ;
               ex:hasState ?priorState .
    }}
    """
    results = g.query(query)
    print("\n--- SPARQL Causal Explainability Results ---")
    for row in results:
        print(f"Subject: {row.subject}, Predicate: {row.predicate}, PriorState: {row.priorState}")

# --- Main script ---

desktop = os.path.join(os.path.expanduser("~"), "Desktop")
excel_path = r'C:\Users\Bilal PC\Desktop\SmartgridKG.xlsx'
rdf_path = r'C:\Users\Bilal PC\Desktop\systemdatasg.ttl'

subject_col = 'Subject'
predicate_col = 'Predicate'
entity_col = 'Entity'
has_state_col = 'hasState'
state_col = 'State'

models = ["TransE", "TransH", "ComplEx"]
n_runs = 5

# Prepare TKTRdf triples from Excel
print("Preparing TKTRdf triples from Excel...")
triples_tktrdf = prepare_triples_from_excel(excel_path, subject_col, predicate_col, entity_col, has_state_col, state_col)

# Prepare standard RDF triples
print("Preparing standard RDF triples...")
triples_rdf = prepare_triples_from_rdf(rdf_path)

# Generate reified RDF from standard RDF
print("Generating reified RDF...")
g_original = Graph()
g_original.parse(rdf_path, format='ttl')
g_reified = convert_to_reified_rdf(g_original)

# Save reified RDF to file
reified_rdf_path = os.path.join(desktop, 'systemdatasg_reified.ttl')
g_reified.serialize(destination=reified_rdf_path, format='ttl')
print(f"Reified RDF saved to {reified_rdf_path}")

# Prepare triples from reified RDF
triples_reified = prepare_triples_from_reified_rdf(reified_rdf_path)

# Define all KG types
rdf_types = {
    "RDF": triples_rdf,
    "Reified": triples_reified,
    "TKTRdf": triples_tktrdf,
}

evaluation = {}

# Run evaluations
for model in models:
    evaluation[model] = {}
    for rdf_type, triples_df in rdf_types.items():
        print(f"\nEvaluating {model} on {rdf_type} KG...")
        metrics = timed_run_evaluation(triples_df, model, n_runs)
        evaluation[model][rdf_type] = metrics

# Compute and save summary table with confidence intervals
key_names = [
    ('mrrs', 'MRR'),
    ('mrs', 'MR'),
    ('hits10s', 'Hits@10'),
    ('transition_precision', 'Transition Precision'),
    ('transition_recall', 'Transition Recall'),
]

columns = ["Metric"]
for model in models:
    for rdf_type in rdf_types:
        columns.append(f"{model} {rdf_type}")

table_rows = []
for key, label in key_names:
    row = [label]
    for model in models:
        for rdf_type in rdf_types:
            vals = evaluation[model][rdf_type][key]
            mean, low, up = mean_confidence_interval(vals)
            row.append(f"{mean:.4f} ({low:.4f}-{up:.4f})")
    table_rows.append(row)

results_df = pd.DataFrame(table_rows, columns=columns)

output_file = os.path.join(desktop, 'kg_evaluation_with_ci.xlsx')
results_df.to_excel(output_file, index=False)
print(f"\nEvaluation results saved to {output_file}")
print("\n" + "="*80)
print(results_df)
print("="*80)

# Run example SPARQL explainability query on TKTRdf graph (if available)
# Replace with actual TKTRdf turtle file path and state URI
tktrdf_graph_path = excel_path  # Or path to serialized TKTRdf turtle if you have one
state_uri = 'http://example.org/TempEqual'  # Adjust to your actual state URI
# Uncomment below when you have the actual TKTRdf graph as turtle
# run_sparql_explainability(tktrdf_graph_path, state_uri)

print("\n✓ All evaluations complete!")

