import os
import pandas as pd
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from rdflib import Graph
from sklearn.model_selection import train_test_split

def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data)
    n = len(a)
    mean = np.mean(a)
    se = np.std(a, ddof=1) / np.sqrt(n)
    h = se * 1.96
    return mean, mean-h, mean+h

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
            random_seed=42 + run
        )
        metrics = result.metric_results

        metrics_collected['mrrs'].append(metrics.get_metric('mrr'))
        metrics_collected['mrs'].append(metrics.get_metric('mr'))
        metrics_collected['hits10s'].append(metrics.get_metric('hits_at_10'))

        # For transition precision/recall:
        true_transitions = [tuple(x) for x in test_df[test_df['predicate'] == 'transitionsTo'][['subject', 'predicate', 'object']].values]
        # Here, replace prediction extraction with actual predicted transitions from your model.
        predicted_transitions = [tuple(x) for x in test_df[test_df['predicate'] == 'transitionsTo'][['subject', 'predicate', 'object']].values] # mock; replace this!
        precision, recall = transition_precision_recall(true_transitions, predicted_transitions)
        metrics_collected['transition_precision'].append(precision)
        metrics_collected['transition_recall'].append(recall)

    return metrics_collected

desktop = os.path.join(os.path.expanduser("~"), "Desktop")
excel_path = r'C:\Users\Bilal PC\Desktop\SmartbuildingKG.xlsx'
rdf_path = r'C:\Users\Bilal PC\Desktop\systemdatasb.ttl'
subject_col = 'Subject'
predicate_col = 'Predicate'
entity_col = 'Entity'
has_state_col = 'hasState'
state_col = 'State'
models = ["TransE", "TransH", "ComplEx"]
n_runs = 5

triples_tktrdf = prepare_triples_from_excel(excel_path, subject_col, predicate_col, entity_col, has_state_col, state_col)
triples_rdf = prepare_triples_from_rdf(rdf_path)

evaluation = {}
for model in models:
    print(f"Evaluating {model} on TKTrdf KG...")
    tktrdf_metrics = run_evaluation_multiple(triples_tktrdf, model, n_runs)
    print(f"Evaluating {model} on RDF KG...")
    rdf_metrics = run_evaluation_multiple(triples_rdf, model, n_runs)

    key_names = [
        ('mrrs', 'MRR'),
        ('mrs', 'MR'),
        ('hits10s', 'Hits@10'),
        ('transition_precision', 'Transition Precision'),
        ('transition_recall', 'Transition Recall'),
    ]
    eval_entry = {}
    for key, label in key_names:
        tk_mean, tk_low, tk_up = mean_confidence_interval(tktrdf_metrics[key])
        rdf_mean, rdf_low, rdf_up = mean_confidence_interval(rdf_metrics[key])
        eval_entry[f"{label} TKTrdf"] = f"{tk_mean:.4f} ({tk_low:.4f}-{tk_up:.4f})"
        eval_entry[f"{label} RDF"] = f"{rdf_mean:.4f} ({rdf_low:.4f}-{rdf_up:.4f})"
    evaluation[model] = eval_entry

metrics_order = [lab for _, lab in key_names]
table_rows = []
for metric_label in metrics_order:
    row = [metric_label]
    for model in models:
        row.append(evaluation[model].get(f"{metric_label} RDF"))
        row.append(evaluation[model].get(f"{metric_label} TKTrdf"))
    table_rows.append(row)

columns = ["Metric"]
for model in models:
    columns.extend([f"{model} RDF", f"{model} TKTrdf"])

results_df = pd.DataFrame(table_rows, columns=columns)
output_file = os.path.join(desktop, 'kg_evaluation_with_ci.xlsx')
results_df.to_excel(output_file, index=False)
print(f"Evaluation saved to {output_file}")
print(results_df)

