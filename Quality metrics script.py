import pandas as pd
import networkx as nx
import numpy as np
from collections import Counter
from rdflib import Graph

def rdf_to_networkx_graph(rdf_path):
    """Convert RDF/Turtle file to NetworkX graph"""
    g = Graph()
    g.parse(rdf_path, format='ttl')
    G = nx.Graph()
    for s, p, o in g:
        G.add_edge(str(s), str(o), predicate=str(p))
    return G

def get_entities_and_relations_from_rdf(rdf_path):
    """Extract entities and relations from RDF/Turtle file"""
    g = Graph()
    g.parse(rdf_path, format='ttl')
    entities = set()
    relations = set()
    for s, p, o in g:
        entities.add(str(s))
        entities.add(str(o))
        relations.add(str(p))
    return entities, relations

def analyze_rdf_graph(rdf_path, kg_name):
    """Analyze RDF graph from Turtle file"""
    print(f"\n{'='*60}")
    print(f"Analyzing {kg_name}: {rdf_path}")
    print(f"{'='*60}")
    
    G = rdf_to_networkx_graph(rdf_path)
    entities, relations = get_entities_and_relations_from_rdf(rdf_path)

    # Compute metrics
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    num_components = nx.number_connected_components(G)
    degrees = np.array([d for n, d in G.degree()])
    degree_std = np.std(degrees) if degrees.size > 0 else 0
    
    rel_counts = [data['predicate'] for u, v, data in G.edges(data=True)]
    rel_counter = Counter(rel_counts)
    rel_std = np.std(list(rel_counter.values())) if rel_counter else 0
    
    clustering_coeff = nx.average_clustering(G) if G.number_of_nodes() > 0 else 0

    # Print results
    print(f"Coverage - Entities: {len(entities)}")
    print(f"Coverage - Relations: {len(relations)}")
    print(f"Average Degree: {avg_degree:.4f}")
    print(f"Connected Components: {num_components}")
    print(f"Degree Std Dev: {degree_std:.4f}")
    print(f"Relation Std Dev: {rel_std:.4f}")
    print(f"Clustering Coefficient: {clustering_coeff:.4f}")

    return {
        'kg_name': kg_name,
        'entity_count': len(entities),
        'relation_types': len(relations),
        'average_degree': avg_degree,
        'connected_components': num_components,
        'degree_stddev': degree_std,
        'relation_stddev': rel_std,
        'clustering_coefficient': clustering_coeff
    }

def analyze_tktrdf_excel(excel_path, kg_name, entity_col='Entity', subject_col='Subject', predicate_col='Predicate'):
    """Analyze TKTRdf graph from Excel file"""
    print(f"\n{'='*60}")
    print(f"Analyzing {kg_name}: {excel_path}")
    print(f"{'='*60}")
    
    df = pd.read_excel(excel_path)
    G = nx.Graph()

    # Add edges from Excel data
    for index, row in df.iterrows():
        s = row[subject_col]
        p = row[predicate_col]
        o = row[entity_col]
        G.add_edge(str(s), str(o), predicate=str(p))

    entities = list(G.nodes)
    relations = [data['predicate'] for u, v, data in G.edges(data=True)]

    # Compute metrics
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    num_components = nx.number_connected_components(G)
    degrees = np.array([d for n, d in G.degree()])
    degree_std = np.std(degrees) if degrees.size > 0 else 0
    
    rel_counter = Counter(relations)
    rel_std = np.std(list(rel_counter.values())) if rel_counter else 0
    
    clustering_coeff = nx.average_clustering(G) if G.number_of_nodes() > 0 else 0

    # Print results
    print(f"Coverage - Entities: {len(entities)}")
    print(f"Coverage - Relations: {len(set(relations))}")
    print(f"Average Degree: {avg_degree:.4f}")
    print(f"Connected Components: {num_components}")
    print(f"Degree Std Dev: {degree_std:.4f}")
    print(f"Relation Std Dev: {rel_std:.4f}")
    print(f"Clustering Coefficient: {clustering_coeff:.4f}")

    return {
        'kg_name': kg_name,
        'entity_count': len(entities),
        'relation_types': len(set(relations)),
        'average_degree': avg_degree,
        'connected_components': num_components,
        'degree_stddev': degree_std,
        'relation_stddev': rel_std,
        'clustering_coefficient': clustering_coeff
    }

# --- Main Script ---

# File paths - Update these with your actual file paths
rdf_path = r'C:\Users\Bilal PC\Desktop\systemdatasg.ttl'
reified_rdf_path = r'C:\Users\Bilal PC\Desktop\systemdatasg_reified.ttl'
tktrdf_excel_path = r'C:\Users\Bilal PC\Desktop\SmartgridKG.xlsx'

# Analyze all three KGs
results = []

# Analyze RDF
rdf_metrics = analyze_rdf_graph(rdf_path, "RDF")
results.append(rdf_metrics)

# Analyze Reified RDF
reified_metrics = analyze_rdf_graph(reified_rdf_path, "Reified RDF")
results.append(reified_metrics)

# Analyze TKTRdf
tktrdf_metrics = analyze_tktrdf_excel(tktrdf_excel_path, "TKTRdf")
results.append(tktrdf_metrics)

# Create summary table
summary_df = pd.DataFrame(results)
print(f"\n{'='*60}")
print("SUMMARY TABLE - Quality Metrics Comparison")
print(f"{'='*60}")
print(summary_df.to_string(index=False))

# Save to Excel
output_path = r'C:\Users\Bilal PC\Desktop\kg_quality_metrics_sg.xlsx'
summary_df.to_excel(output_path, index=False)
print(f"\nResults saved to: {output_path}")


