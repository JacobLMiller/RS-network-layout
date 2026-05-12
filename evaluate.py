import networkx as nx 
import numpy as np

import json 
import os

from copy import deepcopy
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist

from embed.similarity_matrices import embed_rich_by_abstract, embed_rich_by_keywords, jaccard_coathorship_similarity
from metrics import neighborhood_radius, SNKL, SNS, NHit, shape_faithfulness, avg_nbr_radius, neighborhood_preservation
from metrics import node_distribution, angular_resolution, edge_length_variation
from embed._utils import read_json_reprsentation


def get_embedded_G(G: nx.classes.graph.Graph, pos:dict):
    emb_G = deepcopy(G)
    for node, vec in pos.items():
        emb_G.nodes[node]['x'] = vec[0]
        emb_G.nodes[node]['y'] = vec[1]
    return emb_G

def get_embedding_from_pos(pos:dict, nodelist:list=None):
    if nodelist is None:
        return np.array(list(pos.values()))
    vecs = []
    for node in nodelist:
        vecs.append(pos[node])
    return np.array(vecs)

def get_embedding_from_G(G: nx.classes.graph.Graph):
    coords = []
    for node, data in G.nodes(data=True):
        coords.append([data['x'], data['y']])
    embedding = np.asarray(coords)
    return embedding

def get_apsp(G):
    d = np.zeros((G.number_of_nodes(), G.number_of_nodes()))

    dists = dict(nx.all_pairs_shortest_path_length(G))
    for u in dists:
        for v in range(u):
            if v not in dists[u]: 
                d[u,v] = -1 
                d[v,u] = -1
                continue 
            d[u, v] = dists[u][v]
            d[v, u] = dists[u][v]
    return d

def get_rich_dists(G):
    X = embed_rich_by_abstract(G)
    
    return pairwise_distances(X)


def get_sparse_dists(G):
    return 1 - jaccard_coathorship_similarity(G,sparse_type='sparse')

def get_ideal_locs(G,gname):
    G.graph['name'] = gname 

    d_all = get_apsp(G)
    np.save(f"matrices/{gname}_all.npy", d_all)
    
    d_rich = get_rich_dists(G)
    np.save(f"matrices/{gname}_rich.npy", d_rich)

    d_sparse = get_sparse_dists(G)
    np.save(f"matrices/{gname}_sparse.npy", d_sparse)    

    return d_all, d_rich, d_sparse


if __name__ == "__main__": 
    import tqdm
    embeddings_dir = "outputs"
    for graph in os.listdir(embeddings_dir):
        if not os.path.isdir("matrices"):
            os.mkdir("matrices")
        if not os.path.exists(f"matrices/{graph}_all.npy"):
            with open(f"outputs/{graph}/Semantic.json", 'r') as fdata:
                data = json.load(fdata)

            G = nx.node_link_graph(data,edges='links')
            G = nx.convert_node_labels_to_integers(G)

            D_all,D_rich,D_sparse = get_ideal_locs(G,graph)

        else:     
            D_all    = np.load(f"matrices/{graph}_all.npy")
            D_rich   = np.load(f"matrices/{graph}_rich.npy")
            D_sparse = np.load(f"matrices/{graph}_sparse.npy")

        metricdata = dict()
        for method in tqdm.tqdm(['Semantic', 
                                 'Union_FR', 
                                 'Union_UMAP',
                                 'Independent_S', 
                                 'Independent_R'
                                ]):
            
            mname = method.replace(".json", "").replace(" ", "_")
            metricdata[mname] = dict()

            with open(f"outputs/{graph}/{method}.json", 'r') as fdata:
                data = json.load(fdata)

            G = nx.node_link_graph(data,edges='links')
            G = nx.convert_node_labels_to_integers(G)
            
            X = np.zeros((G.number_of_nodes(), 2),dtype=np.float32)


            richindex = list()
            sparseindex = list()
            for i,v in enumerate(G.nodes()):
                X[i,0] = G.nodes[v]['x']
                X[i,1] = G.nodes[v]['y']
                if G.nodes[v]['type'] == "rich": richindex.append(i)
                else: sparseindex.append(i)

            median_dist = np.median(np.concatenate([pdist(X)]))

            dX = pairwise_distances(X[richindex])
            metricdata[mname]["Rich"] = {
                "SNS"   :  SNS(D_rich, X[richindex],X_is_dist=True, Y_is_dist=False).compute(), 
                "SNKL"  : SNKL(D_rich, dX,X_is_dist=True, Y_is_dist=True).compute(),
                "NH_7"  : NHit(D_rich, dX,X_is_dist=True, Y_is_dist=True).compute(7), 
                # "NH_15" : NHit(D_rich, dX,X_is_dist=True, Y_is_dist=True).compute(15),
                # "NH_50" : NHit(D_rich, dX,X_is_dist=True, Y_is_dist=True).compute(50),
                # "NH_100": NHit(D_rich, dX,X_is_dist=True, Y_is_dist=True).compute(100),
            }

            dX = pairwise_distances(X[sparseindex])
            metricdata[mname]["Sparse"] = {
                "SNS"   :  SNS(D_sparse, X[sparseindex],X_is_dist=True, Y_is_dist=False).compute(), 
                "SNKL"  : SNKL(D_sparse, dX,X_is_dist=True, Y_is_dist=True).compute(),
                "NH_7"  : NHit(D_sparse, dX,X_is_dist=True, Y_is_dist=True).compute(7), 
                # "NH_15" : NHit(D_sparse, dX,X_is_dist=True, Y_is_dist=True).compute(15),
                # "NH_50" : NHit(D_sparse, dX,X_is_dist=True, Y_is_dist=True).compute(50),
                # "NH_100": NHit(D_sparse, dX,X_is_dist=True, Y_is_dist=True).compute(100),
            }

            dX = pairwise_distances(X)
            metricdata[mname]["Between"] = {
                # "SNS"   :  SNS(D_all, X,X_is_dist=True, Y_is_dist=False).compute(), 
                # "SNKL"  : SNKL(D_all, dX,X_is_dist=True, Y_is_dist=True).compute(),
                # "NH_7"  : NHit(D_all, dX,X_is_dist=True, Y_is_dist=True).compute(7), 
                # "NH_15" : NHit(D_all, dX,X_is_dist=True, Y_is_dist=True).compute(15),
                # "NH_50" : NHit(D_all, dX,X_is_dist=True, Y_is_dist=True).compute(50),
                # "NH_100": NHit(D_all, dX,X_is_dist=True, Y_is_dist=True).compute(100),
                'Average Neighborhood radius (rich-view)': avg_nbr_radius(G, "rich") / median_dist,
                'Average Neighborhood radius (sparse-view)': avg_nbr_radius(G, "sparse") / median_dist,             
                "Shape faithfulness": shape_faithfulness(G),
                "Neighborhood Preservation (r = 2)": neighborhood_preservation(G,r=2)
            }      

            metricdata[mname]['Aesthetic'] = {
                "ND": node_distribution(G, X), 
                "AR": angular_resolution(G,X), 
                "ELV": edge_length_variation(G,X)
            } 

                  

        if not os.path.isdir("metric_out"): os.mkdir("metric_out")
        with open(f"metric_out/{graph}.json", 'w') as fdata:
            json.dump(metricdata, fdata,indent=4)
    