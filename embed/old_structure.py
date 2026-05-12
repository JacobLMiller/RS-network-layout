import networkx as nx
import numpy as np
from datetime import datetime
import re
import pylab as plt

from nltk.stem import WordNetLemmatizer

from data import set_arr_and_save, nx_to_array

def dimension_reduction(Y:np.ndarray, alg: str):
    if alg == "UMAP":
        from umap import UMAP 
        return UMAP(n_neighbors=15,min_dist=1e-2).fit_transform(Y)
    
    elif alg == "TSNE":
        from sklearn.manifold import TSNE
        return TSNE(perplexity=30).fit_transform(Y)
    
    elif alg == "MDS":
        from sklearn.manifold import MDS 
        return MDS().fit_transform(Y)

def embed_word(compute_embedding=False, srcpath = "processing/raw_data/dagstuhl_seminars_k3.graphml",
               embed_alg = "UMAP"):
    """
    Loads graph from srcpath and extracts the list of keywords in each node's "seminar_keywords" attribute. 
    Each keyword is actually a key phrase (possibly many words) and are treated as a sentence and embedded 
    using a sentence transformer based on the BERT natural language model. Each node is then assigned a position in 
    the transformer space based on the average of it's keywords' positions.

    compute_embedding: bool     Default: False. Whether or not to recompute the sentence embedding (can be a lengthy process depending on GPU). If there is no saved embedding, ignored. 
    srcpath: str                Default: "processing/dagstuhl_seminars_k3.graphml" The path of the source graph 
    embed_alg: str              Default: "UMAP". Which algorithm to use to compute the low dimensional embedding. Accepts ["UMAP", "TSNE", "MDS"]
    """

    GCC = nx.read_graphml(srcpath)
    print(f"SS graph has {GCC.number_of_nodes()}")

    words = set()
    lemmatizer = WordNetLemmatizer()

    to_rem = list()
    for node, data in GCC.nodes(data=True):

        mywords = list()
        if "seminar_keywords" in data:
            for word in data["seminar_keywords"].split("; "):
                word = word.lower()
                word = word.replace("-", " ").replace("/", " ").replace("_", " ")
                word = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", word)
                word = lemmatizer.lemmatize(word)
                
                if len(word) < 1: continue
                words.add(word)
                mywords.append(word)
            
            data['words'] = mywords
            print(mywords)
        else:
            to_rem.append(node)

    print(f"Removing {len(to_rem)} nodes without keywords")
    GCC.remove_nodes_from(to_rem)

    index_map = dict()

    import os 
    if compute_embedding or not os.path.exists("processing/embedding.npy"):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        sentences = list(words)

        #Sentences are encoded by calling model.encode()
        print("Begin keyword-level embedding (may take time depending on machine architecture)")
        embeddings = model.encode(sentences,show_progress_bar=True)

        print(f"Size of keyword embeddings: {embeddings.shape}")

        embed_map = dict(zip(sentences,embeddings))

        Y = np.zeros( (GCC.number_of_nodes(), embeddings.shape[1]) )

        for i,(node,data) in enumerate(GCC.nodes(data=True)):
            myvec = sum( embed_map[w] for w in data['words'] ) / len(data['words'])
            Y[i] = myvec
            index_map[node] = i
        np.save("processing/embedding.npy", Y)
    else: 
        Y = np.load("processing/embedding.npy")
        for i, node in enumerate(GCC.nodes()):
            index_map[node] = i

    print(Y)

    one_hot = embed_word_with_people(GCC)
    Y = np.concatenate((Y,one_hot), axis=1)

    print("Reducing node embedding to 2 dimensions")
    X = dimension_reduction(Y,embed_alg)
    X += np.random.normal(0,1e-7,X.shape)


    for v,data in GCC.nodes(data=True):
        data['words'] = ";".join(s for s in data["words"])

    print("Completed, saving graph")

    set_arr_and_save(X,GCC)


def update_dict(D: dict, items: list[any]):
    for item in items:
        print(item)
        if item not in D:
            D[item] = 0
        D[item] += 1
    print()

def assign_clusters():
    """
    From the output of embed_word(), read in embedding and assign two levels of nested clustering. 
    This is a wrapper for the get_clustering() function, see it for details
    """
    from compute_clusters import get_clustering 
    G = nx.read_graphml("processing/output_data/SS_k3_embedded.graphml")
    X = nx_to_array(G)
    # Y = np.load("processing/embedding.npy")
    c1,c2 = get_clustering(X,n=2)

    c2id = { v: c2[i] for i,v in enumerate(G.nodes()) }
    clusters2 = {c: dict() for c in range(np.max(c2)+1)}

    cid = {v: c1[i] for i,v in enumerate(G.nodes())}
    clusters1 = {c: dict() for c in range(np.max(c1)+1)}
    for v in cid:
        if G.nodes[v]["words"]:
            words = G.nodes[v]["words"].split(";")
            update_dict(clusters1[cid[v]], words)
            update_dict(clusters2[c2id[v]], words)            
            # clusters1[cid[v]].extend(words)
            # clusters2[c2id[v]].extend(words)

    from scipy.stats import mode 
    print(max(clusters1[0].values()))
    cluster_labels = [max(clusters1[c], key=clusters1[c].get) for c in clusters1]
    cluster_labels2 = [max(clusters2[c], key=clusters2[c].get) for c in clusters2]

    G.graph["cluster_labels_1"] = ";".join(cluster_labels)
    G.graph["cluster_labels_2"] = ";".join(cluster_labels2)

    for v,data in G.nodes(data=True):
        data["c1"] = cid[v]
        data["c2"] = c2id[v]

    set_arr_and_save(X,G)


def embed_people():
    G = nx.read_graphml("processing/output_data/SS_k3_embedded.graphml")
    index_map = {v: i for i,v in enumerate(G.nodes())}

    X = np.load("embedding.npy")

    P = nx.read_graphml("processing/raw_data/dagstuhl_people_k3.graphml")
    PY = np.zeros((P.number_of_nodes(), X.shape[1]))

    for i,p in enumerate(P):
        Sem = P.nodes[p]['seminars'].split(" ")
        PY[i] = sum(X[index_map[v]] for v in Sem) / len(Sem)

    # PX = UMAP().fit_transform(PY)
    PX = PY

    pos = {v: PX[i] for i,v in enumerate(P.nodes())}
    for v, data in P.nodes(data=True):
        data['x'] = pos[v][0]
        data['y'] = pos[v][1]

    nx.write_graphml_lxml(P, "src/application/static/data/PP_k3_embedded.graphml")
    nx.write_graphml_lxml(P, "processing/output_data/PP_k3_embedded.graphml")    


def shorten(title, num_words=4):
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.tag import pos_tag    

    words = word_tokenize(title)
    stop_words = set(stopwords.words("english"))
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    
    tagged = pos_tag(words)
    keywords = [word for word, tag in tagged if tag.startswith("N") or tag.startswith("J")]

    return " ".join(word for word in keywords[:num_words])



def shorten_titles():
    """
    To run this you will have to download a bunch of nltk databases
    """

    G = nx.read_graphml("processing/output_data/SS_k3_embedded.graphml")

    for _,data in G.nodes(data=True):
        title = data["seminar_name"]
        data["seminar_name_short"] = shorten(title, 5) if len(title.split(" ")) > 5 else data["seminar_name"]


    nx.write_graphml(G, "src/application/static/data/SS_k3_embedded.graphml")
    nx.write_graphml(G, "processing/output_data/SS_k3_embedded.graphml")    


def embed_word_with_people(GS):
    # GS = nx.read_graphml("processing/raw_data/dagstuhl_seminars_k3.graphml")
    GP = nx.read_graphml("processing/raw_data/dagstuhl_people_k3.graphml")

    #First, map people to integers (indices). Then, map seminars to lists of integers (people)
    pep_ind = dict(zip(GP.nodes(),range(GP.number_of_nodes())))
    sem_ind = dict(zip(
        GS.nodes(), 
        ([pep_ind[p] for p in GS.nodes[v]["attendees"].split(";") if p in pep_ind] 
           for v in GS.nodes()
        )
    ))
    
    one_hot = np.zeros((GS.number_of_nodes(), GP.number_of_nodes()))
    for i,v in enumerate(GS.nodes()):
        one_hot[i, sem_ind[v]] = 1.0

    from sklearn.decomposition import PCA 
    Y = PCA(n_components=50).fit_transform(one_hot)

    from umap import UMAP
    X = UMAP().fit_transform(Y)

    return X


if __name__ == "__main__":
    embed_word_with_people()