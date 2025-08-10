import os
import json
import csv
from collections import defaultdict
from collections import Counter


def convert_instacart(target_dir="json_dir", raw_data_dir = "raw_data/instacart", min_rich_deg=150, min_sparse_deg=25):
    """
    Save instacart dataset
    """


    # Load product data

    with open(os.path.join(raw_data_dir, "aisles.csv")) as f:
        reader = csv.DictReader(f)
        aisles = {row['aisle_id']: row['aisle'] for row in reader}

    with open(os.path.join(raw_data_dir, "departments.csv")) as f:
        reader = csv.DictReader(f)
        departments = {row['department_id']: row['department'] for row in reader}

    with open(os.path.join(raw_data_dir, "products.csv")) as f:
        reader = csv.DictReader(f)
        products = {
            row['product_id']: {
                'name': row['product_name'],
                'department': departments[row['department_id']],
                'aisle': aisles[row['aisle_id']],
            }
            for row in reader
        }

    del aisles
    del departments
        


    # Load product-user graph data

    with open(os.path.join(raw_data_dir, "order_products__prior.csv")) as f:
        reader = csv.DictReader(f)
        order_products_prior = {row['order_id']: row['product_id'] for row in reader}

    with open(os.path.join(raw_data_dir, "order_products__train.csv")) as f:
        reader = csv.DictReader(f)
        order_products_train = {row['order_id']: row['product_id'] for row in reader}

    with open(os.path.join(raw_data_dir, "orders.csv")) as f:
        reader = csv.DictReader(f)
        orders = {
            row['order_id']: {
                'user_id': row['user_id'],
                'eval_set': row['eval_set']
            }
            for row in reader
        }

    prod_orders = defaultdict(list)

    for order_id, order_data in orders.items():
        if order_data['eval_set'] == 'prior':
            order_dict = order_products_prior
        elif order_data['eval_set'] == 'train':
            order_dict = order_products_train
        else:
            continue


        product_id = order_dict[order_id]
        s_id = f"s_{order_data['user_id']}"
        prod_orders[product_id].append(s_id)

    # Weighted edges
    for prod_id, s_ids in prod_orders.items():
        prod_orders[prod_id] = [(s_id, weight) for s_id, weight in Counter(s_ids).items()]



    # Only keep rich nodes with minimum degree min_rich_deg
    top_prods = ([prod_id for prod_id in prod_orders if len(prod_orders[prod_id]) >= min_rich_deg])
    prod_orders = {prod_id: prod_orders[prod_id] for prod_id in top_prods}

    # Only keep sparse nodes with minimum degree min_sparse_deg
    s_id_degree = defaultdict(int)
    for s_ids_w_wt in prod_orders.values():
        for s_id, _ in s_ids_w_wt:
            s_id_degree[s_id] += 1

    print(f"Number of initial s_ids: {len(s_id_degree)}")
    # print({s_id : degree for s_id, degree in s_id_degree.items() if degree < 2})

    isolated_prods = list()
    for prod_id in prod_orders:
        prod_orders[prod_id] = [(s_id, weight) for s_id, weight in prod_orders[prod_id] if s_id_degree[s_id] >= min_sparse_deg]
        if not prod_orders[prod_id]:
            isolated_prods.append(prod_id)
    # Remove rich nodes that are isolated as a result of the previous step
    for prod_id in isolated_prods:
        prod_orders.pop(prod_id)

    print("After filtering:")
    print("----------------")
    s_id_degree = defaultdict(int)
    for s_ids in prod_orders.values():
        for s_id, _ in s_ids:
            s_id_degree[s_id] += 1

    print(f"Number of final s_ids: {len(s_id_degree)}")
    # print({s_id : degree for s_id, degree in s_id_degree.items() if degree < 2})

    # Rich data
    rich_data = list()
    for product_id, s_ids_w_wt in prod_orders.items():
        data = list(products[product_id].values())
        rich_data.append(
            {
                "id" : f"r_{product_id}",
                "data": data,
                "s_ids": s_ids_w_wt
            }
        )

    # Sparse data
    unique_s_ids = sorted(set(s_id for r_data in rich_data for s_id, _ in r_data['s_ids']),
                            key=lambda x: int(x.split('_')[1]))
    sparse_data: list = [{'id': s_id} for s_id in unique_s_ids]



    data_dict = {
        '_processing_comments' : {
            "Min rich deg": min_rich_deg,
            "Min sparse deg": min_sparse_deg
        },
        'rich': rich_data,
        'sparse': sparse_data,
    }

    file_name = "instacart.json"
    with open(os.path.join(target_dir, file_name), 'w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, ensure_ascii=False, indent=4)
    print(f"Dictionary saved to {file_name}")




if __name__ == "__main__":
    raw_data_dir = "raw_data/instacart"

    target_dir = "json_data"
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    convert_instacart(target_dir, raw_data_dir)