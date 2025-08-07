import os
import json
import csv
from collections import defaultdict


def convert_instacart(target_dir="json_data", raw_data_dir = "raw_data/instacart"):
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




    # Rich data
    rich_data = list()
    for product_id, s_ids in prod_orders.items():
        data = list(products[product_id].values())
        rich_data.append(
            {
                "id" : f"r_{product_id}",
                "data": data,
                "s_ids": s_ids
            }
        )

    # Sparse data
    unique_s_ids = sorted(set(s_id for r_data in rich_data for s_id in r_data['s_ids']),
                            key=lambda x: int(x.split('_')[1]))
    sparse_data: list = [{'id': s_id} for s_id in unique_s_ids]



    data_dict = {
        'rich': rich_data,
        'sparse': sparse_data,
    }

    file_name = "instacart_graph.json"
    with open(os.path.join(target_dir, file_name), 'w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, ensure_ascii=False, indent=4)
    print(f"Dictionary saved to {file_name}")


if __name__ == "__main__":
    raw_data_dir = "raw_data/instacart"

    target_dir = "json_data"
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    convert_instacart(target_dir, raw_data_dir)