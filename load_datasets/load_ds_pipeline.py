from load_datasets.convert_to_json import convert_dagstuhl, convert_vast, convert_vispub
from load_datasets.convert_netflix_to_json import convert_netflix
from load_datasets.convert_instacart_to_json import convert_instacart
import os 

def main():
    
    target_dir = "json_data"
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    convert_dagstuhl()
    convert_dagstuhl(1990,2015,"dagstuhl-before2015")
    convert_dagstuhl(2016,2030,"dagstuhl-after2015")

    convert_vispub()
    convert_vast()

    convert_netflix(target_dir)

    convert_instacart(target_dir)
    
if __name__ == '__main__':
    main()