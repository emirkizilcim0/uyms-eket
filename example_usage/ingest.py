from EKET.data_ingest.ingestion import data_ingestion
from EKET.utils import get_config, Path

def main():
    print("DEBUGGING Documents in data folder:\n")
    for f in Path(get_config()['DATA_DIR']).glob('*'):
        print(f"- {f.name}")
    
    
    print("DEBUGGING Data ingestion has started.")
    result = data_ingestion()       # .json file is saved.
    print("DEBUGGING Data ingestion has ended.")
    
    

if __name__ == "__main__":
    main()