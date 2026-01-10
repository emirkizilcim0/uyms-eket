from EKET.summarize_data.summarize import summarize_data 
from EKET.utils import get_config, Path

def main():

    print("DEBUGGING Data ingestion has started.")
    result = summarize_data()                    # .json file is saved.
    print("DEBUGGING Data ingestion has ended.")
    
    

if __name__ == "__main__":
    main()