from EKET.generate_answer.generate_answer import generate_answer_from_question


def main():

    print("DEBUGGING generating of answer from the question.\n")
    answer = generate_answer_from_question() # holds ".json" file for answer. 
    print("\nDEBUGGING Generating is over.")

if __name__ == '__main__':
    main()