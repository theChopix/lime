HAM_TAG = "OK"
SPAM_TAG = "SPAM"


# transforms content of classification-file into dictionary preserving the same information
def read_classification_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
        lines = content.split("\n")
        classification_dict = {}

        for line in lines:
            line_content = line.split(" ")

            if len(line_content) == 2:
                filename = line_content[0]
                classification = line_content[1]
                classification_dict[filename] = classification

    return classification_dict
