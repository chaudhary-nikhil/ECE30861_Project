from enum import Enum
import re

# A URL consists of the URL string and a category: dataset, model, code.
# Relevant regexes:
#   Dataset URL (assuming for now that Hugging Face is the only source): https:\/\/huggingface\.co\/datasets\/(\w+\/?)+
#   Model URL (only run if Dataset fails): https:\/\/huggingface\.co\/(\w+\/?)+
#   Code URL (assuming for now that GitHub is the only source): https:\/\/github.com\/(\w+\/?)+


class UrlCategory(Enum):
    DATASET = 1
    MODEL = 2
    CODE = 3
    INVALID = 4


def determine_category(link: str) -> UrlCategory:
    datasetRegex = r"https:\/\/huggingface\.co\/datasets\/(\w+\/?)+"
    modelRegex = r"https:\/\/huggingface\.co\/(\w+\/?)+"
    codeRegex = r"https:\/\/github.com\/(\w+\/?)+"

    if re.match(datasetRegex, link):
        return UrlCategory.DATASET
    elif re.match(modelRegex, link):
        return UrlCategory.MODEL
    elif re.match(codeRegex, link):
        return UrlCategory.CODE
    else:
        return UrlCategory.INVALID


class Url:
    def __init__(self, link: str, category: UrlCategory = UrlCategory.INVALID):
        self.link = link
        # If given an invalid category, determine the category ourselves. If it actually is invalid, print an error
        if category == UrlCategory.INVALID:
            self.category = determine_category(link)
            if self.category == UrlCategory.INVALID:
                print(f"{link} Invalid URL: Not a dataset, model or code URL")
        else:
            self.category = category

    def __str__(self) -> str:
        return str(self.link + " Category: " + self.category.__str__())
