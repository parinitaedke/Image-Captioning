import re


# Cleans the captions by removing punctuation and converting the captions to lower case
class AnnotationCleaner:
    def __init__(self, annotations):
        self.annotations = annotations
        self.clean_annotations = []

    def clean(self):
        for annotation in self.annotations:
            self.clean_annotations.append(self.run(annotation))
        return self.clean_annotations

    @staticmethod
    def remove_punctuation(self, annotation):
        # remove the punctuations
        return re.sub(r'[^\w\s]', '', annotation)

    @staticmethod
    def lower(self, annotation):
        # convert annotations to lower case
        return annotation.lower()

    @staticmethod
    def remove_extra_space(self, annotation):
        # remove extra spaces
        return re.sub(r" {2,}", " ", annotation)

    def run(self, annotation):
        output = self.remove_punctuation(annotation)
        output = self.lower(output)
        output = self.remove_extra_space(output)
        return output

    def get_cleaned(self):
        return self.clean_annotations
