# General Python Imports
from typing import List, Tuple, Union
import shutil
import os

# Regex Import
import re

# PDF Processing Imports
import pandas as pd
import PyPDF2

# Spacy Language Model Import
import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler


class DocumentExtractor(object):
    """Class with set of functionalities to extract certain Info. from Financial Documents."""

    def __init__(self):
        """Constructor for the class DocumentExtractor()."""

        # Dabases with Indian Names/ Indian-Centric Company Names.
        # The names within this database will be used to improve
        # Named Entity Recognition (NER) results on Indian names/ Companies,
        # based on Spacy's Default English-Language Model.

        self.name_databases = ["Dataset-Indian-Names/Indian-Male-Names.csv",
                               "Dataset-Indian-Names/Indian-Female-Names.csv",
                               "Dataset-Indian-Names/Indian_Names.csv"]

        self.company_database = "Company_Names_Dataset/bse_companies.csv"

    def get_person_names_from_dataset(self) -> List[str]:
        """Reads the Name's from the CSV-files defined in self.name_databases.

        Returns:
            List[str]: A list of names read from the self.name_databases files
        """

        name_list = []

        for file in self.name_databases:
            pd_data_frame = pd.read_csv(file, encoding="ISO-8859-1")

            name_list.extend(list(pd_data_frame['name']))

        return name_list

    def get_company_names_from_dataset(self) -> List[str]:
        """Reads the Name's from the CSV-files defined in self.company_database.

        Returns:
            List[str]: A list of names read from the self.company_database file
        """

        pd_data_frame = pd.read_csv(self.company_database, encoding="ISO-8859-1")

        return list(pd_data_frame['Company Name'])

    def generate_entity_ruler_train_list(self, name_list: List[str], tag: str = "PERSON") -> List[dict]:
        """Sets up the "Train Data" of names of Persons/ Companies into the format required by Spacy's EntityRuler.
           Refer: https://spacy.io/usage/rule-based-matching

        Args:
            name_list (List[str]): List of Names belonging to a particular "tag"
            tag (int): The "type" of Names (PERSON/ ORG) provided in name_list

        Returns:
            List[dict]: Training Data List, which can be inputted to Spacy's EntityRuler.
        """

        train_list = []

        for name in name_list:
            train_dict = {'label': tag, "pattern": name}
            train_list.append(train_dict)

        return train_list

    def generate_entity_train_data(self) -> List[dict]:
        """Generates up the "Train Data" of Persons/ Companies names to be used by Spacy's EntityRuler.
           Refer: https://spacy.io/usage/rule-based-matching

        Returns:
            List[dict]: Training Data List (Of Person + Company Names), which can be inputted to Spacy's EntityRuler.
        """

        person_name_list = self.get_person_names_from_dataset()
        person_train_data = self.generate_entity_ruler_train_list(person_name_list, "PERSON")

        company_name_list = self.get_company_names_from_dataset()
        company_train_data = self.generate_entity_ruler_train_list(company_name_list, "ORG")

        train_list = person_train_data
        train_list.extend(company_train_data)

        return train_list

    def train_entity_ruler(self) -> None:
        """Initializes Spacy's EntityRuler module, and sets self.entity_model

        In addition, updates Spacy's EntityRuler with the Train Data available
        """

        nlp = spacy.load("en_core_web_sm")

        ruler = EntityRuler(nlp)
        patterns = self.generate_entity_train_data()
        ruler.add_patterns(patterns)
        nlp.add_pipe(ruler)

        self.entity_model = nlp

    def pdfreader_generate_text(self, PDF_file: str) -> List[str]:
        """Extracts Text from the <PDF_file> Document Specified.

        Returns a List of Text(str), where each element's contents
        correspond to the PDFs page contents.

        Args:
            PDF_file (str): The full path to the PDF Document

        Returns:
            List[str]: Page wise Contents of the PDF of interest.
        """
        page_contents = []

        pdfFileObj = open(PDF_file, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

        numPages = pdfReader.numPages

        for page in range(0, numPages):
            pageObj = pdfReader.getPage(page)

            page_contents.append(pageObj.extractText())

        pdfFileObj.close()

        return page_contents

    def extract_name_around_email(self, pdf_path: str) -> List[str]:
        """Extracts Names of Person's around an Email-Id in a Document

        As Spacy's Named Entity Recognition (NER) does not work well for Person's Name (Indian Person),
        this method is an alternate approach to finding a Person's name from a document.

        This method relies on the Person's Name to be around the Email-Id of the Person.

        Algorithm:
            1) Do a Reg-Exp Search, to Isolate where E-Mails Occur in the Document (Get the Email-Span)
            2) Search for a Person's name from the Text near the found E-mail (100 Characters before the Email)
            3) Using Spacy's Parts of Speech Tagging (POS Tagging), Classify 2 or more
               consecutive words with POS == PROPN (AKA Proper Noun) as a Name

        Args:
            PDF_file (str): The full path to the PDF Document

        Returns:
            List[str]: Possible List of Persons Names
        """
        text_contents = self.pdfreader_generate_text(pdf_path)

        # Initialize names_through_email, which will store the names of every person
        names_through_email = []

        # Cycle Through all the Pages
        for text in text_contents:

            # Try to Isolate the Text around Emails
            # r'[\w\.-]+@[\w\.-]+' is the Reg-Exp used to match Email Patterns
            for match in re.finditer(r'[\w\.-]+@[\w\.-]+', text):

                # Find the Span of the Emails using match.span().

                # Through the Email Span, find the Text "Around" the
                # Email which may contain the corresponding Author's Name

                # For our Usecase, we observed the Author's Name is most
                # likely to occur, within 100 characters before the "Email String"
                search_start_ind = max(match.span()[0] - 100, 0)
                search_end_ind = match.span()[0]

                possible_name = text[search_start_ind: search_end_ind]

                filtered_sent = []

                name_doc = self.entity_model(possible_name)
                for word in name_doc:

                    # Filter Out Stop Words
                    if word.is_stop == False:
                        filtered_sent.append(word)

                # LOGIC to Extract Person Name from "possible_name/ filtered_sent"
                # Use Spacy's Parts Of Speech Tagger (POS Tagger)
                # Check for 2 Consecutive words of type: PROPN (Proper Nound)
                # Any set of words that Satisfy This Criteria will be considered a Author/ Person Name

                # We don't use Spacy's Entity Tagger, as this Method is an alternative to
                # Spacy's Entity Tagger based Search (Done through

                # Temporary Variables
                temp_name_list = []
                ind = 0

                while ind < len(filtered_sent) - 1:

                    if filtered_sent[ind].pos_ != 'PROPN':

                        # Not a PROPN, Save any valid "Name" if we found any so far!

                        # We assume atleast 2 consecutive PROPN Words will be considered a valid "Name"
                        if len(temp_name_list) > 1:

                            name_to_append = " ".join(temp_name_list)

                            if name_to_append.lower() not in names_through_email:
                                names_through_email.append(name_to_append.lower())

                        temp_name_list = []

                    else:

                        # Else it is a 'PROPN'. Add the Word to temp_name_list

                        temp_name_list.append(filtered_sent[ind].text)

                    ind += 1

                # Save if any Valid Name is still to be added to the list
                if len(temp_name_list) > 1:

                    name_to_append = " ".join(temp_name_list)

                    if name_to_append.lower() not in names_through_email:
                        names_through_email.append(name_to_append.lower())

        return names_through_email

    def extract_name_and_org_from_pdf(self, pdf_path: str) -> Union[bool,
                                                                    Tuple[List[str],
                                                                          List[str],
                                                                          List[str]]]:
        """Extracts possible Author Name(s), Company Author Name(s), All Companies mentioned in the document.

        Uses Spacy's Named Entity Recognition (NER) to find a Persons Name (doc.ents_ == PERSON) and a
        Company name (doc.ents_ == ORG)

        Algorithm:
            1) For Author Name(s), Company Author Name(s), search the first 3 pages of the document.
               Any Entity matching: PERSON will be considered a Document Author
               Any Entity matching: ORG will be considered a Document Company Author (Company that authored the doc.)

               The reason we look at the First 3 pages, as there is no way to distinguish between Person Author names/
               Company Author Name; V/S/ other general Persons/ Company Names mentioned in the document

            2) For Extracting All companies mentioned in the doc, search all the pages of the PDF.
               Any Entity matching: ORG will be considered a valid Company Name.

        Args:
            PDF_file (str): The full path to the PDF Document

        Returns:
            possible_author_name (List[str]): Possible Person Author Names of the Document
            possible_author_comp (List[str]): Possible Company Author Names of the Document
            all_companies (List[str]): All companies mentioned in the document

            False: On a failure
        """
        if self.entity_model == None:
            print("Initialize the Model first")
            return False

        text_contents = self.pdfreader_generate_text(pdf_path)

        # Initializing Different List's.

        # all_companies --> Names of Companies (Spacy Tag: ORG) extracted from all pages
        # possible_author_name --> Name's of Authors (Spacy Tag: PERSON) extracted from 1st to 3rd page
        # possible_author_comp --> Name's of Companies (Spacy Tag: ORG) extracted from the 1st to 3rd page

        # We work with an assumption that Author Name/ Company Author Name will come up within
        # the first 3 pages, and hope one of the names extracted would be the actual company name, author name

        # To improve the Search Results, we could take the following steps:
        # 1) If we know which section of the page Author/Company Name comes up, We could only parse portion of the text
        #    This can be done through OpenCV/PyTesseract Libraries (Parse only a Region of Interest)

        all_companies = []
        possible_author_name = []
        possible_author_comp = []

        # For Author Person Name, and Author Company Name,
        # search the pages <first_page_ind> to <last_page_ind - 1> only!!
        last_page_ind = 3
        first_page_ind = 0

        # Current Page Index
        index = 0

        # Iterate through all the Pages
        for text in text_contents:

            # Using Spacy's NLP Model, get the Document Entities, which can be accessd through "doc"
            # doc.ents_ == PERSON --> Implies it is a Person
            # doc.ents_ == ORG --> Implies it is an Organisation
            doc = self.entity_model(text)

            for ent in doc.ents:

                # Perform some additional String Processing to ensure we have clean Text

                # A Company/ Organisation can be composed of AlphaNumeric Characters
                organisation_name_text = re.sub(r'[^A-Za-z0-9 ]+', '', ent.text).strip().lower()

                # A person will have only Alpha Characters (A-Z). Should generally not have any Numeric Characters
                person_name_text = re.sub(r'[^A-Za-z ]+', '', ent.text).strip().lower()

                # An Organization/ Person Name will generally have 2 or more words.
                # We also Put a Limit on the Max Number of Words for an Organisation Name. Person Name.

                # Unfortunately the Results from ent.label_ produces a lot of False-Positives.
                # This is one way of filtering out a lot of False Positives, with little probability
                # of loosing out on Real Companies/ Real Person Names

                if ent.label_ == "ORG" and \
                        1 < len(organisation_name_text.split()) < 7:

                    # all_companies will have entries from all pages
                    if organisation_name_text not in all_companies:
                        all_companies.append(organisation_name_text)

                    # The Company Author of the Document is assumed to be within:
                    # [<first_page_ind>, last_page_ind)
                    if first_page_ind <= index < last_page_ind and \
                            organisation_name_text not in possible_author_comp:
                        possible_author_comp.append(organisation_name_text)


                # Author's will have ent.label_ as PERSON

                # A person's name is expected to be within (1, 5) Words

                # The Author Name of the Document is assumed to be within:
                # [<first_page_ind>, last_page_ind)
                elif ent.label_ == "PERSON" and \
                        1 < len(person_name_text.split()) < 5 and \
                        first_page_ind <= index < last_page_ind and \
                        person_name_text not in possible_author_name:

                    possible_author_name.append(person_name_text)

            # Increment the Page Counter
            index += 1

        return possible_author_name, possible_author_comp, all_companies

    def get_target_price_and_recommendation(self, pdf_path) -> Union[List[dict],
                                                                     List[str],
                                                                     List[str]]:
        """Searches for a Target Price/ Recommendations in a Document (Financial Doc)

        Algorithm:
            1) Do a Search of "Target Price"/ "Price Target" in the document
            2) Get the Text in the vicinity of "Target Price"/ "Price Target"
            3) Do RegExp Search of the text in step (2) to get the Target Price/ Recommendations

        Args:
            PDF_file (str): The full path to the PDF Document

        Returns:
            price_recommendations_list (List[dict]) : A mapping between the Target Price and
                                                      Corresponding Recommendations
            target_price_list (List[str]) : All Extracted Target Price from the Document
            recommendation_list (List[str]) : All Extracted Recommendations from the Documenr
        """
        # We search for the Target Price/ Recommendation "around" target price/ price target text
        target_price_tags = ['target price', 'price target']

        text_contents = self.pdfreader_generate_text(pdf_path)

        # A dictionary storing the Mapping between Price Recommendation, and Corresponding Recommendations
        price_recommendations_list = []

        # A list of Valid Target Prices Extracted
        target_price_list = []

        # A list of Valid Recommendations Extracted
        recommendation_list = []

        # Cycle through all the Pages
        for text in text_contents:

            # Remove \n literals to help with text processing
            filtered_text = text.lower().replace('\n', '')

            # Check all Possible Target Price Tags (target price/ price target)
            for target_str in target_price_tags:

                # Search the Document for possible search hits on "target price/ price target"
                for match in re.finditer(target_str, filtered_text, re.MULTILINE | re.DOTALL):

                    span = match.span()

                    # For Target Price, search the Text upto 15 Characters "Right" of "target price/ price target"
                    target_price_search_start_ind = span[0]
                    target_price_search_end_ind = span[1] + 15

                    # For Recommendations, search the Text upto 50 Characters "Right" and "Left of
                    # "target price/ price target"
                    reco_search_start_ind = max(span[0] - 50, 0)
                    reco_search_end_ind = span[1] + 50

                    target_price_search_str = filtered_text[target_price_search_start_ind:
                                                            target_price_search_end_ind + 1]
                    reco_search_str = filtered_text[reco_search_start_ind: reco_search_end_ind + 1]

                    # Get the Target Price and Recommendations

                    # target_price is a single "str"
                    target_price = self.extract_target_prices_from_text(target_price_search_str)

                    # recommendations is a List[str]
                    recommendations = self.extract_recommendations_from_text(reco_search_str)

                    # Store the Target-Price
                    if len(target_price) > 0 and target_price not in target_price_list:
                        target_price_list.append(target_price)

                    # Store the Recommendations
                    for recommendation in recommendations:

                        if recommendation not in recommendation_list:
                            recommendation_list.append(recommendation)

                    # If we Found a valid Target Price; add Target Price and Recommendation Mapping to the list
                    if len(target_price) > 0:

                        price_recom_dict = {target_price: recommendations}
                        if price_recom_dict not in price_recommendations_list:

                            price_recommendations_list.append(price_recom_dict)

        return price_recommendations_list, target_price_list, recommendation_list

    def extract_target_prices_from_text(self, text: str) -> str:
        """Searches for Target Price from the <text> of interest

            Does a Reg-Exp search to obtain the Target Price

        Args:
            PDF_file (str): The full path to the PDF Document

        Returns:
            price_filtered[0] (str): The Target Price
        """
        # If the text is something like: "target price rs.1,1000 (24 %)"
        # OP will be like ['target price rs.1,100', '(24']
        prices = re.findall(r' \(*[a-zA-Z \`\~]*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{3})*', text)

        if len(prices) == 0:
            return ""

        # Work with 'target price rs.1,100' only (i.e. Text closest to "target price")
        price = prices[0]

        price_text = price.strip()

        # Do one more level of Text Filtration, to obtain the Raw Number (1,100)
        price_filtered = re.findall(r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})*', price_text)

        return price_filtered[0]

    def extract_recommendations_from_text(self, text: str) -> List[str]:
        """Searches for Financial Recommendations from the <text> of interest

            Does a Reg-Exp search to obtain the Financial Recommendations, and returns any
            word that matches any of the words in <recommendation_tags> below

        Args:
            PDF_file (str): The full path to the PDF Document

        Returns:
            reco_list (List[str]): The Recommendations
        """
        # These are Potential Recommendation Tags
        recommendation_tags = ['buy', 'sell', 'neutral', 'add', 'reduce', 'hold', 'outperform', 'maintain',
                               'peer perform', 'mkt perform', 'recomm list', 'equal weight']

        reco_list = []

        # Get Any text that match with a String in recommendation_tags
        recommendations = [x for x in recommendation_tags if len(re.findall(x, text)) > 0]

        for recommendation in recommendations:

            if recommendation not in reco_list:
                reco_list.append(recommendation)

        return reco_list
