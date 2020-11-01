# Document Processing for Financial Documenrts

Extracting Fields of Interest like Author Name/ Affiliation, Stock Target Price, Recommendations etc. from Financial Documents.

  - Extracts Author Name (Person), Company Author Name (Company that wrote the Document), All Companies mentioned in the document throgh SPACY's Named Entity Recognition (NER) module.
  - Extracts Author Name (Person), by doing a Reg-Exp search of Text around valid E-Mail IDs.
  - Extracts Stock Target Price/ Stock Recommendation (Buy, Sell, etc.) using Reg-Exp search.

### Files - DocumentExtractor . py
Implements the DocumentExtractor() class which provides functionality to process Financial Documents. Includes the Following Methods:
  - ***extract_name_and_org_from_pdf()***
      - **Overview:**
          Uses Spacy's Named Entity Recognition (NER) to find Person Author Name (doc.ents_ == PERSON) and a Company Author Name (doc.ents_ == ORG) and all Companies/ Organisations (doc.ents_ == ORG) mentioned in the document.

    - **Issues/ Assumption Made:**
        - It cannot distinguish between the Document Author (Person/ Company) name against  
          general companies/ person names in the document. The Workaround this Method does is to assume any Person/ Company mentioned within the first 3 pages of the Document could be valid Authors.
        - Spacy's default English Model for Indian names don't work for Indian Person names. To overcome this issue, implemented another method **extract_name_around_email()** which is an alternate (more reliable) way to search for Person Author Names within these financial documents.
        
    - **Experiments Tried, Possible Solutions/ Future Work:**
        - For Document Author Name (Person/ Company), extract the text from the Header/ Footer. This will help distinguishing between Document Authors (Person/ Company)  and general Person/ Company names. (OpenCV, PyTesseract could be handy for such tasks)
        - Tried Retraining Scapy's NER module with Indian Names/ Indian Company names. But this led to the "Catastropic Forgetting Problem". Hence, used Spacy's EntityRuler to "Train" Spacy's NER model with Indian Person/ Company names through a Rule-Based Approach. 
        This worked well for Indian Companies, because the Company Names Train Data had significant overap with the Test Data. But not for Person Names (where there was no significant overlap between Train/ Test data Person names)
        
            **TO DO:** Retrain Scapy's NER module with Default Model Train Data (On which Spacy's default English model was trained on) + New Indian Names Train data to avoid the “Catastropic Forgetting Problem”. Reference: https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
            
  - ***extract_name_around_email()***
        - **Overview**
            - Extracts Names of Person's around an Email-Id in a Document through Reg-exp based search and Scapy's Part's of Speech Tagging (POS) module.
            -   As Spacy's default Named Entity Recognition (NER) does not work well for Person's Name (Indian Person), this method is an alternate approach to finding a Person's name from a document.
            -   
        - **Issues**
            - If the doocument does not have text in the format <Author Name *...... txt ...* Author mail>, results will be inaccurate
    
  - ***get_target_price_and_recommendation()***
        - **Overview:**
            Searches for a Target Price/ Recommendations in a Document (Financial Doc) through Reg-Ex based search
        - **Issues:**
            Depends on the Target Price/ Recommendation Info. being in the vicinity of text *"Target Price"* and *Price Target*
    
### Files - main . py
Processes PDFs in the folder */needle_pdf_docs* and saves the Processed Info at */Results*.
    
- File: ***Name_Org_Results.csv*** : Contains Fields -->
            
        ['File Name', 'Author Name - Through Email', 'Author Name - Through Spacy Model', 'Author Institution Through Spacy Model', 'All Companies Through Spacy Model']
- File: ***Target_Price_Reco_Results.csv*** : Contains Fields --> 

        ['File Name', 'Target Price', 'Recommendation', 'Price - Recommendation Mapping']


### Requirements

```sh
Python --- 3.8.1
pandas --- 1.0.4
PyPDF2 ---  1.26.0
spacy --- 2.3.2
```

### Example Use

```sh
from DocumentExtractor import DocumentExtractor

obj = DocumentExtractor()

obj.train_entity_ruler()

pdf_file = "C:/Users/62871/Desktop/Deb_Learn/needle_ai/pdf_docs/[Kotak]_576.pdf"

author_name, author_company, all_company = obj.extract_name_and_org_from_pdf(pdf_file)

In[9]: author_name
Out[9]: 
['smp    predatory',
 'rohit chordia',
 'aniket sethi',
 'al views',
 'not available']
 
In[10]: author_company
Out[10]: 
['kotak institutional equities research',
 'kotak securities',
 'kotak securities research']

In[11]: all_company
Out[11]: 
['kotak institutional equities research',
 'kotak securities',
 'kotak securities research',
 'uk ltd',
 'portsoken house',
 'kotak mahindra inc',
 'the monetary authority', .......]
 
 
In[13]:  email_names = obj.extract_name_around_email(pdf_file)
Out[13]: 
['kotak institutional equities research',
 'rohit chordia',
 'aniket sethi',
 'kotak mahindra inc',
 'officer details',
 'mr. manoj agarwal',
 'compliance officer',
 'mr. kamlesh rao']
 
 
 price_reco_mapping, all_prices, all_reco = obj.get_target_price_and_recommendation(pdf_file)
 
In[21]: all_prices
Out[21]: ['105', '1,050', '201']

In[22]: all_reco 
Out[22]: ['buy', 'maintain']

In[23]: price_reco_mapping
Out[23]: [{'105': ['buy', 'maintain']}, {'1,050': ['buy']}, {'201': ['buy']}]
