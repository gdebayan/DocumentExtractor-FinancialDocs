import os
import csv

from DocumentExtractor import DocumentExtractor


if __name__ == "__main__":

    obj = DocumentExtractor()
    obj.train_entity_ruler()

    BASE_PATH = "needle_pdf_docs"
    fields = ['File Name', 'Author Name - Through Email',
              'Author Name - Through Spacy Model',
              'Author Institution Through Spacy Model',
              'All Companies Through Spacy Model']

    f = open("Results//Name_Org_Results.csv", "w")
    write = csv.writer(f)
    write.writerow(fields)

    fields2 = ['File Name', 'Target Price', 'Recommendation', 'Price - Recommendation Mapping']
    f2 = open("Results//Target_Price_Reco_Results.csv", "w")
    write2 = csv.writer(f2)
    write2.writerow(fields2)

    count = 0
    for filename in os.listdir(BASE_PATH):
        if filename.endswith(".pdf"):

            pdf_file = BASE_PATH + "//" + filename
            pdf_file = os.path.abspath(pdf_file)

            try:

                # Get Author/ ORG name through SPACY Model
                author_name, author_company, all_company = obj.extract_name_and_org_from_pdf(pdf_file)

                # Get Author Names through the text around EMAIL IDs, as SPACY's NER for Indian Person names dont
                #  work great
                email_author_names = obj.extract_name_around_email(pdf_file)

                fields = [filename, email_author_names, author_name, author_company, all_company]

                write.writerow(fields)

                print("Count", count)
                print("File Name", filename)
                print("email_author_names", email_author_names)
                print("author_name", author_name)
                print("author_company", author_company)
                print("all_company", all_company)

                price_reco_mapping, target_price, recommendations = obj.get_target_price_and_recommendation(pdf_file)

                fields_recommend = [filename, target_price, recommendations, price_reco_mapping]

                write2.writerow(fields_recommend)
                print("target_price", target_price)
                print("recommendations", recommendations)
                print("price_reco_mapping", price_reco_mapping)

                print("\n\n")

            except Exception as e:
                print("Exception processing File", pdf_file, "\n Excepton: ", e)

            count += 1

    f.close()
    f2.close()



# Eg. Call
# obj = DocumentExtractor()
# obj.train_entity_ruler()
# pdf_file = "C:/Users/62871/Desktop/Deb_Learn/needle_ai/pdf_docs/[Kotak]_576.pdf"
#
# author_name, author_company, all_company = obj.extract_name_and_org_from_pdf(pdf_file)
#
# email_names = obj.extract_name_around_email(pdf_file)
#
# price_reco_mapping, all_prices, all_reco = obj.get_target_price_and_recommendation(pdf_file)
