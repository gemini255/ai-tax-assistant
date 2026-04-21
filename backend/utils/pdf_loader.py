from langchain_community.document_loaders import PyPDFLoader


def load_pdfs():

    files = [
        "datasets/income_tax_act_2025.pdf",
        "datasets/faq1.pdf",
        "datasets/faq2.pdf",
        "datasets/faq3.pdf"
    ]

    docs = []

    for file in files:
        loader = PyPDFLoader(file)
        documents = loader.load()
        docs.extend(documents)

    return docs