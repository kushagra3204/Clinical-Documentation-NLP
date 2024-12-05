import os
import PyPDF2

pdf_path = "./books/pdf"
txt_path = "./books/txt"

for book in os.listdir(pdf_path):
    with open(os.path.join(pdf_path,book), "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        with open(os.path.join(txt_path,f"{book[:-4]}.txt"), "w", encoding="utf-8") as txt_file:
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    txt_file.write(text + "\n\n")
    print(f"Text extracted from PDF {file.name}")