import argparse
import os
import pathlib
import pprint
import re
import pdfplumber
import json
import pandas as pd
import nltk
    
class TextExtract:
    
    def __init__(self):
        self.max_pages=0
        self.total_pages=0
        self.file_count=0
    def find_files(self, folder_path: str, ext: str):
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(ext):
                    filepath = os.path.abspath(os.path.join(root, file))
                   
                    with open(filepath, 'rb') as f:
                        header = f.read(4)
                        if header[:4] == b'%PDF':
                            with pdfplumber.open(filepath) as pdf:      
                                self.total_pages += len(pdf.pages)
                                self.file_count += 1
                                if self.max_pages < len(pdf.pages):
                                    self.max_pages = len(pdf.pages)
        
if __name__ == "__main__":    
    te = TextExtract()
    extensions = (".pdf")
    te.find_files(folder_path=r"F:\LLM-TESTDATA", ext=extensions)
    print(f"Total pages {te.total_pages}")
    print(f"Max pages {te.max_pages}")
    print(f"File Count {te.file_count}")