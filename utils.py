# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:15:45 2024

@author: lich5
"""
import json
 
def convert_ipynb_to_py(ipynb_file, py_file):
    with open(ipynb_file, 'r',encoding='utf-8') as f:
        notebook = json.load(f)
 
    with open(py_file, 'w',encoding='utf-8') as f:
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                f.write(''.join(cell['source']) + '\n\n')
                
convert_ipynb_to_py('chp1_numpy_eg.ipynb', 'gpu_movie_reviews.py')
