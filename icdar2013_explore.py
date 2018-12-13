
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import sys
import os
import re
import pandas as pd
from bs4 import BeautifulSoup
import argparse

### Load ground truth data

def get_xml_tables(path,filename):
    
    with open(os.path.join(path,filename)) as filepath:
        
        # load xml data
        name = filename.split('.')[-2]
        xml = BeautifulSoup(filepath,'lxml')
        
        # remove unneccessary tags
        for tag in xml.find_all('instruction'):
            tag.decompose()
        for tag in xml.find_all('bounding-box'):
            tag.decompose()
            
        #return all tables as a list
        return xml.find_all('table')

def get_all_documents_xml(datapath):
    pattern = re.compile(".*-str\.xml")
    documents = {}
    filenames = [f for f in os.listdir(datapath) if pattern.match(f)]
    for filename in filenames:
        name = filename.split('.')[-2]
        documents[name] = get_xml_tables(datapath,filename)
    return documents

### Transform data

def get_cell_position(cell):
    start_col = int(cell.attrs['start-col']) if cell.has_attr('start-col') else 0
    start_row = int(cell.attrs['start-row']) if cell.has_attr('start-row') else 0
    end_col = int(cell.attrs['end-col']) if cell.has_attr('end-col') else start_col
    end_row = int(cell.attrs['end-row']) if cell.has_attr('end-row') else start_row
    return start_col,start_row,end_col,end_row

def get_row_and_columns_count(table):
    cells = table.find_all('cell')
    max_cols = 0
    max_rows = 0
    for cell in cells:
        start_col,start_row,end_col,end_row = get_cell_position(cell)
        max_cols = max(max_cols, start_col, end_col)
        max_rows = max(max_rows, start_row, end_row)
    return max_rows,max_cols

def lookup_cell(table,row,column,max_rows,max_cols, type='simple'):
    if type == 'simple':
        return table.find('cell', attrs={'start-col':str(column),'start-row':str(row)})
    elif type == 'spanning':
        cells = table.find_all('cell')
        for cell in cells:
            start_col,start_row,end_col,end_row = get_cell_position(cell)
            if column >= start_col and column <= end_col and row >= start_row and row <= end_row:
                return cell
    else:
        return None

def xml_to_list(table):
    max_rows, max_cols = get_row_and_columns_count(table)
    output = []
    for r in range(max_rows+1):
        row = []
        for c in range(max_cols+1):
            cell = lookup_cell(table,r,c,max_rows,max_cols,type='simple')
            entry = cell.content.string if cell is not None else ''
            row.append(entry)
        output.append(row)
    return output

def batch_xml_to_list(tables):
    output = []
    for tbl in tables:
        output.append(xml_to_list(tbl))
    return output


def transform_all(documents_xml):
    documents = {}
    for key in documents_xml.keys():
        documents[key] = batch_xml_to_list(documents_xml[key])
    return documents

### Output transformed data

def output_as_csv(documents, output_dir):
    for k in documents.keys():        
        n_tables = len(documents[k])
        for n in range(n_tables):
            df = pd.DataFrame(documents[k][n])
            folder = os.path.join(output_dir,k)
            filename = 'table' + str(n) + '.csv'
            filepath = os.path.join(folder,filename)
            if not os.path.exists(folder):
                os.mkdir(folder)
            df.to_csv(filepath,index=False,header=False)


### Main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir')
    parser.add_argument('target_dir')
    args = parser.parse_args()
    datapath = '/home/imran/work/datasets/icdar2013-competition-dataset-with-gt/competition-dataset-eu/'
    documents = transform_all(get_all_documents_xml(args.source_dir))
    output_as_csv(documents,args.target_dir)    



