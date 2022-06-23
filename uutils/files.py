import os
import json

'todo add try except'

def write(data,file):
    with open(file,'w') as f:
        f.write(data)

def writelines(li,file)
    write(['\n'.join(li)],file)

def read(file):
    with open(file,'r') as f:
        return f.read()

def readlines(file):
    return [line for line in read(file).split('\n')

def append(data,file):
    with open(file,'a') as f:
      f.write('\n'+data)
