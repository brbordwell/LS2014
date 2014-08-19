# !/bin/User/bin/python

import csv
import numpy as np
import astropy.io.fits as fits
from sklearn import svm
from glob import glob



class Table:
    """A class for the data object. """
    def __init__(self,fyle,attr):
        self.name = fyle
        self.attr = attr
        self.data = getdata(fyle,attr)

        
    def __str__(self):
        return self.name


    def getdata(self, tags=False):
        """This function will create arrays from the csvfile based on specific schema, and will save the results as a fits file as desired"""
    
        #Expanding the filename
        csvfile = self.name
        check = glob(csvfile)
        if len(check) != 1:
            print "ERROR: Name of file given is not unique"
            return
        else:
            if len(check) == 0:
                print "ERROR: No matching file"
                return
            else:
                csvfile = check[0]

        #Getting the desired selection of the data
        fyle = open(csvfile,'rb')
        names = csv.reader(fyle).next()
        if tags: inds = schema.index(tag)  
        else: inds = range(len(names))
        np.genfromtxt(csvfile, delimiter=',',names=names, dtype='float64',
                      skip_header=1, usecols=inds)
    




def pregunta(csvfile, tag=False):
    """This function will create lists from the csvfile so that the schema are slightly more searchable"""
    
    #Setting up the reader
    fyle = open(csvfile,'rb')
    rdr = csv.reader(fyle)


    #Reading in the table
    table = []  ;  run = 1
    while run:
        try:
            table.append(rdr.next())
        except:
            run = 0


    #Obtaining the schema tags
    tags = [x[0] for x in table]  ;  tags = tags[1:]
    defs = [x[-1] for x in table]  ;  defs = defs[1:]

    

    #Matching the tags
    if tag:
        ind = tags.index(tag)  ;  match = defs[ind]
        print match
        return match
    else:
        return [[tags],[defs]]
