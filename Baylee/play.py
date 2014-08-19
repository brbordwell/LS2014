# !/bin/User/bin/python

import csv
import numpy as np
import astropy.io.fits as fits
from sklearn import svm
from glob import glob
import pylab as pl


class Table:
    """A class for the data object. """
    def __init__(self,fyle,parms):
        self.name = fyle
        self.params = parms
        self.data = getdata(fyle,params)
        self.attr = []
        self.clss = []
        
    def __str__(self):
        return self.name


    def getdata(self, tags=False):
        """This function will create arrays from the csvfile based
        on specific schema"""
    
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
        if tags:
            if type(tags[0]) == str: 
                inds = [schema.index(tag) for tag in tags]  
                self.tags = self.attr
            else: 
                inds = tags
                self.tags = [names[i] for i in self.attr]
        else: inds = range(len(names))
        np.genfromtxt(csvfile, delimiter=',',names=names, dtype='float64',
                      skip_header=1, usecols=inds)
    
    def sel(self,tags):
        """Grab part of the data"""
        part = np.array([self.data[tag] for tag in tags])
        siz = part.shape  
        if len(siz) > 1: if siz[0] > siz[1]: part = part.transpose()


def pregunta(csvfile, tag=False):
    """This function will create lists from the csvfile so that
    the schema are slightly more searchable. Until intelligently 
    edited, this function will require providing the name of the
    file containing the schema."""
    
            #Setting up the reader
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








#Hokay, so we have the table class to get the data and work with the attributes, now we need to play with the machine learning aspects. I would like to add exploratory aspects to the table class to allow visual exploration of the parameter space as well.

def learn(fil,params):
    """Holds all the ML stuff"""

    cl_names = ["spiral","elliptical","uncertain"]
    attr = [p for p in params if p not in cl_names]
    #weights = would be interesting to repopulate the paramaterspace on the basis of the unbiased probabilities from galaxy zoo

    
    data = Table(fil,params)
    
    data.attr = attr
    data.clss = cl_names
    names = data.tags



    clf = svm.SVC()
    clf.fit(data.sel(data.attr),data.sel(data.clss))
