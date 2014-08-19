# !/bin/User/bin/python

import csv
import numpy as np
import astropy.io.fits as fits
from sklearn import svm
from sklearn import cross_validation as cvd
from sklearn.grid_search import GridSearchCV as GSCV
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
        self.train = [[],[]]
        self.test = [[],[]]
        self.mixed = []

    def __str__(self):
        return self.name


    def getdata(self, tags=self.params):
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

        return part


    def split(self,size,cv=True):
        """Split the data into training and testing sets"""

        if cv:
            data.mixed = cvd.StratifiedShuffleSplit(self.clss,5,test_size = size)
        else:
            self.train[0],self.test[0],self.train[1],self.test[1] = 
            cvd.train_test_split(
                self.sel(self.attr),
                self.sel(self.clss),
                test_size=size)

            self.data = [] #avoiding weighing python down, can replace using getdata




class Result:
    """A class for results? """
    def __init__(self):
        self.cvd_size = [] #can vary size
        self.train_file = [] #can vary training set
        self.kernel = [] #can vary kernel
        self.acc = [] #varies with input variation
        self.best_estimator = [] #from the 
        self.grid_variables = [] #this list should receive dictionaries which describe which variables are being varied as the keys,and specifies the ranges as the values


    def save(self,name):
        """ Allows you to save all of the results. """

        #should add something to automatically generate intelligent name
        np.savez(name,self)




def pregunta(csvfile, tag=False):
    """This function will create lists from the csvfile so that the schema 
    are slightly more searchable"""
    
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









#Hokay, so we have the table class to get the data and work with the attributes, now we need to play with the machine learning aspects. I would like to add exploratory aspects to the table class to allow visual exploration of the parameter space as well.

def learn(fyle,params,cvd_size=1/3.,simple=True, simplest=False):
    """Holds all the ML stuff"""

    cl_names = ["spiral","elliptical","uncertain"]
    attr = [p for p in params if p not in cl_names]
    #weights = would be interesting to repopulate the paramaterspace on the basis of the unbiased probabilities from galaxy zoo

    
    data = Table(train,params)
    data.attr = attr  ;  data.clss = cl_names
    names = data.tags


    if simplest:
        pass
    else:
        if simple:
            clf = svm.SVC()
            data.split(cvd_size)
            scores = cvd.cross_val_score(clf, 
                                         data.data[0], 
                                         data.data[1], 
                                         cv = data.mixed)
        

    #Note: using GSCV requires at least 2 values in the grid
        else:
            C_range = None
            param_grid = dict(C=C_range)
            grid = GSCV(svm.SVC(), 
                        param_grid = param_grid, 
                        cv = data.mixed)
            grid.fit(data.train[0],data.train[1])


    

    #print clf

    clss_pred = clf.predict(data.test[0])
    N_match = (clss_pred == data.test[1]).sum()
    




def wrapper():
    """This function will make the necessary calls to learn and apply"""
