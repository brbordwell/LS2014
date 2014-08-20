# !/bin/User/bin/python

import csv
import numpy as np
import astropy.io.fits as fits
from sklearn import svm
from sklearn import cross_validation as cvd
from sklearn.grid_search import GridSearchCV as GSCV
from glob import glob
import pylab as pl


#CONVERT CLASSES TO NUMBERS YO, VERY IMPORTANT

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


def cls(clss,inv=False):
    """Making the classes more like classes"""
    
    if inv:
        iclss = np.zeros((max(clss)+1,len(clss)))
        for i in xrange(iclss.shape[0]):
            ind = np.where(clss == i)
            iclss[i][ind] = 1
        return iclss
    else:
        clss = np.array(clss)
        clss[np.where(clss EQ 0)] = -100
        clss[1] += 1
        clss[2] += 2
        return clss[np.where(clss GT 0)]
    




#Hokay, so we have the table class to get the data and work with the attributes, now we need to play with the machine learning aspects. I would like to add exploratory aspects to the table class to allow visual exploration of the parameter space as well.

def learn(fyle,params,clf=svm.LinearSVC(),cvd_size=1/3.,simple=True, simplest=False):
    """FUNCTION: LEARN(FYLE,PARAMS,CLF,CVD_SIZE,SIMPLE,SIMPLEST)
    PURPOSE: Holds all the ML stuff
    INPUT:
      fyle = 
      params = 
      clf = 
      cvd_size = 
      simple = 
      simplest = 
      
    OUTPUT:
      """

    cl_names = ["spiral","elliptical","uncertain"]
    attr = [p for p in params if p not in cl_names]
    #weights = would be interesting to repopulate the paramaterspace on the basis of the unbiased probabilities from galaxy zoo
    

    #Woo data
    data = Table(train,params)
    data.attr = attr  ;  data.clss = cl_names
    names = data.tags


    #Without intense shuffling (faster)
    if simplest:
        data.split(cvd_size,cv=False)
        clf.fit(data.train[0], cls(data.train[1]))
        pred = clf.pred(data.test[0], cls(data.test[1]))
        acc = (pred == data.test[1]).sum()/len(data.test[1])
        return clf.pred()

    #With intense shuffling (slower)
    else:
        #Working with fixed values of C and kernel params...
        data.split(cvd_size)
        if simple:
            scores = cvd.cross_val_score(clf, 
                                         data.sel(data.attr), 
                                         cls(data.sel(data.clss)), 
                                         cv = data.mixed)
            acc = scores.mean()
            return clf.pred()
            
        #Ranging through parameter space of C and the kernel params...
        else:
            C_range = None
            param_grid = dict(C=C_range)
            grid = GSCV(clf, 
                        param_grid = param_grid, 
                        cv = data.mixed)
            grid.fit(data.sel(data.attr),
                     cls(data.sel(data.clss)))
            
            if False: #Too lazy to comment
                scores = cvd.cross_val_score(grid.best_estimator_, 
                                             data.sel(data.attr), 
                                             cls(data.sel(data.clss)), 
                                             cv = data.mixed)
                acc = scores.mean()
            
            return grid.best_estimator_

   

def apply(fitter, datafile, params):
    """Work that unclassified data yo"""
    #Setup
    cl_names = ["spiral","elliptical","uncertain"]
    attr = [p for p in params if p not in cl_names]

    #Fit
    data = Table(datafile,params)    
    data.attr = attr  ;  data.clss = cl_names
    res = cls(fitter.predict(data.data)),inv=True)

    #Save
    names = params  ;  for cl in cl_names: names.append(cl)
    s = data.data.shape  ;  if s[0] > s[1]: data.data.transpose()
    data = np.append(data,res,axis=1)
    save_name = data_file[0:-5]+'_classd.csv'
    np.savetxt(save_name,,delimiter=',')



    
def wrapper():
    """This function will make the necessary calls to learn and apply"""
    pass
