import sys
import csv
import numpy as np
from numpy.random import shuffle
import astropy.io.fits as fits
from sklearn import svm
from sklearn import cross_validation as cvd
from sklearn.grid_search import GridSearchCV as GSCV
from glob import glob
import pylab as pl
import plotter
import threading
from threading import Thread
from itertools import ifilter
import pickle




class Table:
    """A class for the data object. """

    def __init__(self,fyle,parms,N,cut):
        self.name = fyle
        self.params = parms
        self.data = getdata(fyle,params)
        self.N = N
        self.cut = cut
        self.attr = []
        self.clss = []
        self.train = [[],[]]
        self.test = [[],[]]
        self.mixed = []
        

    def __str__(self):
        return self.name


    def getdata(self, tags=None, N=None, cut=None):
        """This function will create arrays from the csvfile based
        on specific schema"""
    
        if tags == None: tags=self.params
        if N == None: tags=self.N
        if cut == None: tags=self.cut

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

        with open(csvfile) as table:
            def crit(x,cut):
                x = x[0].split(',')
                return x[1] > cut and x[2]

            filt = ifilter(crit(x,cut), table)
            dat = np.genfromtxt(filt, delimiter=',', names=names, 
                                 dtype='float64', skip_header=1, usecols=inds)
            l = np.max(dat[0].shape)
            if l > Num:
                inds = range(Num) 
                for a in xrange(5): shuffle(inds)
                return dat[inds]
            else:
                return dat

    def sel(self,tags):
        """Grab part of the data"""
        part = np.array([self.data[tag] for tag in tags])

        siz = part.shape  
        if len(siz) > 1: 
            if siz[0] > siz[1]: part = part.transpose()

        return part



    def split(self,size,cv=True):
        """Split the data into training and testing sets"""

        if cv:
            data.mixed = cvd.StratifiedShuffleSplit(self.clss,5,test_size = size)
        else:
            self.train[0],self.test[0],self.train[1],self.test[1] = cvd.train_test_split(
                self.sel(self.attr),
                self.sel(self.clss),
                test_size=size)

            self.data = [] #avoiding weighing python down,getdata will recreate




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
    
    #Expanding the filename
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




def cls(clss,p_cutoff=.8,inv=False):
    """Making the classes more like classes"""
    clss = clss > p_cutoff

    if inv:
        iclss = np.zeros((max(clss)+1,len(clss)))
        for i in xrange(iclss.shape[0]):
            ind = np.where(clss == i)
            iclss[i][ind] = 1
        return iclss

    else:
        clss = np.array(clss)
        clss[np.where(clss == 0)] = -100
        for i in xrange(clss.shape[0])+1: clss[i] += i

        return clss[np.where(clss > 0)]




def cls_old(clss,inv=False):
    """Making the classes more like classes"""
    
    if inv:
        iclss = np.zeros((max(clss)+1,len(clss)))
        for i in xrange(iclss.shape[0]):
            ind = np.where(clss == i)
            iclss[i][ind] = 1
        return iclss
    else:
        clss = np.array(clss)
        clss[np.where(clss == 0)] = -100
        clss[1] += 1
        clss[2] += 2
        return clss[np.where(clss > 0)]
    




#Hokay, so we have the table class to get the data and work with the attributes, now we need to play with the machine learning aspects. I would like to add exploratory aspects to the table class to allow visual exploration of the parameter space as well.

def learn(fyle,params,clf=svm.LinearSVC(),cvd_size=1/3.,cut=.8,Num=20,
          simple=True,simplest=False,param_grid=None):
    """FUNCTION: LEARN(FYLE,PARAMS,CLF,CVD_SIZE,SIMPLE,SIMPLEST)
    PURPOSE: Holds all the ML stuff
    INPUT:
      fyle = The file with training data.
      params = The parameters to be used.
      clf = The ML fitter to be used.
      cvd_size = The split size of the amount of data to be used as the
                 test set.
      cut = The value for the probability of being eliptical or spiral that 
            I use to distinguish them from the unknowns. 
      simple = Cross validation, one test, no variation (Default)
      simplest = No cross validation, one test, no variation. 
      param_grid = If simple and simplest are both off then function will 
                   perform a grid search through fitting parameters.
      
    OUTPUT:
      The trained fitter function object.
      """

    #cl_names = ["spiral","elliptical","uncertain"]
    cl_names = ["p_el_debiased","p_cs_debiased"]
    attr = [p for p in params if p not in cl_names]
    #weights = would be interesting to repopulate the paramaterspace on the basis of the unbiased probabilities from galaxy zoo
    

    #Woo data
    data = Table(train,params,Num,cut)
    data.attr = attr  ;  data.clss = cl_names
    names = data.tags


    #Without intense shuffling (faster)
    if simplest:
        data.split(cvd_size,cv=False)
        clf.fit(data.train[0], cls(data.train[1],cut))
        pred = clf.pred(data.test[0], cls(data.test[1],cut))
        acc = (pred == data.test[1]).sum()/len(data.test[1])
        
        np.savez()
        return clf.pred(), acc, 0

    #With intense shuffling (slower)
    else:
        #Working with fixed values of C and kernel params...
        data.split(cvd_size)
        if simple:
            scores = cvd.cross_val_score(clf, 
                                         data.sel(data.attr), 
                                         cls(data.sel(data.clss),cut), 
                                         cv = data.mixed)
            acc = scores.mean()  ;  std = scores.std()
            return clf.pred(), acc, std
            
        #Ranging through parameter space of C and the kernel params...
        else:
            grid = GSCV(clf, 
                        param_grid = param_grid, 
                        cv = data.mixed)
            grid.fit(data.sel(data.attr),
                     cls(data.sel(data.clss),cut))
            
            scores = cvd.cross_val_score(grid.best_estimator_, 
                                         data.sel(data.attr), 
                                         cls(data.sel(data.clss),cut), 
                                         cv = data.mixed)
            acc = scores.mean()  ;  std = scores.std()
            
            return grid.best_estimator_, acc, std

   



def apply(fitter, datafile, params, cut, Num):
    """Work that unclassified data yo"""
    #Setup
    cl_names = ["spiral","elliptical","uncertain"]
    attr = [p for p in params if p not in cl_names]

    #Fit
    data = Table(datafile,params,Num,cut)    
    data.attr = attr  ;  data.clss = cl_names
    res = cls(fitter.predict(data.data),cut,inv=True)

    #Save
    names = params    
    for cl in cl_names: names.append(cl)
    s = data.data.shape  
    if s[0] > s[1]: data.data.transpose()
    data = np.append(data,res,axis=1)
    save_name = data_file[0:-5]+'_classd.csv'
    np.savetxt(save_name,data,delimiter=',')



    
def wrapper(fyle=None, count=None, prep=False):
    """This function will make the necessary (highly specific) calls to learn and apply. AKA this is the shitty, not-pretty function"""

    if fyle == None: fyle = sys.argv[0]
    if len(sys.argv) > 2: prep = sys.argv[2]
    names = [i.replace('\n','') for i in open(fyle).readline() if i != 'objID']
    crange=dict(C=10**np.arange(-4,4))    
    Args = (fyle,names,svm.LinearSVC(),1/3.,.8,20,True,False,None,)    

    if prep:
        #Find the best training parameters for C
        Args[-1] = crange  ;  Args[-3] = False
        fitter = learn(*Args)

        #Find the best number of samples to use

        Args[1] = fitter   ;  Args[-3] = True  ; Args[-1] = None
        N = 2**np.arange(5,17)
        for n in 0, xrange(len(N)):
            t = Thread(target=learn, args=Args)
            if n < len(N)-1: t.daemon(True)
            t.start()

        t.join()
 #should keep all threads running until the most complicated thread finishes



        #Save them
        parm_fit = 


    else:
        if count == None: count = sys.argv[1]
