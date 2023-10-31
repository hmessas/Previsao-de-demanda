import pandas as pd
import math
import numpy as np
pd.set_option('display.float_format',lambda x:'%.5f'%x)

class SimpleRegression(object):
    def __init__(self, X,Y):
        self.df=pd.DataFrame({'X':X, 'Y':Y})
        self.n=len(self.df)

    def LinearRegression(self):
        self.df['X²']=self.df['X']**2
        self.df['XY']=self.df['X']*self.df['Y']
        a=((self.df["Y"].sum()*self.df['X²'].sum())-(self.df['X'].sum()*self.df['XY'].sum()))/(self.n*self.df['X²'].sum()-(self.df['X'].sum())**2)
        b=(self.n*self.df['XY'].sum()-self.df['X'].sum()*self.df['Y'].sum())/(self.n*self.df['X²'].sum()-self.df['X'].sum()**2)
        return (a,b)

    def ExponencialRegression(self):
        self.df['logY']=np.log(self.df['Y'])
        self.df['X²']=self.df['X']**2
        self.df['XY']=self.df['X']*self.df['logY']
        a=((self.df["logY"].sum()*self.df['X²'].sum())-(self.df['X'].sum()*self.df['XY'].sum()))/(self.n*self.df['X²'].sum()-(self.df['X'].sum())**2)
        a=np.exp(a)
        b=(self.n*self.df['XY'].sum()-self.df['X'].sum()*self.df['logY'].sum())/(self.n*self.df['X²'].sum()-self.df['X'].sum()**2)
        return (a,b)

    def LogarithmicRegression(self):
        self.df['logX']=np.log(self.df['X'])
        self.df['X²']=self.df['logX']**2
        self.df['XY']=self.df['logX']*self.df['Y']
        a=((self.df["Y"].sum()*self.df['X²'].sum())-(self.df['logX'].sum()*self.df['XY'].sum()))/(self.n*self.df['X²'].sum()-(self.df['logX'].sum())**2)
        b=(self.n*self.df['XY'].sum()-self.df['logX'].sum()*self.df['Y'].sum())/(self.n*self.df['X²'].sum()-self.df['logX'].sum()**2)
        return (a,b)

    def PotencialRegression(self):
        self.df['logX']=np.log(self.df['X'])
        self.df['logY']=np.log(self.df['Y'])
        self.df['X²']=self.df['logX']**2
        self.df['XY']=self.df['logX']*self.df['logY']
        a=((self.df["logY"].sum()*self.df['X²'].sum())-(self.df['logX'].sum()*self.df['XY'].sum()))/(self.n*self.df['X²'].sum()-(self.df['logX'].sum())**2)
        a=np.exp(a)
        b=(self.n*self.df['XY'].sum()-self.df['logX'].sum()*self.df['logY'].sum())/(self.n*self.df['X²'].sum()-self.df['logX'].sum()**2)
        return (a,b)

class MultivariableRegression(object):
    def __init__(self,df):
        self.df=df

    def Regression(self):
        Ys=self.df['Y']
        Xs=self.df.drop('Y',axis=1)
        Xs.insert(0,'1',[1 for i in range(len(Xs))])
        ys=np.matrix(Ys)
        xs=np.matrix(Xs)
        xst = xs.transpose()
        xstx = np.dot(xst,xs)
        xstxi =np.linalg.inv(xstx)
        #print((np.dot(xst,ys.transpose())))
        np.set_printoptions(suppress=True)
        betas=np.dot(xstxi,(np.dot(xst,ys.transpose())))
        betas=np.around(betas,3)
        return betas

