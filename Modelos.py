import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np



class WinterModel():
    def __init__(self,df:pd.DataFrame, n:int, alpha:float=0.5,beta:float=0.5,gamma:float=0.5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.window = n
        self.df = df
        self.df=self.preprocessing()
        self.L0,self.T0=self.LinearRegression()[0],self.LinearRegression()[1]
        self.df=self.fit()

    def preprocessing(self):
        self.df = self.df[['Demanda']]
        self.df['Demanda'].interpolate()
        self.df['Periodo'] = [i+1 for i in range(len(self.df['Demanda']))]
        return self.df
    
    def LinearRegression(self):
        L=len(self.df)
        self.df['Periodo²']=self.df['Periodo']**2
        self.df['XY']=self.df['Periodo']*self.df['Demanda']
        a=((self.df["Demanda"].sum()*self.df['Periodo²'].sum())-(self.df['Periodo'].sum()*self.df['XY'].sum()))/(L*self.df['Periodo²'].sum()-(self.df['Periodo'].sum())**2)
        b=(L*self.df['XY'].sum()-self.df['Periodo'].sum()*self.df['Demanda'].sum())/(L*self.df['Periodo²'].sum()-self.df['Periodo'].sum()**2)
        self.df=self.df[['Demanda','Periodo']]
        return (a,b)
    
    def fit(self):
        self.si=[]
        self.df['DDT']=self.L0+self.T0*self.df["Periodo"]
        self.df['Fator Sazonal']=self.df['Demanda']/self.df['DDT']
        self.L=[]
        self.T=[]
        self.S=[]
        self.P=[]
        for i in range(self.window):
            self.si.append(self.df['Fator Sazonal'].iloc[i::self.window].mean())
            #print(self.si)
        for i in range(len(self.df)):
            if i==0:
                self.L.append(self.L0)
                self.T.append(self.T0)
                self.S.append(self.gamma*(self.df['Demanda'].iloc[i]/self.L[i])+(1-self.gamma)*self.si[i])
                self.P.append(np.NaN)
            elif i<self.window:
                #print(i,len(self.L),len(self.T),len(self.S))
                Li=self.alpha*(self.df['Demanda'].iloc[i]/self.si[i])+(1-self.alpha)*(self.L[i-1]+self.T[i-1])
                self.L.append(Li)
                Ti=self.beta*(self.L[i]-self.L[i-1])+(1-self.beta)*(self.T[i-1])
                self.T.append(Ti)
                Si=self.gamma*(self.df["Demanda"].iloc[i]/self.L[i])+(1-self.gamma)*self.si[i]
                #print((self.df["Demanda"].iloc[i]),self.L[i],self.si[i])
                self.S.append(Si)
                #print(self.L,self.T,self.S)
                self.P.append((self.L[i-1]+self.T[i-1])*self.si[i])
            else:
                Li=self.alpha*(self.df['Demanda'].iloc[i]/self.S[i-self.window])+(1-self.alpha)*(self.L[i-1]+self.T[i-1])
                self.L.append(Li)
                Ti=self.beta*(self.L[i]-self.L[i-1])+(1-self.beta)*(self.T[i-1])
                self.T.append(Ti)
                Si=self.gamma*(self.df["Demanda"].iloc[i]/self.L[i])+(1-self.gamma)*self.S[i-self.window]
                #print((self.df["Demanda"].iloc[i]),self.L[i],self.si[i])
                self.S.append(Si)
                self.P.append((self.L[i-1]+self.T[i-1])*self.S[i-self.window])

        self.df['L']=self.L
        self.df['T']=self.T
        self.df['S']=self.S
        self.df['Previsão']=self.P

        return self.df
            
    def evaluate(self):
        mae = np.mean(np.abs(self.df['Demanda'] - self.df['Previsão']))
        mad = mae  # MAD and MAE are the same in this context
        mse = np.mean((self.df['Demanda'] - self.df['Previsão'])**2)
        rmse = np.sqrt(mse)
        mpe = np.mean((self.df['Demanda'] - self.df['Previsão']) / self.df['Demanda']) * 100  # As percentage
        mape=np.mean(np.abs(self.df['Demanda'] - self.df['Previsão']) / self.df['Demanda'])
        errors={"MAE": mae, "MAD": mad, "MAPE": mape, "MSE": mse, "RMSE": rmse, "MPE": mpe}
        report=pd.DataFrame(errors,index=['Valor'])
        report=report.transpose()
        return report

    def graph(self):
        fig=plt.figure()
        ax1=fig.add_subplot(211,ylabel='Unidades')
        self.df['Demanda'].plot(ax=ax1,color='b',lw=.8)
        self.df['Previsão'].plot(ax=ax1,color='g')
        ax1.legend(loc='best')

    def predict(self,future):
        fs=list(self.df['S'][-self.window:])
        last_L=self.df['L'].iloc[-1]
        last_T=self.df['T'].iloc[-1]
        #print(last_L,last_T)
        predictions=[]
        #print(fs)
        for i in range(future):
            predictions.append((last_L+(i+1)*last_T)*fs[i])
        return predictions
        