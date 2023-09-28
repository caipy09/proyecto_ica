import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

today=pd.Timestamp.today()
start=today-pd.Timedelta(days=365)
start_date=start.strftime('%Y-%m-%d')
#print(start_date)

symbol='SPY'
#df_stock=yf.download(symbol,start=start_date,end=today)
df_stock=yf.download(symbol,start=start_date,end=today)[['Adj Close']].copy()
#print(df_stock.head())

df_stock["returns"]=np.log(df_stock['Adj Close']/df_stock['Adj Close'].shift(1))
df_stock.dropna(inplace=True)
#print(df_stock["returns"])

mean_returns=df_stock["returns"].mean()
std_returns=df_stock["returns"].std()
#print(mean_returns,std_returns)

1+np.random.normal(mean_returns,std_returns,10)
#print(1+np.random.normal(mean_returns,std_returns,10)

########## Funcion para la simulacion de Montecarlo ####

def montecarlo_simulation(df_prices, n_simulations=10000, n_days=30, mu=0, sigma=1):
    simulated_prices=np.zeros((n_simulations, n_days))
    initial_price=df_prices.iloc[-1] #seria el precio final: df_stock.tail()
    simulated_prices[:,0]=initial_price

    for i in range(1, n_days):
        simulated_prices[:,i]= simulated_prices[:,i-1]*(1+np.random.normal(mu,sigma,n_simulations))
    return simulated_prices



simulated_prices=montecarlo_simulation(df_stock["Adj Close"], n_simulations=10000, n_days=30, mu=mean_returns, sigma=std_returns)

print(simulated_prices)

###ahora lo paso a un arreglo plano y que el precio inicial(columna 0)
simulated_prices_flat=simulated_prices[:,1:].flatten()
print(len(simulated_prices_flat)) #tendra 29000000 pues le sacamos 1000000 del precio inicial


######## Vamos a calcular con un 95% de confianza cual sera el precio minimo esperado
######## o sea que esperamos con un 95% de confianza que los precios vayan por encima de este

percentile_5=np.percentile(simulated_prices_flat,5)
print(f"95% of the simulated prices will be above {percentile_5:.2f}")

#calculo el intervalo de confianza en el ultimo dia
percentile_2_5=np.percentile(simulated_prices[:,-1],2.5)
percentile_97_5=np.percentile(simulated_prices[:,-1],97.5)
print(f"95% de confianza de que el precio estara entre {percentile_2_5:2f} y {percentile_97_5:2f}")

#valor esperado para la ultima simulacion
expected_value=np.mean(simulated_prices[:,-1])

#####Grafico de las simulaciones
plt.figure(figsize=(12,8))
plt.plot(simulated_prices.T, color="grey", alpha=0.5) #matriz traspuesta
plt.axhline(percentile_5, color="red",linestyle="--")
plt.axhline(percentile_2_5, color="green",linestyle="--")
plt.axhline(expected_value, color="blue",linestyle="--")
plt.show()


df_stock.tail()
expected_value

