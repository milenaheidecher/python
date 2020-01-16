import math
import numpy as np
import matplotlib.pyplot as plt
#import time #necessário para calcular o tempo de execução


###################################
# LENDO A OPÇÃO ESCOLHIDA PELO USUÁRIO
###################################
escolha = input("Digite o número do método a ser utilizado\n 1: Euler\n 2: Euler Modificado \n 3: Runge Kutta 4 ordem \n 4: Taylor de ordem 2 \n 5: Nystrom\n")
#inicio = time.time()    #marca o tempo inicial 
###################################
# definindo a função dos métodos
###################################

if escolha == "1":
    def phi(t,y,h,f):
    # define discretization function     (EULER)
        k1 = f(t, y)
        return k1     # end: Euler

elif escolha == "2":
    def phi(t,y,h,f):
    # define discretization function     (EULER MODIFICADO)
        k1 = f(t, y)
        k2 = f(t+h,y+h*k1)
        return 0.5*(k1+k2)       # end: Euler Modificado

elif escolha == "3":
    def phi(t,y,h,f):
    # define discretization function   (RUNGE KUTTA 4 ordem)
        k1 = f(t, y)
        k2 = f(t+h/2, y + h/2*k1)
        k3 = f(t+h/2, y + h/2*k2)
        k4 = f(t+h, y + h*k3)    
        return 1/6*(k1 + 2*k2 + 2*k3 + k4)     # end: RK4

elif escolha == "4":
    # define discretization function     (TAYLOR 2 ordem)
    def phi(t,y,h,f):
        k1 = f(t, y)
        k2 = fy(t, y)
        return k1 + (h/2)*k2  #end: Taylor segunda ordem
    
elif escolha == "5":
    def phi(t,y,h,f):
    # define discretization function   (NYSTROM)
        k1 = f(t, y)
        k2 = f(t+(2/3)*h, y + (2/3)*h*k1)
        k3 = f(t+(2/3)*h, y + (2/3)*h*k2)
        return 1/4*(k1 + (3/2)*(k2 + k3))     # end: Nystrom
        
###################################
# parametros iniciais
###################################

t0 = 0.0                   # instante inicial
tf = 5.0                    # instante final
n_lista = [16,32,64,128,256,512,1024,2048,4096,8192]  # qtd de ptos a cada execucao para m=1,2,3,...,9

y0 = np.array([0.5])     # y0 tq y(t0)=y0

def f(t, y):
    f0 = 2*y - 8*t    # edo a ser resolvida
    return np.array([f0]) 

def fy(t, y):
    f1 = 4*y - 16*t - 8  #derivada para taylor ordem 2
    return np.array([f1])

def solucao(t):    # expressao analitica para y(t)

    return np.array(4*t-1.5*math.exp(2*t)+2) 

###################################
# executar o metodo algumas vezes
# e fazer os graficos
###################################

def main ():
    # inicialização de cada lista
    h_lista = []
    t_lista = []
    y_lista = []
    erro_lista = []
        
    for n in n_lista:
        
        h = (tf-t0)/float(n)  

        # executa o metodo

        y = np.zeros([n+1])
        t = np.zeros(n+1)
        
        t[0] = t0
        y[0] = y0
        
        for i in range(0,n):
            t[i+1] = t0 + (i+1)*h                  
            y[i+1] = y[i] + h*phi(t[i],y[i],h,f)
        # adiciona nos dados para o grafico e a tabela
        t_lista.append(t)
        y_lista.append(y)
        h_lista.append(h)
        erro_lista.append( np.linalg.norm(y[n] - solucao(tf)))
            
    # exibe o grafico com as curvas de y
    for w in range(len(n_lista)):
        t = t_lista[w]
        y = y_lista[w]
        
        plt.plot(t, y, label="n=%d"%n_lista[w], color = 'black', linestyle=(0,(w+1,2,w,3)), linewidth =0.7)
    
    truevalue=list(solucao(x) for x in t)
    plt.title('aprox da função com n variando')
    plt.xlabel('t[i]')
    plt.ylabel('y[i]')
    plt.grid(True)
    # plota a solução exata
    plt.plot(t, truevalue, color = 'black', label = "sol exata", linewidth =0.5)
    plt.legend()
    plt.show()
      
    # exibe o grafico com o erro
    
    plt.title('decaimento do erro no instante final')
    plt.xlabel('h')
    plt.ylabel('erro(h, %1.2f)'%tf)
    plt.grid(True)
    plt.plot(h_lista,erro_lista, color = 'black', linewidth =0.8)
    plt.show()
    
    # escreve a tabela
    print(len(n_lista))
    print()
    print("n", "h", "erro(h, %1.2f)"%tf, "razão entre erros", sep='\t\t')
    for w in range(len(n_lista)):
        r=0
        if w > 0:
            r = abs(erro_lista[w-1])/abs(erro_lista[w])
            
        print("%5d %24.16e %24.16e %12.5e " % (n_lista[w], h_lista[w], erro_lista[w], r))

###################################
# executar
###################################

main()
#fim = time.time()  #marca o tempo final
#print(fim - inicio)  #mostra o tempo final
