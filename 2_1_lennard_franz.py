"""Berechnung und Darstellung des relativen Fehler numerischer Differenziation 
mithilfe verschiedener Methoden. Explizite Darstellung der Ableitung Funktion
arctan(x**2) an der Stelle 1/3 im Bereich für h von 1e-10 bis 1"""

import numpy as np
import matplotlib.pyplot as plt  
#Importe

def funktion(x):
    """Definition der gegebenen Funktion"""
    return np.arctan(x**2)

def dfunktion(x):
    """ erste Ableitung von f(x) = arctan(x**2)"""
    return 2*x/(x**4 + 1) 
    
    
def ableitung_V(funktion, x_0, h):
    """Funktion zur Berechnung des Werts der ersten Ableitung mit 
    Vorwärtsdifferenz einer Funktion an der Stelle x_0 für gegebenes 
    h"""
    return 1/h * (funktion(x_0+h)-funktion(x_0))
    

def ableitung_Z(funktion, x_0, h):
    """Funktion zur Berechnung des Werts der ersten Ableitung mit 
    Zentraldifferenz einer Funktion an der Stelle x_0 für gegebenes
     h """
    return 1/h*(funktion(x_0+h/2)-funktion(x_0-h/2))
    
def ableitung_E(funktion, x_0, h):
    """Funktion zur Berechnung des Werts der ersten Ableitung mit 
    Extrapolierter Differenz einer Funktion an der Stelle x_0 
    für gegebenes h """
    return 1/(3*h)*(8*(funktion(x_0+h/4)-funktion(x_0-h/4))-
               (funktion(x_0+h/2)-funktion(x_0-h/2)))
               
def relativer_fehler(num,analy):
    """Funktion zur Berechnung des relativen Fehlers eines 
    Nummerischen Werts (num) vom analytischen Wert (analy)  """
    return abs((analy-num)/analy) 

def main():
    """Hauptprogramm"""
    
    print(__doc__)
    
    h = 10**np.linspace(-10,0,250)   #Array für h-Werte (logarithmisch)
    x_0 = 1/3                       #geg. Stelle zur Auswertung der Ableitung
    dYdx = dfunktion(x_0)       #analytischer Wert der 1. Ableitung
   
    dydx_Z = ableitung_Z(funktion, x_0, h)
    dydx_V = ableitung_V(funktion, x_0, h)
    dydx_E = ableitung_E(funktion, x_0, h)
    #Berechnung der Werte der ersten Ableitung für x_0=1/3 und h  

    fehler_Z = relativer_fehler(dydx_Z, dYdx)
    fehler_V = relativer_fehler(dydx_V, dYdx)
    fehler_E = relativer_fehler(dydx_E, dYdx)
    #Berechnung des relativen Fehlers des nummersich berechneten Werts
    #vom analytischen Wert.

    fig= plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, xscale="log", yscale="log")
    #Erstellung des Plotfensters mit doppelt logarithmischer Skala
    
    ax.set_xlim([10**(-10) , 1])
    ax.set_ylim([10**(-15), 10])
    #Achsenbereich festlegen 
    
    plt.plot(h,fehler_V, 'g', ls = 'None' , marker= '.' , ms= 2, 
    label='Vorwärtsdiffernz')
    plt.plot(h,fehler_Z, 'b', ls = 'None' , marker= '.' , ms= 2, 
    label='Zentraldifferenz' )
    plt.plot(h,fehler_E, 'r', ls = 'None' , marker= '.' , ms= 2,
    label='Extrapolierte Differenz' )
    #Plotten der relativen Fehler in Abhängigkeit von h     
    #Nutzen von Farben und Labels zur Kennzeichnung 
    
    ax.grid(True)
    #Hinzufügen eines Gitters um Abhängigkeit des Fehlers einfach
    #ablesen zu können 
    
    #Einzeichnen und Beschriften des Skalierungsverhaltens
    h_werte = np.array([1e-8, 1.0])
    ax.plot(h_werte, 3*h_werte, 'g-', lw=1.5)
    ax.annotate(r'$\sim h^1$', xy=(1e-5, 1e-3), color='g')
    
    h_werte = np.array([1e-5, 1.0])
    ax.plot(h_werte, 0.2*h_werte**2, 'b-', lw=1.5)
    ax.annotate(r'$\sim h^2$', xy=(1e-04, 1e-07), color='b')
    
    h_werte = np.array([2e-3, 1.0])
    ax.plot(h_werte, 0.05*h_werte**4 , 'r-', lw=1.5)
    ax.annotate(r'$\sim h^4$', xy=(5e-03, 1e-8), color='r')
    
    h_werte = np.array([1e-11, 1e-3])
    ax.plot(h_werte, 1e-18*h_werte**(-1), 'k-', lw=1.5)
    ax.annotate(r'$\sim h^{-1}$', xy=(1e-07,5e-13), color='k')
    
    ax.legend(numpoints=3) #Legende  

    ax.set_title('Relativen Fehler der numerischen Differenziation')
    ax.set_xlabel("h")
    ax.set_ylabel("relativer Fehler")
    
   

    plt.show()

if __name__ == "__main__":
    main()



#Ursache des h^-1 Verhalten:
#Prinzipiell sollte der relative Fehler für beleibig kleine Werte von h
#aufgrund der besseren Näherung ebenfalls beleibig klein werden,
#jedoch kommt es aufgrund der Verwendung von Gleitkommazahlen zur 
#Auslöschung da zwei fast gleich große Zahöen voneinander subtrahiert
#werden und es somit zu einem Verlust der Genauigkeit kommt.
#Weiterhin ergibt sich für genügend kleiner werdende h bei allen drei Methoden 
#der Wert 0 (Auslöschung) wodurch der Fehler abhängig von der gewählten Stelle 
#zur Auswertung beleibig groß werden kann. 
#Explizit kommt es bei beispielsweise bei der Vorrwärtsdifferenziation 
#aufgrund der Subtration von f(x+h) und f(x) zu Auslöschung. 


#optimaler Wert für h:
#Methode;Fehler;h-Wert
#Vorwärtsdiffernz;2e-08 ;1.2e-08 
#Zentraldiffernz;2.2e-11 ;1.5e-05   
#Extrapolierte Differnz;3.7e-13 ;2.5e-03

#Alle Werte für h und die dazugehörigen Fehler würden mithilfe der
#Zoomfunktion und der Angabe der Mausposition ermittelt.
#Es wurde der jeweils kleinste Wert für den relativen Fehler bei einem 
#klaren lineraren Verlauf gewählt, wodurch der Fehler vieleicht zu groß
#abgeschätz wird, jedoch mit Sicherheit angenommen werden kann.
