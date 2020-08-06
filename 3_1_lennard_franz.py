""" Vergleich nummerischer Integrationsmethoden 

Methoden:
- Mittelpunktregel
- Trapezregel
- Simpsonregel

Der Betrag des relativen Fehlers zum exakten Wert des Integrals wird
doppeltlogarithmisch ueber h dargestellt.

Es wird zusaetzlich das jeweils erwartete Skalierungsverhalten eingetragen

"""

import numpy as np
import matplotlib.pyplot as plt  #Importe


def funktion_1(x):
    """ Gegebene Funktion innerhalb des ersten Integrals"""
    return np.cosh(2*x)


def funktion_2(x):
    """ Gegebene Funktion innerhalb des zweiten Integrals"""
    return np.exp(-100*(x**2))


def funktion_3(x):
    """ Gegebene Funktion innerhalb des dritten Integrals"""
    return 0.5 * (1.0+np.sign(x))
    
#benötigte Funktionen für Integrale

def integration_Mittelpunkt(funktion, a, b, N):
    """ Funktion zur nummerischen Integration mittels Mittelpunktregel 
    einer Übergebenen Funktion im Intervall (a,b) mit N Stützstellen"""
    x,h = np.linspace(a, b, N,endpoint=False, retstep=True)
    #Verwendung von linspace um Array der Positionen der Stützstellen 
    #im Intervall (a,b), sowie die Schrittweite h zu bestimmen
    #Umsetzung der Formel lt. Vorlesung 
    return h*np.sum(funktion(x+h/2))
    
def integration_Trapez(funktion,a,b,N):
    """ Funktion zur nummerischen Integration mittels Trapezregel 
    einer Übergebenen Funktion im Intervall (a,b) mit N Stützstellen"""
    x, h = np.linspace(a,b,N,endpoint=False, retstep=True)
    #Verwendung von linspace um Array der Positionen der Stützstellen 
    #im Intervall (a,b), sowie die Schrittweite h zu bestimmen
    #Umsetzung der Formel lt. Vorlesung 
    T=h*(funktion(a)/2.0 + np.sum(funktion(x+h)) - funktion(b)/2.0)
    return T
    
    
       
def integration_Simpson(funktion,a,b,N):
    """ Funktion zur nummerischen Integration mittels Mittelpunktregel 
    einer Übergebenen Funktion im Intervall (a,b) mit N Stützstellen"""
    
    x, h = np.linspace(a,b,N,endpoint= False, retstep=True)
    #Verwendung von linspace um Array der Positionen der Stützstellen 
    #im Intervall (a,b), sowie die Schrittweite h zu bestimmen
    #Umsetzung der Formel lt. Vorlesung 
    S = (h/6.0)*(-funktion(a)+ 2.0*np.sum(funktion(x)) + 
         4.0*np.sum(funktion(x+(h/2.0))) + funktion(b)) 
    return S
    
    
def relativer_fehler(num, analy):
    """Funktion zur Berechnung des Betrags des relativen Fehlers eines 
    Nummerischen Werts (num) vom analytischen Wert (analy)  """
    return abs((analy-num)/analy) 
    
def integral_funktion_1(a, b):
    """Funktion zur analytischen Berechnung des Integrals von funktion_1
    im Intervall (a,b) mithilfe der Stammfunktion""" 
    return np.sinh(2*b)/2 - np.sinh(2*a)/2

def integral_funktion_2(a, b):
    """Ausgabe des genäherten Werts des Integrals"""
    #Berechnung mit Stammfunktion war nicht möglich deswegen nur Näherung
    #mithilfe von Wolfram Alpha
    return 0.177245385090551602
    

def integral_funktion_3(a, b):
    """Funktion zur analytischen Berechnung des Integrals von funktion_3
    im Intervall (a,b) mithilfe der Stammfunktion""" 
    #Berechnung mit Stammfunktion war nicht möglich deswegen nur Näherung
    #mithilfe von Wolfram Alpha
    return np.pi/4
    

def iteration(integrations_methode, funktion, a, b, analytisches_Integral, 
              anzahl):
    """Funktion zur Iteration des relativen Fehlers der numerischen 
    Integration über verschiedene Anzahlen von Stützstellen N bzw. 
    der Schrittweite h im Intervall (a,b).
    Mögliche Funktionen der Integrationsmethoden: integration_Mittelpunkt
                                                  integration_Trapez
                                                  integration_Simpson 
    analytisches_Integral: Benötigt Funktion zur Berechnung eines 
                           analytischen Vergleichswert im Intervall(a,b)
    anzahl: Wahl der gewünschten Anzahl an Iterationen 
    
    """
    #Erstellung eines Arrays für Anzahl von Teilintervallen
    #logarithmische Wahl der Anzahl der Teilintervalle aufgrund 
    #doppeltlogaritmischer Darstellung um gewünschten h Bereich zu erhalten
    #(Vorgabe auf Baltt)
    #Ausserdem Verwendung von np.unique, da ausgrund Verwendung von Integers 
    #bei kleinen n Zahlen mehfach vorkommen.
    n = np.unique(np.int32(10**np.linspace(0, 5, anzahl)))
    #Bestimmung der neuen Länge des Array nach Verwendung von np.unique
    #neue Länge wird für weitere Berechnungen benötigt (Arrays für rel. fehler 
    #und h)
    len_n = len(n)
    
    fehler = np.zeros(len_n)
    h = np.zeros(len_n)             #Erstellung leerer Arrays   
    
    for i in range(1,len_n):
        fehler[i] = relativer_fehler(integrations_methode(funktion, a, b, n[i]),
                                     analytisches_Integral(a,b))  
        h[i]= (b-a)/n[i]
                            #Berechnung des relativen Fehlers für 
                            #gegebene Parameter und einfügen in Arrays  
    return h,fehler

def main():
    """Hauptprogramm"""

    print(__doc__)
    
    #Wahl der zu integrierenden Funktion und Intervall 
    Funktion = funktion_1
    a = -np.pi/2.0
    b =np.pi/4.0
    Analytisches_Integral = integral_funktion_1
    #Länge der Iteration um Forderung nach 10^(-4)<h<1 zu erfüllen
    anzahl=1000
    #Berechnung des Relativen Fehlers mit dazugehörigen h-Werten durch
    #Iteration. 
    h_M,Fehler_M=iteration(integration_Mittelpunkt, Funktion, a,
                           b, Analytisches_Integral, anzahl)
    
    h_T,Fehler_T=iteration(integration_Trapez, Funktion, a,
                           b, Analytisches_Integral, anzahl)
    
    h_S,Fehler_S=iteration(integration_Simpson,Funktion, a,
                           b, Analytisches_Integral, anzahl)
                           
    #Erstellen einer Plotfensters mit doppellogarithmischer Skala 
    fig= plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, xscale="log" , yscale="log")
    #Gitter hinzugefügt
    ax.grid(True)
    
    #Achsenbereich und Plotbeschriftung 
    ax.set_xlim([1e-4, 1e0])
    ax.set_ylim([1e-16, 1e0])
    ax.set_xlabel("$h$")
    ax.set_ylabel("|Relativer Fehler|")
    ax.set_title("Numerische Integrationsmethoden")
    
    #Plot der Punnkte mit versch. Farben und Beschriftung für Legende 
    ax.plot(h_M,Fehler_M, linestyle='None', marker='.', color='g' ,ms=3,
            label='Mittelpunktsregel' )
    ax.plot(h_T,Fehler_T, linestyle='None', marker='.', color='b', ms=3, 
            label='Trapezregel' )
    ax.plot(h_S,Fehler_S, linestyle='None', marker='.', color='r', ms=3, 
            label='Simpsonregel' )
    
    #Einzeichnen der Legende
    ax.legend(numpoints=3)
    
    #Einzeichnen des erwarteten Skalierungverhaltens
    h_werte = np.array([2e-4, 1.0])
    ax.plot(h_werte, 0.13*h_werte**2, 'g--', lw=1.5)
    ax.text(4.5e-2, 1e-4, r"$\sim h^2$", fontsize=18, color="g")
    
    h_werte = np.array([2e-4, 1.0])
    ax.plot(h_werte, 0.42*h_werte**2, 'b--', lw=1.5)
    ax.text(2.5e-2, 1e-3, r"$\sim h^2$", fontsize=18, color="b")
    
    h_werte = np.array([1e-3, 1])
    ax.plot(h_werte, 0.003*h_werte**4, 'r--', lw=1.5)
    ax.text(5e-2, 1e-8, r"$\sim h^4$", fontsize=18, color="r")

    plt.show()

if __name__ == "__main__":
    main()

""" 
Analytische Integrale
a) integral_Funktion_1=6.9250191298
b) integral_Funktion_2=0.1772453850
c)integral_Funktion_3=pi/4


Diskusion der Ergebnisse
a)
Für das erste zu berechnende Integral zeigt sich das die Mittelpunkt 
und Trapezregel ähnliche Genauigkeiten liefern. Wobei die 
Mittelpunktsregel jedoch für alle werte von h einen geringern relativen 
Fehler liefert. Bei beiden Methoden skaliert der relative Fehler mit h^2.
Der minimal erreichbare Fehler, im geforderten Bereich, beider Methoden 
liegt bei etwas über 10e-09 für h^2.

Bei der Simpsonregel ergibt sich im Vergleich zu den andern beiden
Methoden ein deutlich geringerer Fehler bei allen h-Werten. Bei h-Werten
im Bereich zwischen 1e-01 und 1 ist der Fehler der Simpsonregel in etwa 
eine Größenordnung kleiner als die der anderen Methoden und sinkt 
aufgrund des Skalierungsverhaltens von h^4 stark weiter. 
Das Minimum des relativen Fehlers mithilfe der Simpsonregel liegt unter 
1e-15. Ab h-Werten von 5e-04 verhält sich der Fehler nicht mehr linear 
(in der doppellog. Darstellung), sondern es bilden sich Plataeus aus. 
Diese Ausbildung von Plataeus stellt diskrete Werte für die fehler dar, dies 
kommt aufgrund von Auslöschung aufgrund begrenzter Genauigkeit zustande.

b)
Bei der numerischen Integration des zweiten Integrals ist bei keiner der 
Methoden das zu erwartende Skalierungsverhalten zu erkennen. 
Für alle drei Methoden ergeben sich Plateaus (diskrete Werte (siehe a)) für
h-Werte ab 5e-2 und kleiner. Für größere h-Werte steigt der relative Fehler
aller drei Methoden stark an. Der relative Fehler liegt für h=1e-1 nur noch 
im Bereich von 1e-3.
Dies kommt aufgrund des Verlaufs der Funktion zustande. Die Funktion ähnelt der
Deltadistibution, weshalb beim integriere nur ein kleiner Bereich um 0 einen 
Beitrag zum Integral liefert. Deswegen ist schon durch eine relativ geringe 
Anzahl an Teilintervallen eine sehr gute Approximation möglich. Andererseits 
kann es durch zu große Teilintervalle und des straken Ansteigs der Funktion um 0 
sehr schnell zu einer großen Überschätzung des Integrals kommen, wodurch der
starke Ansteig des relativen Fehlers für h>5e-2 erklärt werden kann.

c)
Die relativen Fehler der numerischen Methoden zeigen beim dritten Integral nicht
die zu erwartenden Skalierungsverhalten mit h. Alle drei Methoden zeigen ein 
Skalierungsverhalten mit h**1. Weiterhin ist zu erkennen, dass bei einigen 
einzelnen Werten für h (unterhalb 2e-2) relative Fehler im Bereich von 1e-16 
auftauchen. Diese vereinzelten sehr kleinen Werte für den relativen Fehler 
kommen aufgrund der Unstetigkeit und Symmetrie der Funktion Zustande.
  
"""

