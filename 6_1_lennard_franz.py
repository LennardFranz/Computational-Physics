"""Quantenmechanik von 1D-Potentialen: Doppelmuldenpotential

Bestimmung und Darstellung der Eigenwerte (Eigenenergien) und 
Eigenfunktionen eines asymmetrischen Doppelmuldenpotentials mithilfe der 
Ortsraumdiskretisierung.

Dabei wurde folgendes asymerisches Doppelmuldenpotential mit dem 
Parameter A = 0.15 verwendet.
V(x) = x^4 -x^2 - Ax

Für die Ortsraumdiskretiersung wurde h_eff = 0.07, x in [-1.5,1.5] und 
250 Iterationen verwendet.

 
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh    #Importe


def potential_DM(x, A):
    """Funktion zur Berechnung des asymetrisches Doppelmuldenpotentials
    für gegebene Werte x (Array) und Parameter A.
    """
    return x**4 - x**2 - A*x

def algebra_ew_ev(x, delta_x, N, h_eff, V):
    """Berechnung der Eigenwerte (Eigenenergien) und Eigenfunktionen der 
    diskreten x Werte. 
    Dafür wird eine Matrix gemäß der Vorlesung erstellt. Dabei wird die
    Hauptdiagonale mit V(x) + 2*z und die beiden Nebendiagonalen mit -z.
    """
    #Berechnung des Parametsers z laut Vorlesung 
    z = h_eff**2 / (2.0*(delta_x)**2)
    #Erstellen der benötigten Matrix laut Vorlesung 
    #Erste Summand stellt dei Hauptdiagonale dar, der zweite und dritte
    #Summand die beiden Nebendiagonalen.
    matrix = (np.diag(V + 2.0 * z) + np.diag(np.ones(N - 1)*(-z), k=+1) 
             + np.diag(np.ones(N-1)*(-z), k=-1))
    #scipy-Funktion zur Bestimmung der Eigenwerte 
    #Eigenwerte werden durch die verwendete Funktion in aufsteigender Größe bzw.
    #sortiert zurückgegeben.
    #Die Eigenvektoren werden ebenfalls in der zu den jeweiligen Eigenwerten 
    #passenden Reihenfolge zurückgegeben. 
    ew, ev = eigh(matrix)
    #Normierung der Eigenfunktionen 
    ev = ev/np.sqrt(delta_x)
    return ew, ev
    
def Plot(x, V, E_max, EW, EV, skalierung):
    """ Funktion zur Ploterstellung. 
    Es wird das übergebene Potential mit allen Energieeigenwerten kleiner
    als festgelegtes E_max, sowie den zugehörigen skalierten 
    Eigenfunktionen dargestellt.
    """
    #Plot des Potentials 
    plt.plot(x, V, color='k')
    #Anzahl der benötigten Eigenwerte kleiner als E_max
    anzahl_EW = np.sum(EW < E_max)
    #Schleife zum Plotten der benötigten Eigenwerten
    #Aufgrund der Sortierung der Eigenwerte in aufstiegender Reihenfolge ist 
    #die Verwendung dieser Schleife möglich 
    for i in range(anzahl_EW):
        #Plotten der Eigenwerte. 
        plt.plot(x, np.ones(len(x)) * EW[i], lw=3, ls='--', color='k')
        #Plotten der Eigenwerte zu Eigenfunktionen 
        #Skalierung der Amplituden der Eigenfunktionen zur besseren Darstellung
        plt.plot(x, skalierung * EV[:, i] + EW[i], lw=3)
        

def main():
    """Hauptfunktion
    - Definition der benötigten Parameter 
    - Festlegung des Plotbereichs
    - Diskretisierung
    - Aufruf der benötigten Funktionen 
    - Darstellung des Plot"""
    print(__doc__)
    X = 1.5                        #Intervgallgrenzen                
    N = 300                        #Matrixgröße
    h_eff = 0.07                   #gegebener Parameter für Verfahren               
    A = 0.06                       #Parameter im Doppelmuldenpot.               
    Emax = 0.1                     #festgelegte maximal Energie
    
    #Skalierungsfaktor für Darstellung der Eigenfunktionen 
    skalierung = 0.018
    #Berechnung des Diskretisierungsschrits mit Formel aus Vorlesung
    x_min, x_max = -X, X
    delta_x= (x_max - x_min)/ (N - 1)
    #Berechnung des Werte-Arrays der diskreten Werten für x
    x = np.linspace(x_min + delta_x, x_max - delta_x, N)
    
    #Festlegung des verwendetes Potentials 
    V = potential_DM(x, A)
    
    #Speicherung der Eigenwerte und Eigenfunktionen as definierten Funktion
    EW, EV = algebra_ew_ev(x, delta_x, N, h_eff, V)
    
    #Erstellung des Plotfensters
    plt.figure(figsize=(10,8))
    ax = plt.subplot(111)
    
    #Plotbeschriftung 
    ax.set_title('Teilchen im asymmetrischen Doppelmuldenpotential')
    ax.set_xlabel('x')
    ax.set_ylabel('Potential V(x) und Eigenwerte mit Eigenfunktionen')
   
   #Festlegung Achsenbereich
    ax.set_xlim([-X , X])
    ax.set_ylim([-0.3, 0.1]) 
   
    #Aufruf der Plot Funktion und Darstellung des Plots
    Plot(x, V, Emax, EW, EV, skalierung)
    plt.show()

if __name__ == "__main__":
    main()
"""
a) 
Wahl der Diskretisierungsschrittweite bzw. der Matrixgröße:

Für die Wahl der Matrixgröße, woraus sich dann auch die 
Diskretisierungsschrittweite ergibt, wurde 300 gewählt. 

Für Matrixgrößen kleiner 100 werden die Plots der Eigenfunktionen und des
Potentials etwas eckig dargestellt, was aufgrund der zu kleinen
Iterationszahl während der Berechnung zustande kommt.  

Da für Spätere Betrachtungen noch die Zoom-Funktion verwendet werden muss, 
sollte die Matrixgröße  mindestens zwischen 200-300 gewählt werden um die 
benötigte Genauigkeit zu errerichen. 

Eine weitere Vergrößerung der Matrixgröße hat keinen sichtbaren Einfluss
auf die Eigenfunktionen, sondern nur minimal auf die Eigenwerte.  

Wahl des betrachteten Intervalls:

Das gewählte Intervall [x_min,x_max] entspricht [-1.5, 1.5].

Dieses Intervall wurde aufgrund der Eigenschaften des verwendeten 
asymmetrischen Doppelmuldenpotential gewählt.

Bei kleineren Intervallen bspw. [-1,1] könnte sowohl das Potential als 
auch die Eigenfunktionen nicht komplett dargestellt werden.
Für größere Intervalle werden keine physikalisch relevanten Bereiche
dargestellt, da für das gewählte Intervall alle betrachteten Eigenfunktionen
komplett dargestellt werden können.


b)
Struktur der Eigenfunktionen:

Die allgemeine Struktur der Eigenfunktionen entspricht periodischen
Schwingungen. Wobei für verschiedene Eigenwerte natürlich verschiedene
Periodizitäten entstehen. Des Weitern beinflusst die Asymmetrie des 
Potetials die Symmetrie der Eigenfunktionen, wodurch diese dann in manchen
Bereichen von allgeinen periodischen Schwingungen abweichen.
Für der kleinsten Eigenwert ist die zugehörige Eigenfunktion größteteils 
konstant 0, bis auf eine Extremstelle in der tieferen Mulde des 
Doppelmuldenpot., welche an den Grenzen dann wieder gegen null geht.

Die Eigenfunktion zum nächst größeren Eigenwert ist nur im 
Bereich des der höheren Mulde ungleich 0. Sie besitzt ebenfalls dort eine 
Extremstelle. Durch Zommen ist außerdem ein Nulldurchgang und somit auch
zwei Extremstellen sehr kleiner Amplitude im Bereich des tiefern Potentials
zu erkennen.    

Die dritte Eigenfunktion ist sehr änhlich zu der des kleinsten Eigenwerts
, jedoch hat sie das Maximuum im Bereich der höheren Mulde des Potentials.
Anders als die erste Eigenfunktion ist die dritte jedoch nicht konstant im 
restlichen Bereich. Nach zoomen zeigt sich wieder im Bereich der höheren 
Mulde eine Schwingung kleiner Amplitude mit einem Nulldurchgang.

Die vierte Eigenfunktion ist wieder ähnlich zur dritten, jedoch befinden
sich die eindeutig sichtbaren zwei Extremstellen mit einem Nulldurchgang
im Bereich der höheren Mulde und weitere drei Extremstellen mit kleinerer
Amplitude in der tieferen Mulde.
Die verschieden großen Amplituden der Extremstellen kommen dadurch
zustande, dass die Eigenenergien unterhalb des lokalen Maximums des
Potentials zwischen den beiden Mulden liegen und deswegen die
Aufenthalswahrschinlichkeit in einer Mulde maximal ist.

Die Eigenfunktionend der nächsten zwei größeren Eigenwerte ergebn sich
dann Schwingungen über den gesamten Bereich des Potentials, da jetzt die
Eigenenergien größer als das lokale Maximum des Potentials sind. 
Jedoch werden die Funktionen im Bereich des lokalen Maximums durch dieses 
beeinflusst. Dabei wird das in diesem Bereich vorkommende Maximum der
Eigenfunktion gestreckt.

Für höhere Eigenernergien nähern sich die Eigenfunktionen dann Sinus-
bzw. Cosinusschwingungen über den Bereich des Potentials an.
Knotensatz:
Der Knotensatz ist somit für alle Eigenfunktionen gültig, da die
Eigenfunktion zum n-ten Eigenwert n-1 Nullstellen besitzt.
Dabei ist anzumerken das die Zoom-Funktion benutzt wurde um alle
Nullstellen zu finden.

h_eff:

Für die Vergößerung von h_eff sinkt die Anzahl der Eigenwerte und 
Eigenfunktionen sehr schnell, wohingegen die Veringerung von h_eff eine 
starke Erhöhung der Eigenwerte verursacht. Somit lässt sich sagen, dass 
h_eff antiproportional zu der Anzahl der Eigenwerte ist.


c) 
A=0.0 bzw. symmetrisches Duppelmuldenpotential

Für diesen Fall treten die Eigenwerte und zugehörigen Eigenfunktionen 
immer zu zweit auf. Dabei besitzen die beiden Eigenfunktionen eine 
Symmetrie zueinander.
Die Eigenfunktionen der ersten beiden Eigenwerte besitzen beide eine
Extremstelle im Bereich der linken Mulde, jedoch einmal Minimum und 
Maximum. Die Funktionen scheinen in diesem Bereich symmetrisch, bezüglich
der Horizontalen zu sein. Im Bereich denr rechten Mulde sind beide 
Funktionen annähernd gleich.

Fast Analog verhalten sich die Eigenfunktionen der nächsten beiden
Eigenwerte. Diese sind in der linken Mulde annähernd gleich und in der
Rechten symetrisch zueinander.  

Analog verhalten sich die Eigenfunktionen der nächsten beiden Eigenwerte.

Des Weitern ist zu beobachten das die Differenz der Eigenwerte Duplets
mit größer werdenden Eigenwerten ebenfalls zunimmt. 
"""
