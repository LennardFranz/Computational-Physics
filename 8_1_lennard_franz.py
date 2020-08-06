"""Programm zur Betsimmung der Eigenwerte und -funktionen eines Teilchens im
periodischen Potentials V(x)=A*cos(2*pi*x),x=[0,1) mit A=1 und h_eff=0.2.

Darstellung des Eigenwertspektrums (links) in Abhängigkeit der Bloch-Phase k 
für E(k)<7.

Durch Mausklick ist Wahl eines Wertes für die Bloch-Phase möglich. Für Diesen 
Wert werden im rechten Plot die EIgenfunktionen auf Höhe der Eigenenergien über 
vier Perioden des Potentials dargestellt. 
"""


import numpy as np
import matplotlib.pyplot as plt
import functools
from scipy.linalg import eigh   #Importe



def potential_DM(x, A):
    """Funktion zur Berechnung des periodischen Potentials
    für gegebene Werte x (Array) und Parameter A.
    """
    return A*np.cos(2*np.pi*x)
    

def diagonalisierung(hquer, x, V, k_bloch):
    """ Angepasste Version für periodische Potentiale aus quantenmechanik.py
    Anpassungen an Matrix-Darstellung des HAmilton-Op. nach Vorlesung
    
    Berechne sortierte Eigenwerte und zugehoerige Eigenfunktionen.

    Parameter:
        hquer: effektives hquer
        x: Ortspunkte
        V: Potential als Funktion einer Variable
    Rueckgabe:
        ew: sortierte Eigenwerte (Array der Laenge N)
        ef: entsprechende Eigenvektoren, ef[:, i] (Groesse N*N)
    """
    delta_x = x[1] - x[0]
    v_werte = V(x)                                         # Werte Potential

    N = len(x)
    z = hquer**2 / (2.0*delta_x**2)                        # Nebendiagonalelem.
    #angepasster Hamilton-Op.
    h = (np.diag(v_werte + 2.0*z) +
         np.diag(-z*np.ones(N-1), k=-1) +                  # Matrix-Darstellung
         np.diag(-z*np.ones(N-1), k=1) +                   # Hamilton-Operat.
         np.diag((-z*np.exp(-1j*k_bloch))*np.ones(1), k=N-1) +
         np.diag((-z*np.exp(-1j*k_bloch))*np.ones(1), k=-(N-1)))

    ew, ef = eigh(h)                                       # Diagonalisierung
    ef = ef/np.sqrt(delta_x)                               # WS-Normierung
    return ew, ef



def diskretisierung(xmin, xmax, N, retstep=False):
    """übernommen aus quantenmechanik.py
    
    Berechne die quantenmechanisch korrekte Ortsdiskretisierung.

    Parameter:
        xmin: unteres Ende des Bereiches
        xmax: oberes Ende des Bereiches
        N: Anzahl der Diskretisierungspunkte
        retstep: entscheidet, ob Schrittweite zurueckgegeben wird
    Rueckgabe:
        x: Array mit diskretisierten Ortspunkten
        delta_x (nur wenn `retstep` True ist): Ortsgitterabstand
    """
    delta_x = (xmax - xmin) / (N + 1)                      # Ortsgitterabstand
    x = np.linspace(xmin+delta_x, xmax-delta_x, N)         # Ortsgitterpunkte

    if retstep:
        return x, delta_x
    else:
        return x


def wenn_maus_geklickt(event, ax1, ax2, L, N, h_eff, V, Emax, skalierung):
    """Funktion zur Darstellung der Eigenfunktionen des durch Mausklick 
    gewählten Wertes für die Bloch Phase.
    
    Parameter:
    ax1,ax2: Plotbereiche
    L: Begrenzungsparameter
    N: Anzahl der Diskretisierungspunkte
    h_eff: effektives hquer
    V: Potential
    Emax: maximale Energie
    skalierung: Faktor zur skalierung der EFs
    """
    # Test, ob Klick mit linker Maustaste und im ax1 bzw. linken 
    #Koordinatensystem erfolgt sowie ob Zoomfunktion des Plotfensters 
    #deaktiviert ist
    mode = event.canvas.toolbar.mode
    if event.button == 1 and event.inaxes == ax1 and mode == '':
        ax2.lines=[] #Löschen der vorherigen EFs
        #Festlegung k-Wert durch Maus-Position
        k_mouse = event.xdata
        #Doppelschleife zum Plot des Potentials und der EFs
        for i in range(4):
            x, delta_x = diskretisierung(i*L, (i+1)*L, N, True)
            v_werte = V(x)
            ax2.plot(x, v_werte, color='k')
        
            EW, EV = diagonalisierung(h_eff, x, V, k_mouse)
        
            anz = np.sum(EW <= Emax)
            #feste Farbreihenfolge
            colors = ['b', 'g', 'r', 'c', 'm', 'y'] 
            for j in range(anz):
                ax2.plot(x, EW[j] + skalierung*np.abs(EV[:, j])**2, 
                color=colors[j % len(colors)])
        event.canvas.draw()


def main():
    """Hauptprogramm"""
    print(__doc__)
    
    L = 1                       #Begrenzungsparamter
    
    h_eff = 0.2
    A = 1                       #gegeb. Parameter des Potentials
    
    Emax = 7.0                  #max. Energie der Eigenwerte
    
    N = 200                     #Matrixgröße
    
    step_k = 100                #Anzahl Schritte bei k Variation
    
    skalierung = 0.2            #Skalierungsfaktor für EFs
    
    #Berechnugn der diskreten Ortswerte und des Ortsgitterabstand 
    x, delta_x = diskretisierung(0, L, N, True)
    #Definition des gegeben periodeischen Potentials
    V = functools.partial(potential_DM, A=A)
    
    #Array der k Werte für Variation
    k_bloch_var = np.linspace(-np.pi, np.pi, step_k, endpoint=True)
    #Schleife zur Bestimmung der Anzahl der Eigenwerte kleiner des gewählten 
    #Emax
    #Dabei wird der größte Anzahl an Eigenenergien kleiner des gewählten Emax
    #für alle k bestimmt. Dies ist zur späteren Abspeicherung der Werte in 
    #einem assymmetrischen Array notwendig.
    #Dabei werden einige zu große Eigenwerte später mit abgespeichert, dies ist
    #aber aufgrund der nötigen Definition der shape des assymm. 
    #Arrays notwendig   
    anz_max = 0
    for i in range(step_k):
        #Berechnung des Eigenwerte und -funktionen
        EW, EV = diagonalisierung(h_eff, x, V, k_bloch_var[i])
        #Anzahl der EW < Emax
        anz = np.sum(EW < Emax)
        #max Anzahl der EW<Emax für alle k 
        anz_max = max(anz, np.sum(EW < Emax), anz_max)
    
    #Erstellung asymm. Array zuer Speicherung der EW
    EW_bloch_var = np.zeros([step_k,anz_max])
    #Schleife zur Bereung und Speicherung der EWs
    for i in range(step_k):    
        EW, EV = diagonalisierung(h_eff, x, V, k_bloch_var[i])
        
        EW_bloch_var[i] = EW[0:anz_max:]

    #Ertstellung Plotfenster,Anpassungen und Beschriftung 
    fig = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    
    ax1.set_xlim([-np.pi, np.pi])
    ax1.set_ylim([-1.5, 7.0]) 
    
    ax2.set_xlim([0.0, 4.0])
    ax2.set_ylim([-1.5, 7.0])
    
    ax1.set_title('Eigenwertspektrum ')
    ax1.set_xlabel('Bloch-Phase $k$')
    ax1.set_ylabel('Eigenenergie $E$')
    
    ax2.set_title('Eigenfunktionen und Potential')
    ax2.set_xlabel('Periode')
    ax2.set_ylabel('Eigenenergie $E$')
    
    #Plot des Eigenwertspektrums 
    ax1.plot(k_bloch_var, EW_bloch_var , ls='None', marker ='.', ms=3)
    
    #Interaktion im Plot mithilfe der Funktion wenn_maus_geklickt
    click_function = functools.partial(wenn_maus_geklickt, ax1=ax1, ax2=ax2, 
                                       L=L, N=N, h_eff=h_eff, V=V, Emax=Emax, 
                                       skalierung=skalierung)
    plt.connect('button_press_event', click_function)
    
    
    plt.show()
    

if __name__ == "__main__":
    main()

#Struktur des Eigenenergiespektrums:
#Es zeigt sich das bis Emax=7 sich 6 bzw. 5 Eigenenergien auftretten.
#Dabei sind für Werte von k um 0 nur 5 Eigenenergien kleiner 7 vorhanden.
#Dies kommt aufgrund der entstehenden Bandtsruktur des Energiespektrums 
#zustande. Die Eigenenergien in Abhängigkeit von k sind Acshensymmetrisch um 
#k=0 und besitzen Extrempunkte bei k=0. An den Rändern der Darstellung bzw. bei
#k=+-Pi treffen sich die Funktionen für die dritte und vierte und fünfte und 
#sechste Funktion der Eigenwerte.
#Außerdem sind zeigt sich, dass für k=0 sich die vierte und fünfte Funktion der 
#Eigenwerte treffen.

#Struktur der Eigenfunktionen:
#Zuerst ist zu bemerken, dass die Eigenfunktionen innerhalb jeder periode 
#identisch sind. Im Allgemeinen stellen die Eigenfunktionen (periodische) 
#Schwingungen dar.
#Für k=0 überlagern scheinen die vierte und fünfte Eigenfunktion complemetär 
#zueinander. Analog dazu sin bei k=+-pi jeweils die dritte und vierte und 
#fünfte und sechse EF complementär zueinander.

#sehr schwaches Potential:
#In zu betrachtedem fall bedeutet ein schwaches Potential nur eine kleine 
#Amplitude des Cosinuses des Potentials. Auch die Eigenfunktionen stellen auf 
#ersten Blick keine Schwingungen mehr da, da Auch ihre Amplituden stark
#verringert sind. Für k=0 sind minimale vergrößerungen der Amplitude der dritten
#vierten EF zu beobachten. Für k=+-Pi sind dann bei der zweiten und dritten EF 
#eindeutige Schwingungen mit deutlich größeren Amplituden zu erkennen.    
#Die Eigenwerte sind bei schwachen Potentialen unverändert. 
 

