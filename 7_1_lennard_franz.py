"""Quantenmechanik in 1-D Potentialen: Zeitentwicklung im 
                                       Doppelmuldenpotential
Programm zur dynamaischen Zeitentwicklung eines Gaussschen Wellenpakets
in eimem asymmetrischen Doppelmuldenpotential
Dafür wird bei Mausklick mit der linken Maustaste der Anfangsort für die
Zeitentwicklung festgelegt.
Die gewählten Anfangsorte werden mit der Differnz der Norm für jede 
Zeitentwicklung eines Wellenpakets ausgegeben.

Das Plotfenster zeigt die Darstellung der Eigenfunktionen im asymmetrischen 
Doppelmuldenpotentials mithilfe des Programmes quantenmechanik.py.

Verwendete Parameter:

A = 0.06 (Parameter in der Formel des Doppelmuldenpotential bzw. 
          Asymmetrie)

p_0 = 0.0 (Anfangswert Impuls)

D_x = 0.1 (Breite des Wellenpakets)
"""
import numpy as np
import matplotlib.pyplot as plt
import functools
import quantenmechanik as qm               #Importe


def Gauss(x, D_x, x_0, h_eff, p_0):
    """Definition des gegebenen Gaussschen Wellenpakets.
    x:   Array diskretisierter Ortspunkte
    x_0: Anfangsort/Startpunkt
    p_0: Anfangsimpuls/Startimpuls
    D_x: Breite des Gausspakets
    h_eff: effektives Planksches Wirkungsquantum
    """
    return 1/(2*np.pi * D_x ** 2)**(1/4) * np.exp(-(x-x_0) ** 2 /
          (4 * D_x ** 2)) * np.exp(1j/h_eff * p_0 * x)
                                                  
def potential_DM(x, A):
    """Funktion zur Berechnung des asymetrisches Doppelmuldenpotentials
    für gegebene Werte x (Array) und Parameter A.
    """
    return x**4 - x**2 - A*x
    
def berechnung_koef_entwicklung(delta_x, EV, phi):
    """Funktion zur Berechnung der Etwicklungsfunktionen laut Vorlesung.
    Die Numpy-Funktionn conjugate und transpose werden verwendet um die
     benötigte komplexe Konjugation zu realisieren. 
    """
    return delta_x*np.dot(np.conjugate(np.transpose(EV)), phi)

def entwicklung_wellenpaket(C, EW, EV, t, h_eff):
    """Funktion zur Zeitentwicklung eines Wellenpakets für gegebenes t 
    laut Vorlesung.
    Dabei sind C,EV,EW als Arrays und t und h_eff als Floats.
    j stellt die imaginäre Einehit da.
    C: Entwicklungskoef.
    EV: Eigenvektoren bzw. -funktionen
    EW: Eigenwerte bzw. -energien
    """
    return np.dot(EV, C*np.exp(-1j*EW*t/h_eff))
        
def energieerwartungswerte(C, EW):
    """Funktion zur Berechung der Energieeigenwerte laut Vorlesung.
    EW: Energieeigenwerte als Array
    
    """
    return np.dot(np.abs(C)**2, EW)     
    
def wenn_maus_geklickt(event, ax, x, D_x, h_eff, p_0, delta_x, EW, EV, 
                       skalierung, T):
    """Funktion zur grafischen Darstellung des Betragsquadrat des 
    zeitentwickelten Wellenpakets.
    Dabei legt die x-Position des Klicks den Anfangsorts der Zeitentwicklung
    fest. Außerdem wird die Differenz der Norm für jedes Anfangswellenpaket
    ausgegeben.
    """
    mode = plt.get_current_fig_manager().toolbar.mode
    if mode == '' and event.inaxes and event.button == 1:
        #Anfangsort festlegen 
        x_0 = event.xdata
        #Berechnung des Wellenpakets für definierten Anfangswert x_0
        phi = Gauss(x, D_x, x_0, h_eff, p_0)
        #Berechnung der Entwicklungskoef.
        c_n = berechnung_koef_entwicklung(delta_x, EV, phi)
        #Berechnung der Energieerwartungswerte
        E_eigen = energieerwartungswerte(c_n, EW)
        #Berechnung der Differenz ursprüngliches und rekonstruiertes Wellenpaket
        differenz = phi - entwicklung_wellenpaket(c_n, EW, EV, 0, h_eff)
        #Berechnung der Differenz der Norm und Ausgabe 
        norm = np.sqrt(np.dot(differenz, differenz)*delta_x)
        print('Differenz der Norm:', norm.real)
        
        #Darstellung des Plots des Betragsquadrats 
        P = ax.plot(x, E_eigen + skalierung*np.abs(entwicklung_wellenpaket(
                                                     c_n, EW, EV, 0, h_eff))**2)
        #Darstellung des Betragsquadrats für Zeitentwicklung
        for t in T:
            P_t = E_eigen + skalierung*(np.abs(entwicklung_wellenpaket(c_n, 
                                           EW, EV, t, h_eff))**2)
            P[0].set_ydata(P_t)
            event.canvas.flush_events()
            event.canvas.draw()

def main():
    """Hauptprogramm"""
    print(__doc__)
    #Paramter für quantenmechanik.py
    L = 1.5                                       #Intervallgrenzen
    N = 300                                       #Matrixgröße
    h_eff = 0.07                                  #effektives Plack.-
                                                  #Wirkungsquantum
    p_0 = 0.0                                     #Anfangsmpuls
    A = 0.06                                      #gegebener Paramter
                                                  #im Doppelmuldenpot.
    D_x = 0.1                                     #Breite des Wellenpaket
    
    
    t_max = 100                                  #maximale Zeit für Zeitarray
    T = np.linspace(0, t_max, 100)                #Zeitenarray
    
    skalierung = 0.01                             #Skalierung zur besseren
                                                  #Darstellung 
    #Berechnugn der diskreten Ortswerte und des Ortsgitterabstand mithilfe
    #quantenmechanik.py  
    x, delta_x = qm.diskretisierung(-L, L, N, True)
    #Definition des Doppelmuldenpotentials 
    V = functools.partial(potential_DM, A=A)
    #Berechnung des Eigenwerte und -funktionen mithilfe quantenmechanik.py 
    EW, EV = qm.diagonalisierung(h_eff, x, V)
    #Erstellung des Plotfensters  
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)
    #Plot der Eigenfunktionen durch quantenmechanik.py
    qm.plot_eigenfunktionen(ax, EW, EV, x, V, betragsquadrat=True, 
                            fak=skalierung)
    #Übergabe der Parameter an wenn_maus_geklickt
    klick_funktion = functools.partial(wenn_maus_geklickt,ax=ax, x=x, D_x=D_x, 
                                       h_eff=h_eff, p_0=p_0, delta_x=delta_x, 
                                       EW=EW, EV=EV, skalierung=skalierung,
                                       T=T)
    fig.canvas.mpl_connect("button_press_event", klick_funktion)
    plt.show()
    
if __name__ == "__main__":
    main()
"""
a)
Beim Start eines Wellenpaktes im einem der beiden Minima bzw. Mulden des 
Doppelmuldenpotentials könne Bewegungen um die Maxima der nächsten 
Eigenfunktionen (bei der tieferen Mulde die Eigenfunktion zum kleinsten
Eigenwert und bei der höheren Mulde die Eigenfunktion zum zweitkleinsten
Eigenwert) beoebachtet werden. Das Betragsquadart der Wellenpakte kommt
einer Gausverteilung mit Maximum im jeweiligen Minimum sehr nahe.

Beim Start eines Wellenpakets im Maximum des Doppelmulden Potentials können
Bewegungen ähnlich der Eigenfunktionen des Eigenwerts bei ca. 0.05
beobachtet werden. Somit können Schwingungen über den gesamten Bereich des
Doppelmuldenpotentials beobachtet werden.

b) p_0 = 0.3

Für p_0 = 0.3 ergeb sich im allgemeinen höhere Energieerwartungswerte, da 
das Betragsquadart nun deutlich höher geplottet wird.

Für Wellenpaktete mit Startpunkt im Minimum ergeben sich nun nicht mehr
Gaussverteilungen mit Maximum im Bereich des jeweiligen Minimums, sondern 
verbreitern und verformen die Maxima der Gaussverteilungen und es kommen
Schwingungen über den restlichen Bereich des Potentials hinzu. Außerdem 
enstehen insbesondere für Bewegungen beginnend im tieferen Minima des 
Potentials teilweise zwei (Haupt-)Maxima im Bereich des Minimums.

Für Bewegungen startend im Maximum des Potentials sind die Bewegungen 
analog zu a), jedoch mit dem schon erwähnten Größen Betragsquadrat, 
wodurch dieses nun fast der Eigenfunktion zum größten dargestellten 
Eigenwert entspricht.

c)
Für sehr große Zeiten im nun symmetrischen Doppelmuldenpotential entstehen
nun zwei Maxima des Betragsquadrates in den Bereichen der Minima, 
dabei ist das Maxima in dem Minima, in dem gestartet wurde, größer.
Diese beobachtung lässt sich als Tunneleffekt interpretieren.
"""
