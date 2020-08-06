"""Angetriebenes Doppelmuldenpotential

Untersuchung der Dynamik eines Teilches im angetriebenen Doppelmuldenpotential.
Die entsprechende Hamiltonfunktion lautet dabei:
H(x,p,t)= p^2/2 + x^4-x^2+x*(A+Bsin(omega*t))
mit:
A=0.1
B=0.1
omega=1.0

Dargestellt werden die Trajektorien im Phasenraum und die stroboskopische
Darstellung. Dabei werden die Startwerte für die Berechnung der Dynamik
mithilfe eines Klicks der linken Maustaste im Plottbereich festgelegt.

Außerdem sind in beiden Darstellungen Konturlinen H für geeignet gewählte
Energien geplottet.
"""
import functools
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint          # Importe


def ableitung(y ,t , A , B, omega):
    """Funktion zur Berechnung der zeitlichen Ableitungen der gegebenen 
    Hamiltionfunktion mithilfe der kanonischen Gleichungen. Dafür wird
    ein 2D-Array y, ein Array der Zeitwerte t und drei Parameter A,B und
    omega verwendet."""
    #zeitliche Ableitung aus kanonischen Gleichungen 
    return np.array([y[1], -4.0*y[0]**3.0+2.0*y[0]- A-B*np.sin(omega*t)])
     

def hamilton_fkt(x, p, A, B, omega, t):
    """Funktion zur Berechnung der gegebenen Hamiltonfunktion."""
    return (p**2.0)/2.0 + x**4.0 - x**2.0 + x*(A + B*np.sin(omega*t))

def wenn_maus_geklickt(event, A, B, omega, t, phasenplot, stroboplot, 
                stepsize_strobo ):
    """Funktion zur Berechnung und Plotten der Trajektorien im Phasenraum und
    der stroboskopischen Darstellung. Dabei wird mit der linken Maustaste
    im Plotbereich der Startwert der zu berechnenden Trajektorien 
    festgelegt."""
    # Test, ob Klick mit linker Maustaste und im Koordinatensystem
    # erfolgt sowie ob Zoomfunktion des Plotfensters deaktiviert ist:
    mode = event.canvas.toolbar.mode
    if event.button == 1 and event.inaxes and mode == '':
        #Festlegung Startwerte
        y_0 = np.array([event.xdata, event.ydata])
        
        #Lösung der bestimmten Differentialgleichung
        y = odeint(ableitung, y_0, t, args=(A, B, omega))
        #Aufspalten des 2D-Arrays in zwei 1D-Arrays
        x = y[:, 0]        #Ortsarray                           
        p = y[:, 1]        #Impulsarray

        #Berechnung benötigter Indices für stroboskopische Darstellung
        #Dabei muss die Schrittweite gleich der verwendeten Anzahl an
        #Iterationen gewählt werden. 
        ind = np.arange(0, len(y), stepsize_strobo)
        #Anwenden der bestimmten Indices auf Orts/Impuls-Arrays für
        #stroboskopische Darstellung 
        x_strob = x[ind]
        p_strob = p[ind]
        
        #Plotten der Trajektorien im Phasenraum und der stroboskopische
        #Darstellung
        phasenplot.plot(x, p, lw=2) 
        stroboplot.plot(x_strob, p_strob, ls='None', marker='.', ms=2) 
        event.canvas.draw()

def main():
    """Hauptprogramm
    -Erstellung und Berechnung des Zeitenarrays
    -Berechnungen für Konturplot
    -Initialisierung Plotfenster 
    -Definition Mausinteraktion 
    """
    #Nutzerführung
    print(__doc__)
    
    #Festlegung der gegebenen Parameter 
    A = 0.1
    B = 0.1
    omega = 1.0
   
    anzahl_iterationen = 1000    
    
    #Zeitberechnung 
    anzahl_perioden = 200
    T = 2*np.pi/omega
    t = np.linspace(0.0, anzahl_perioden*T, 
                    anzahl_perioden*anzahl_iterationen)
                   
    #Festlegung Plottgrenzen
    x_limit = 2.0
    p_limit = 2.0
    
    #Energien für Konturlinien
    energien_contour = [-0.5, -0.2, -0.1, 0.0025, 0.2, 0.5, 1.0, 1.5] 

    #Orts- und Impulsarray für Konturplot 
    x_H= np.linspace(-x_limit, x_limit, anzahl_iterationen)
    p_H= np.linspace(-p_limit, p_limit, anzahl_iterationen)

    #Verwendung von Meshgrid zur Erzeugung 2D-Arrays für np.contour 
    x_mesh, p_mesh = np.meshgrid(x_H, p_H)

    #Berechnung der Energie mit Hamiltonfkt.
    H = hamilton_fkt(x_mesh, p_mesh, A, B, omega, 0)

    #Erstellung Plottfenster 
    fig = plt.figure(figsize=(10, 6))
    #Erstellung zwei Subplots
    phasenplot = plt.subplot(121)
    stroboplot = plt.subplot(122)
    
    #Plottbeschriftung und Achsenbereich 
    phasenplot.set_title('Phasenraum')
    phasenplot.set_xlabel('Ort x')
    phasenplot.set_ylabel('Impuls p')
    phasenplot.set_xlim([-x_limit, x_limit])
    phasenplot.set_ylim([-p_limit, p_limit])
    
    #Plottbeschriftung und Achsenbereich 
    stroboplot.set_title('stroboskopische Darstelllung des Phasenraums')
    stroboplot.set_xlabel('Ort x')
    stroboplot.set_ylabel('Impuls p')
    stroboplot.set_xlim([-x_limit, x_limit])
    stroboplot.set_ylim([-p_limit, p_limit])
    
    #Plotten der Contourplots
    phasenplot.contour(x_mesh, p_mesh, H, energien_contour, colors='k')
    stroboplot.contour(x_mesh, p_mesh, H, energien_contour, colors='k')
    
    #Interaktion im Plot mithilfe der Funktion wenn_maus_geklickt
    click_function = functools.partial(wenn_maus_geklickt, omega=omega, A=A, 
                                       B=B, t=t,
                                       phasenplot=phasenplot,
                                       stroboplot=stroboplot,
                                       stepsize_strobo=anzahl_iterationen)
    plt.connect('button_press_event', click_function)
    
    #Endlos-Schleife, die auf Ereignisse wartet:
    plt.show()

if __name__ == "__main__":
    main()

"""


a)
Für B=0.0 in der Hamiltonfunktion ergibt sich ein Doppelmuldenpotetial mit 
zwei Minima  verschiedener Tiefe.
Für Energien E<0 ergeben sich im Phasenraum abhängig von der Ortskomponente
(negative Ortskomponente = Minima_1 positive Ortskomponente = Minima_2)
Ellipsen, was im Ortsraum Schwingungen um die jeweiligen Minima entspricht.

Für E~0 entsteht im Phasenraum die sog. Separatrix, welche ungefähr einer
nach rechts gedrehten 8 entspricht. Im Ortsraum entpricht dies dann 
Schwingungen um eines der beiden Minima. Aufgrund des instabilen lokalen
Maximums zwischen beiden Minimas lässt sich nicht eindeutig sagen um 
welches Minima sich die Schwingung ausbildet.
Außerdem trennt die Seperatrix den Bereich der Schwingung um nur ein Minimum
von dem der Schwingungen über beide Minima.

Für E>0 ergeben sich dann im Phasenraum Kurven, welche die Sepersatrix
umhüllen (Erdnussförmig). Für immer größer werdende Energien ergeben sich
dann Ellipsen um die Seperatrix mit abgeflachten Spitzen oben und unten.
Im Ortsraum entspricht dies dann Schwingungen durch beide Minima.


b)
Für B=0.1 ergibt sich nun ein zeitlich veränderliches Doppelmuldenpotential.
Für kleine Energien (im Verhältnis zu der Tiefe der Minima) ergeben sich
wie für B=0.0 Ellipsen im Phasenraum, also Schwingungen um die jeweiligen
Minima. Dabei sind jedoch nicht wie für B=0.0 die Ellipsen scharfe Linien,
sondern es enstehen eher ellipsenförmige Ringe.
Für größere Energien sind dann die Trajektorien aufgrund chaotischer Dynamik
keine Kurven, sondern Bereiche. Dies zeigt auch die stroboskopische
Darstellung. 

Bei wieder größer werdenden Energien ergben sich dann wieder 
analog zu Aufgabe a) Ellipsen, die Form Seperatrix umhüllende Kurven. 
Jedoch sind diese auch wie die Ellipsen für kleine Energien nicht scharfe
Linien sondern besitzen eine gewisse Breite.

Außerdem sind noch einzelne Inseln (stroboskopische Darstellung) 
periodischen Verhaltens in Bereichen chatotischer Dynamik zu beobachten. 

c)

Startkoordinaten:
x = 0.719 (Ort)
p = -0.032 (Impuls)

Anzahl Punkte: 15

Periode: 94.248

"""




















