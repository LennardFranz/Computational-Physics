"""Darstellung des Phasenraumes der Standardabblindung für 
periodische Randbedingungen
"""
import functools
import numpy as np
import matplotlib.pyplot as plt             #Importe der benötigten Module

def s_abbildung (theta ,p, K, N):
    """ Funktion zur rekursiven Berechnung der gewählten Werte für Phasenwinkel
    theta ind Impuls p der Standardabbildung. 
    
    K: wählbarer Parameter innerhlab der Standardabbildung
    
    N: Anzahl der Iterationen"""

    a = np.zeros(N+1)       #Erstellung des Phasenwinkelarray für 1000 
                            #Iterationen und Anfangswert (Arraylänge=1001) 
    
    b = np.zeros(N+1)       #Erstellung des Impulsarray analog zum 
                            #Phasenwinkelarray
    
    a[0] = theta            #Setzen der Anfangswerte
    b[0] = p 
    
    for i in range(1,N+1):                 #Schleife zur Berechnung 
                                           #von Phi und p
        theta = theta + p
        p = p + K*np.sin(theta)
        
        a[i] = theta                   #Füllen der Arrays mit berechneten
        b[i] = p                       #Werten für p und theta  
 
    
    a = a % (2.0*np.pi) 
    b = (b + np.pi) % (2.0*np.pi) - np.pi   #Modulo Operation zur Realisierung 
                                            #der Randbed. 
    
    return a,b                              #Rückgabe der Arrays




def wenn_maus_geklickt(event,ax, K, N):
    """Verwendung der gegebenen Funktion"""
    # Test, ob Klick mit linker Maustaste und im Koordinatensystem
    # erfolgt sowie ob Zoomfunktion des Plotfensters deaktiviert ist:
    mode = event.canvas.toolbar.mode
    if event.button == 1 and event.inaxes and mode == '':
        x, y = s_abbildung(event.xdata, event.ydata, K, N)
        
        ax.plot(x, y, linestyle='None', marker= '.', markersize=3)
        # Fensterbereich aktualisieren
        event.canvas.draw()
        #Dieser Block wurde dem Aufgabenblatt entnommen 
        

def main():
    """Hauptprogramm: Initialisierung Plotfenster + Def. Mausinteraktion."""
    K = 2.6    #gegebener Parameter
    N = 1000   #gegebene Anzahl an Iterationen 

    # Nutzerfuehrung und Parameterausgabe
    print(__doc__)
    print("Mit Linker Maustaste Anfangswerte für Theta und p festlegen.")
    print("Es wird der Phasenraum für 1000 Iterationen und K=2.6 dargestellt.")

    # Erzeuge Plotfesnter 
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Achsenbereiche setzen
    ax.set_xlim([0, 2*np.pi])
    ax.set_ylim([-np.pi, np.pi])

    # Plotbeschriftungen setzen
    ax.set_xlabel("$\Theta$")
    ax.set_ylabel("p")
    ax.set_title("Standardabbildung für K=2.6")
    plt.xticks( [0,np.pi, 2*np.pi],
            [ r'$0$',r'$\pi$', r'$2\pi$'])
    plt.yticks( [-np.pi, 0,np.pi],
            [r'$-\pi$',r'$0$',r'$\pi$']) #Achsenenteilung und -beschriftung
            
    #Bei Mausklick werden an Funktion s_abbildung Startwerte 
    #für Phi und p, sowie der Plotbereich ax übergeben und geplottet.  
    klick_funktion = functools.partial(wenn_maus_geklickt, ax=ax, K=K, N=N) 
                                       
    fig.canvas.mpl_connect('button_press_event', klick_funktion)
    
    # Endlos-Schleife, die auf Ereignisse wartet:
    plt.show()
    

if __name__ == "__main__":
    main()
   
#Bei K=0 ergeben sich horizontale Linien.

#Bei K>0 sind reguläre elliptische Fixpunkte erkennbar.

#Im Bereich 0<K<1 sind neben einigen gekrümmten Linien chaotische 
#Bereiche mit regulären elliptischen Inseln zu beobachten.
 
#Ab K>1 und mit steigendem K werden die chaotischen Bereiche immer größer und 
#die regulären Inseln immer seltener und kleiner.

# Ab ca. K=4.5 beginnt sich der zentralle Bereich aus elliptischen Fixpunkten
# in zwei bereiche aufzuspalten 

#Bei K=6 sind nur noch chaotishe Bereiche erkennbar.

#Bei K=6.6 können wieder Bereiche mit Fixpunkten gefunden werden.


 

 
