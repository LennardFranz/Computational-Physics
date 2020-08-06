"""Programm zur Simulation einer gerichteten Diffusion mit absorbierendem Rand 

   Die Zeitentwicklung findet mithilfe der Langevin-Gleichung statt.
   Es werden R=10000 Realisierungen über ein Zeitintervall von 0 bis 40, 
   Anfangsort für t=0 x=0, Dirftgeschw. v=0.1, Diffusionskonst D =1.5 und dem
   abs. Rand bei x_abs=15 betrachtet. 
   
   Durch Klicken wird die Simulation gestartet, dabei werden für jede 
   Zeiteinheit das Histogramm, die Wahrscheinlichkeitsdichte, Norm, 
   Erwartungswert und Varianz berechnet und dargestellt (schwarz).
   
   Ausserdem werden in rot die theoretischen Werte ohne abs. Rand dargestllt. 

"""
import numpy as np
import matplotlib.pyplot as plt
import functools                  #Importe

def normalverteilung(x, m, v):
    """Funktion zur Berechnung der Normalverteilung
    x: Wertearray
    m: Mittelwert der Verteilung
    v: Varianz der Verteilung
    """
    return 1.0/np.sqrt(2.0*np.pi*v)+np.exp(-(x-m)**2/(2.0*v))
    
def langevin(R, x_t, v, delta_t, D):
    """Funktion zur Berechnung der Langevin-Gleichung
    
    R: Realisierungen 
    x_t: Wertearray
    v: driftgeschw.
    delta_t: zu entwickelder Zeitschritt
    D:diffusionskonst.
    """
    #weißes Rauschen bzw. Normalverteilung mit EW=0 und Varianz 1
    #zur Vollständigkeit: sigma * np.random.randn() + Erwartungswert 
    eta_t = 1.0 * np.random.randn(R) + 0.0
    
    return x_t + v * delta_t + np.sqrt(2.0*D*delta_t) * eta_t

def Wahrscheinlichkeitsdichte(x, x_0, x_abs, v, t, D):
    """Funktion zur Berechnung der Whs.dichte laut Vorlesung
    
    x: Wertearray
    x_0: Startwerte des Wertearray
    x_abs: Position des abs. Randes
    v: Driftgeschw.
    t: Zeit
    D: Diffusionskonstante
    """
    return normalverteilung(x, x_0+v*t, 2.0*D*t) - \
           normalverteilung(x, 2.0*x_abs-x_0+v*t, 2.0*D*t) * \
           normalverteilung(x_abs, x_0+v*t, 2*D*t)/ \
           normalverteilung(x_abs, 2.0*x_abs-x_0+v*t, 2.0*D*t)


def wenn_maus_geklickt(event, R, x_0, x_abs, T_max, delta_t, v, D, S, ax1, ax2,
                       ax3, ax4):
    """Funktion zur dynamischen Darstellugn der gerichteten Diffusion und 
    Berechnung aller geforderten Parameter.
            
    R, x_0, x_abs, delta_t,v,D siehe andere Funktionen
    T_max: Maximalwert der Zeitent.
    S: Schrittanzahl"""
    mode = plt.get_current_fig_manager().toolbar.mode
    if mode == '' and event.inaxes and event.button == 1:
        
        x = np.ones(R) * x_0
        #def Zeitenarray
        T = np.linspace(0, T_max, T_max/delta_t + 1)
        
        P = ax1.plot(x, Wahrscheinlichkeitsdichte(x, x_0, x_abs, v, 1.0, D),
                    color = 'k')
        
        P_no_abs = ax1.plot(x, normalverteilung(x, x_0+v*1.0, 2.0*D*1.0), 
                            color='r')
         
        R_0 = R
        #setzen der bins für Histogramm
        bins = np.linspace(-40, x_abs, 100)
        
        #Arrays für benötigte Parameter
        #länge 41 ergibt sich aus Naforderung an t_n
        norm = np.zeros(41)
        erwartungswert = np.zeros(41)
        varianz = np.zeros(41)
        t_n = np.zeros(41)
        
        
        N = ax2.plot(t_n, norm, color = 'k')
        EW = ax3.plot(t_n, norm, color = 'k')
        V = ax4.plot(t_n, norm, color = 'k')
        #Array zum Plot der theor. Werte
        t_theo = np.linspace(0, 40, 41)
        
        #da ohne Absorption immer R_t=R gilt bleibt die Norm konstant 1 
        N_no_abs = ax2.plot(t_theo, np.ones_like(t_theo), color='r')
        #laut Vorlesung
        EW_no_abs = ax3.plot(t_theo, x_0 + v*t_theo, color='r')
        V_no_abs = ax4.plot(t_theo, 2*D*t_theo, color='r')
        
        #Definition der 0.ten elemente der Arrays für Parameter laut Vorlesung
        #bzw. per Definition 
        norm[0] = 1.0
        erwartungswert[0] = x_0
        varianz[0] = 0.0
        
        #Schleife für Zeitentwicklung 
        for t in T:
            
            x = langevin(R, x, v, delta_t, D)
            #def array für plot der Whs.dichte theor. Wert
            x_whs_no_abs = np.linspace(min(x), max(x), R)
            #Realisierung des abs. Randes
            x = x[x < x_abs]
            R = len(x)
            #def array für plot der Whs.dichte
            x_whs = np.linspace(min(x), max(x), R)
            #Schleife zur Berechnung der Paramter zu Zeiten t_n
            if t%1/S==0 and t>0:
                #Löschen der vorherigen patches des Histogramms
                ax1.patches = []
                
                P_t_n = Wahrscheinlichkeitsdichte(x_whs, x_0, x_abs, v, t, D)
                P_no_abs_t_n = normalverteilung(x_whs_no_abs, x_0 + v*t, 2*D*t)
                #Aktualisieren der Plots
                P[0].set_xdata(x_whs)
                P[0].set_ydata(P_t_n)
                
                P_no_abs[0].set_xdata(x_whs_no_abs)
                P_no_abs[0].set_ydata(P_no_abs_t_n)
                #berechnugn der weights für plt.hist
                weights = np.ones(R)/(R)
                #darstellung Histogramm
                ax1.hist(x, bins=bins, weights=weights, color = 'g')
        
                #Berechnung der benötigten Parameter
                norm[int(t)]= R/R_0
                erwartungswert[int(t)] = np.mean(x)
                varianz[int(t)] = np.var(x)
                t_n[int(t)] = t
                
                #Aktualisierung der Plots
                N[0].set_xdata(t_n)
                N[0].set_ydata(norm)
                
                EW[0].set_xdata(t_n)
                EW[0].set_ydata(erwartungswert)
                
                V[0].set_xdata(t_n)
                V[0].set_ydata(varianz)
                
                event.canvas.flush_events()
                event.canvas.draw()
                
def main():
    """Hauptprogramm"""
    
    print(__doc__)
    #Definition  benötigte Paramter (siehe doc Funktionen)
    x_0 = 0.0       
    T_max = 40    
    x_abs = 15
    v = 0.1
    D = 1.5
    R = 10000    
    delta_t = 0.01 
    S = 100
    
    #Initialisierung Plotfenster
    fig = plt.figure(figsize=(10, 8))
    
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    
    ax1.set_xlim([-40.0, 15.0])
    ax1.set_ylim([0.0, 1.1])
    ax1.set_xlabel('Ort $x$')
    ax1.set_ylabel('$P(x,t)$')
    
    ax2.set_xlim([0.0, 40.0])
    ax2.set_ylim([0.8, 1.1])
    ax2.set_xlabel('$t_n$')
    ax2.set_ylabel('Norm $R(t_n)/R$')
    
    ax3.set_xlim([0.0, 40.0])
    ax3.set_ylim([-2.0, 4.0])
    ax3.set_xlabel('$t_n$')
    ax3.set_ylabel('Erwartungswert $m$')
    
    ax4.set_xlim([0.0, 40.0])
    ax4.set_ylim([0.0, 80.0])
    ax4.set_xlabel('$t_n$')
    ax4.set_ylabel('Varianz $\sigma^2$')    
    
    
    
    klick_funktion = functools.partial(wenn_maus_geklickt, R=R, x_0=x_0, 
                                       x_abs=x_abs, T_max=T_max, delta_t=delta_t
                                       ,  v=v, D=D, S=S, ax1=ax1, ax2=ax2, 
                                       ax3=ax3, ax4=ax4)
    
    fig.canvas.mpl_connect("button_press_event", klick_funktion)
    
    plt.show()
    
if __name__ == "__main__":
    main()



#Diskussion
#a)
#Der Einfluss des absorbierenden Randes wirkt sich auf alle betrachteten 
#Parameter aus. ohne Absorbierenden Rand bleibt die Teilchenzahl konstant, 
#wodurch die Norm konstant 1 bleibt und nicht sinkt.
#Weiterhin bewirkt der absorbierende Rand einen Verschiebung des 
#Erwartungswerts in den negativen Bereich, im Gegensatz zum Erwartungswert 0 
#ohne abs. Rand. Die Varianz im allgemeinen kleiner mit abs. Rand. Dies liegt 
#an der fehlenden Möglichkeit zur Ausdehnung in positive Richtung der Teilchen.
#Der Einfluss auf die Parameter wird nach 10 bis 15 Zeiteinhieten deutlich
#erkennbar.

#b)
#Das Maximum der der Wahrscheinlichkeitsdichte mit abs. Rand verschiebt sich 
#minimal in Richtung 0 (Vergrößerung). 
      
#c)
#Die starke Vergrößerung der Driftgeschwindigkeit führt zu sehr viel 
#prägnanteren Abweichungen der betrachteten Parameter von theor. Vorhersagen.
#Für t>10 sind nur noch kleine Änderungen des Erwartungswerts und der Varianz 
#erkennbar. Anhand der norm ist erkannbar das sehr viel mehr Teilchen absorbiert
#werden.  

    
