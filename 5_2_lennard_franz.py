"""
Erweiterung des SIR Modells zum "SIRD" Modells durch unterscheidung zwschen 
Genesenenen und Gestorbenen.
dafür wird eine weitere Gruppe D und eine Lethalität (Wahrscheinlichkeit des
Versterbens) innerhlab des Differentialgleichungssystems hinzugefügt.  

Darstellung des zeitlichen Verlaufs der COVID-19 Pandemie mithilfe eines 
SIR-Modells.

Im oberen Plot werden die Anteile der drei Gruppen S (susceptible), I (infected)
und R (recovered) dargestellt. Dabei beschreiben die durchgezogenen Linien den 
Verlauf für die gegebenen Paramater beta = 0.5 (Übertragungswahrscheinlichkeit)
und gamma = 0.1.

Die unterbrochenen Linien stellen den Verlauf unter Berücksichtigung eines 
Lockdowns am Tag t_hammer = 20 mit einer Reduzierung auf beta_hammer = 0.3*beta
und Lockerungen am Tag t_dance mit einem Anstieg auf beta_dance = 0.5*beta.

Im unteren Plot ist der maximale Anteil an Infizierten in Abhängigkeit der 
Reproduktionsrate = beta/gamma dargestellt. Ebenensfalls stellt ist die gegebene 
Belatsungsgrenze des Gesundheitssystems von 10% dargestllt. Für die Variation 
der Reproduktionsrate wurde beta im bereich von 0.11 bis 0.5 variiert und gamma 
konstant gehalten. 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint  #Importe

def beta_lockdown(t, beta, t_hammer, t_dance):
    """ Funktion zur Umsetzung der Veränderung der 
    Übertragungswahrscheinlichkeit während eines Lockdowns.
    t: Zeitenarray
    
    beta: Basis Infektionswahrscheinlichkeit
    
    t_hammer und t_dance: Beginn und den Zeitpunkt der Lockerungen 
    innerhalb des Lockdowns
    """
    if t < t_hammer:
        return beta
    elif t > t_hammer - 1 and t < t_dance:
        return 0.3*beta
    else:
        return 0.5*beta

def diff_sys (f, t, beta, gamma, sigma):
    """Funktion zur Umsetzung des Differentialgleichungssystem des SIR-Modells. 
    f: Verktor mit Anfangswerten 
    t: Zeitenarray
    beta: Infektionswahrscheinlichkeit
    gamma: Genesungswahrscheinlichkeit
     
    """
    S, I, R, D = f
    dS = -beta * S * I
    # Unterscheidung Überlebende (recovered)und Gestrorbene (death) mit 
    #Lethalität sigma
    #gamma stellt nun die Genesungswahrscheinlichkeit dar. 
    dI = beta * S * I - gamma * I - sigma * I
    dR = gamma*I
    dD = sigma * I
    
    return dS, dI, dR, dD


def diff_sys_lockdown (f, t, beta, gamma, sigma, t_hammer, t_dance):
    """Funktion zur Umsetzung des Differentialgleichungssystem des SIR-Modells.
    Analog zur Funktion diff_sys, jedoch wird kein fest gewähltes beta verwendet
    , sondern zur Simulation eines Lockdowns die bereits definierte Funktion 
    beta_lockdown
    f: Verktor mit Anfangswerten 
    t: Zeitenarray
    beta: Infektionswahrscheinlichkeit
    gamma: Genesungswahrscheinlichkeit
    t_hammer und t_dance: Beginn und den Zeitpunkt der Lockerungen 
                          innerhalb des Lockdown
    """
    S, I, R, D = f
    dS = -beta_lockdown(t, beta, t_hammer, t_dance) * S * I
    dI = beta_lockdown(t, beta, t_hammer, t_dance) * S * I - gamma * I - sigma*I
    #analog diff_sys
    dR = gamma*I
    dD = sigma * I
    return dS, dI, dR, dD

def main():
    """Hauptprogramm """
    print(__doc__)
    
    N = 80000000   #Gesamtpopulation
    gamma = 0.1    #Genesungswahscheinlichkeit 
    beta = 0.5     #Infektionswahrscheinlichkeit
    
    sigma = 0.01   #Lethalität
    
    #Festlegung der Anfangswerte für Simulation
    #Es wird von einem Infizierten (I) und keinem Genesenem (R) zum Zeitpunkt 
    #0 Tage ausgegangen. Der Anzteil der Ansteckbaren (S) ergibt sich aus der 
    #Forderung S+I+R=1
    #S,I,R stellen Anteile und keine absoluten Zahlen dar!  
    I_0 = 1.0/N
    R_0 = 0.0/N
    
    #Anfangswert 0 Tote druch Pandemie
    D_0 = 0/N
    
    S_0 = 1.0 - I_0 - R_0 - D_0
    #Anfangsvektor  
    f_0 = S_0, I_0, R_0, D_0  
    #Erstellung des Anfangsvektors
    t = np.linspace(0, 300, 301)
    #Definition der Zeitpunkte des Lockdowns 
    t_hammer = 20 #Zeitpunkt des Start des Lockdowns
    t_dance = 60  #Zeitpunkt des Lockerung des Lockdowns
    
    #Lösung des gegebens DGL-Systems mithilfe odeint (ohne Lockdown)
    F = odeint(diff_sys, f_0, t, args=(beta, gamma, sigma))
    #Abspeichern der drei Lösungsfunktionen
    S, I, R, D = F.T     #Verwendung von .T zur transponierung der Lösungsmatrix
    
    #analoges Lösen des gegebens DGL-Systems mit Berücksichtigung des Lockdowns
    #zustätzlicher übergabe der Zeitüunkte des Lockdowns  
    F_lock = odeint(diff_sys_lockdown, f_0, t, args=(beta, gamma, sigma, 
                    t_hammer, t_dance))
    S_lock, I_lock, R_lock, D_Lock = F_lock.T #analog wie zuvor
    
    #Festlegung Iterationsparamter für Darstellung der maximal Infizierten in 
    #Abhängigkeit von Reproduktionswahrscheinlichkeit 
    iter_variaton = 100
    #Array zur varrierung des Wertes für beta
    beta_var = np.linspace(0.11, 0.5, iter_variaton)
    #Erstellung angepasstest Zeiten-Array da für kleine beta die Betrachtung
    #von 150 Tagen nicht ausreichend ist.
    t_long = np.linspace(0,1000,1001) 
    
    #Erstellung leerer Arrays für Plot
    Repro = np.zeros(iter_variaton)
    I_max = np.zeros(iter_variaton)
    #Schleife zur Berechnung der maximal Infizierten in 
    #Abhängigkeit von Reproduktionswahrscheinlichkeit
    for i in range(iter_variaton):
        beta = beta_var[i]
        Repro[i] = beta_var[i]/gamma

        F_var = odeint(diff_sys, f_0, t_long, args=(beta, gamma, sigma))
        S_var, I_var, R_var, D_var = F_var.T
        I_max[i] = np.max(I_var)
        
    #Arrays zur Darstellund der Belastbarkeitsgrenze des Gesungheitssystems von
    #10%
    x = np.linspace(1.1,5,iter_variaton)
    y = np.ones(iter_variaton)*0.1

    #Initiierung Plotfenster  
    fig = plt.figure(figsize=(10, 8))
    #Erstellung subplots
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    #Beschriftung
    ax1.set_title('Zeitlicher Verlauf des Anteils der SIR-Gruppen')
    ax1.set_xlabel('Zeit/ Tage')
    ax1.set_ylabel('Anteil')
    #Ploten SIR (ohne Lockdown) 
    ax1.plot(t, S, color ='g', label='Susceptible' )
    ax1.plot(t, I, color ='r', label='Infected' )
    ax1.plot(t, R, color ='b', label='Recovered')
    
    ax1.plot(t, D, color='k', label='Death')
    
    #Plotteb SIR (mit Lockdown)
    ax1.plot(t, S_lock, ls='--', color ='g', label='Susceptible (Lockdown)')
    ax1.plot(t, I_lock, ls='--', color ='r', label='Infected (Lockdown)')
    ax1.plot(t, R_lock, ls='--', color ='b', label='Recovered (Lockdown)')
    
    ax1.plot(t, D_Lock, ls='--', color='k', label='Death')
    
    #Beschriftung
    ax2.set_title('Abhängigkeit der maximalen Anzahl an Infizierten von Reproduktionsrate ')
    ax2.set_xlabel('Reproduktionsrate $R_0$')
    ax2.set_ylabel('Anteil')
    #Abhängigkeit max Infizierte von Repro.rate
    ax2.plot(Repro, I_max, label='maximaler Anteil an Infizierten I')
    ax2.plot(x,y, ls='--', color='r', label='Belastbarkeitsgrenze')
    #Legenden
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    #Grid
    ax1.grid()
    ax2.grid()
    #Anordnung der Plots anpassen
    fig.subplots_adjust(hspace=0.3)
    
    plt.show()

if __name__ == "__main__":
    main()
    
    
# Diskussion zur Ergänzung:

#Durch die Einführung der Gruppe der Toten D kann ein besserer Überblick über 
#den Verlauf der Pandemie geschaffen werden.

#Die vermutlich wichtigsten Kennzahlen stellen den Anteil an Verstorbenen an der 
#Gesamtbevölkerung und das Verhältnis von Verstorbenen zu Genesenen dar.
#Diese beiden kennzahlen hängen einzig von der jeweiligen 
#Genesungswahrscheinlichkeit und Lethalität ab. Da beim verwendeten Modell (ohne
#Lockdown) von einer vollständigen Infizierung der Bevölkerung angegangen wird.

#Weitere Erweiterungen des SIR-Modells:
#- Berücksichtigung Geburten- und allgemeine Sterberate bzw. N nicht constant 
#- Impfungen ab bestimmten Stichtag
#- zeitliche Abhängigkeit des Ansteckungsverhaltens (Ansteckung erst nach x 
#Tagen bzw. bis x tage nach infektion möglich)


    


 

