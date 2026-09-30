import numpy as np
from scipy.integrate import *
from numpy.fft import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from IPython.display import HTML, display
from matplotlib.ticker import ScalarFormatter
from types import SimpleNamespace
import pickle
from tqdm.auto import tqdm

class Modell:
    '''
    ---Az osztály tagjai a különböző paraméterekkel
    kapott adatsorok.
    ---Létrehozáshoz egy fájlnevet adunk meg:
       ez tartalmazza a paramétereket.
    '''
    
    def __init__(self,file=None):
        
        if file==None:
            self.d=0
            self.m=0
            self.R=0
            self.rho=0
            self.k=0
            self.o=0
            self.b=0
            self.u=np.array([])
            self.t=np.array([])
            self.n=0
            self.solution=None
            
        else:
            with open(file, 'rb') as f:
                params = pickle.load(f)
                
            self.d=params.d
            self.m=params.m
            self.R=params.R
            self.rho=params.rho
            self.k=params.k
            self.o=params.o
            self.b=params.b
            self.u=params.u
            self.t=params.t
            self.n=params.n
            self.solution=params.solution

    def details(self):
        print('d: ' ,self.d)
        print('m: ' ,self.m)
        print('R: ' ,self.R)
        print('rho: ' ,self.rho)
        print('k: ' ,self.k)
        print('o: ' ,self.o)
        print('b: ' ,self.b)
        
    def change_params(self,params):
        'Paraméterek SimpleNamespace objektumban.'
        
        self.d=params.d
        self.m=params.m
        self.R=params.R
        self.rho=params.rho
        self.k=params.k
        self.o=params.o
        self.u=params.u       
        self.b=params.b
        self.t=params.t
        self.solution=None

        
    def start_maker(self,n):
    
        self.n=n

        phi_0=np.linspace(0,2*np.pi,self.n, endpoint=False)
        omega_0=np.zeros(self.n)
        theta=np.array([0])
        Omega=np.array([self.o])

        self.X0=np.concatenate((phi_0,omega_0,theta, Omega))
    
    def diffeq(self, t, X):
        """
        A modell: R sugarú körön egyenletesen elrendezett golyók összekötve valamilyen rugóval.
        Körülöttük egy másik golyó halad.

        """

        N=self.n
        tau=(self.m/self.d)**(1/2)
        r=self.rho/self.R
        K=self.b * self.R
        lmbd=2**(1/2)*np.sin(np.pi/N)
        C=(self.k * self.b)/(self.d * self.R)

        phi = np.array(X[:N]) #első N darab elem a szögek
        omega = np.array(X[N:2*N])  #második N darab a szögsebességek
        theta = np.array([X[-2]]) #utolsó előtti elem a kezdeti szög
        Omega = np.array([X[-1]]) #utolsó a szögsebessége

        phi_m1 = np.roll(phi,-1)
        phi_p1 = np.roll(phi,1)

        #belső golyók szöggyorsulása:
        beta = tau**(-2) * (np.sin(phi_m1-phi) - np.sin(phi-phi_p1) +
                           lmbd * ((np.sin(phi - phi_p1))/(np.sqrt(1-np.cos(phi-phi_p1))) -
                           (np.sin(phi_m1 - phi))/(np.sqrt(1-np.cos(phi_m1-phi))) ) +
                           C * r *((np.exp(-K*np.sqrt(1 - 2*r*np.cos(theta - phi) + r**2))*np.sin(theta - phi)) /
                           (np.sqrt(1 - 2*r*np.cos(theta - phi) + r**2 ))) )


        #külső golyó szöggyorsulása:
        Beta = np.array([- tau**(-2)* C/r * np.sum(np.exp(-K*np.sqrt(1 - 2*r*np.cos(theta - phi) + r**2))*np.sin(theta - phi)
                        / np.sqrt(1 - 2*r*np.cos(theta - phi) + r**2 ))])

        U = np.concatenate((omega, beta, Omega, Beta))

        return U
        
    def solve_ivp(self,t, rtol=1e-3, atol=1e-6, max_step=0.01):
        
        self.atol=atol
        self.rtol=rtol
        solution=solve_ivp(self.diffeq, (0,t), self.X0, method='RK45', rtol=rtol, atol=atol, max_step=max_step, dense_output=True)
        self.solution=(solution.sol)
    
    def solution_divider(self,t):
        self.t=t
        self.u=self.solution(t).T
        self.phi=self.u[:,:self.n]
        self.omega=self.u[:,self.n:(2*self.n)]
        self.theta=self.u[:,-2]
        self.Omega=self.u[:,-1]

        
    def save(self,file):
        params=SimpleNamespace(d=self.d, m=self.m, R=self.R, rho=self.rho, k=self.k, o=self.o, u=self.u, t=self.t, n=self.n, b=self.b, solution=self.solution)

        with open(file +'.pkl', 'wb') as f:
            pickle.dump(params, f)

            
    def phonon_spectrum(self, ax):

        v_belso=(self.R*self.omega)
        t=self.t
        kran_b = np.linspace(-np.pi, np.pi, self.n)
        wran_b = np.linspace(-np.pi/np.diff(t)[0],np.pi/np.diff(t)[0], len(t))
        im = ax.pcolormesh(kran_b, wran_b, np.abs(np.fft.fftshift(np.fft.ifft(np.fft.fft(v_belso, axis=1),axis=0), axes=(1,0)))**2, norm=LogNorm(), cmap='viridis')

        ax.set_ylim(0,3)
        ax.get_figure().colorbar(im, ax=ax)
        ax.set_ylabel(r'$\omega$ körfrekvencia', size=13)
        ax.set_xlabel(r'$k$ hullámszám', size=13)
        ax.set_title(r'$\omega (k)$', size=14) 
        
    def energy_calc(self):

        N=self.n
        T=self.t
        d,m,R,rho,k,b=(self.d, self.m, self.R, self.rho, self.k, self.b) 
        l=2*R*np.sin(np.pi/(N))    

        phi_p1 = np.roll(self.phi,1, axis=1)

        V_in=1/2 * d * (R*np.sqrt(2)* np.sqrt(1-np.cos(self.phi-phi_p1)) - l)**2
        V_out=-k*np.exp(-1*b*np.sqrt(R**2-2*R*rho*np.cos(self.theta[:, np.newaxis]-self.phi)+rho**2))
        K_in=1/2 * m * R**2 * self.omega**2
        K_out=1/2 * m * self.Omega**2 *rho**2
        
        E=np.sum(V_in+V_out+K_in, axis=1)+K_out

        self.E=np.array(E)
        self.K_in=np.array(K_in)
        self.K_out=np.array(K_out)
        self.V_out=np.array(V_out)
        self.V_in=np.array(V_in)

    def animation(self, title, zoom=True):
        '''
        Animációt készít a modellről.
        -- bemenetként kapja: cím , bool zoom kapcsoló
        '''

        if zoom:
            fig, ax = plt.subplots(figsize=(4, 2)) 
            zoom_window_y=self.rho-self.R + (self.rho-self.R)*0.2
            zoom_window_x=0.5

            # Összes belső pont kirajzolásához
            bent = ax.scatter([], [], s=20, color='navy', alpha=0.6)
            # A külső golyó mindig az origóban lesz (0, 0)
            kint = ax.scatter([0], [0], s=25, color='red', zorder=5)
            
            # A tengelyeket egy fix, kicsi ablakra állítjuk be az origó (külső golyó) körül
            ax.set_xlim(-zoom_window_x, zoom_window_x)
            # Kicsit több hely a "lent" irányba, hogy a középpont felé lévő golyók látszódjanak
            ax.set_ylim(-zoom_window_y, (self.rho-self.R)*0.2)  
            ax.set_aspect('equal')
            ax.set_title(title)
            ax.grid(True, linestyle='--', alpha=0.5)          
            
            def update(frame):
                # Abszolút koordináták kiszámítása az adott frame-ben
                x_out = self.rho * np.cos(self.theta[frame])
                y_out = self.rho * np.sin(self.theta[frame])
                
                x_in = self.R * np.cos(self.phi[frame, :])
                y_in = self.R * np.sin(self.phi[frame, :])
                
                # Koordináta-transzformáció (Eltolás)
                x_rel = x_in - x_out
                y_rel = y_in - y_out
            
                # Forgatás szöge
                alpha = self.theta[frame]
                
                # Forgatási mátrix alkalmazása úgy, hogy a középpont ALUL (negatív Y) legyen
                x_rot = -x_rel * np.sin(alpha) + y_rel * np.cos(alpha)
                y_rot =  x_rel * np.cos(alpha) + y_rel * np.sin(alpha)

                kint.set_offsets(np.column_stack(([0], [0])))
               
                bent.set_offsets(np.column_stack((x_rot, y_rot)))
            
                return bent, kint
            
        else:
            fig, ax = plt.subplots(figsize=(4, 4)) 
            bent = ax.scatter(self.R*np.cos(self.phi[0,:]), self.R*np.sin(self.phi[0,:]), s=10, color='navy', alpha=0.6)
            kint = ax.scatter(self.rho*np.cos(self.theta[0]), self.rho*np.sin(self.theta[0]), s=15, color='red')

            ax.set_xlim(-self.R-0.3, self.R+0.3)
            ax.set_ylim(-self.R-0.3, self.R+0.3)
            ax.set_aspect('equal')
            ax.set_title(title)

            ax.grid(True, linestyle='--', alpha=0.5)


            def update(frame):

                x1 = self.R * np.cos(self.phi[frame, :])
                y1 = self.R * np.sin(self.phi[frame, :])
                bent.set_offsets(np.column_stack((x1, y1)))

                x2 = self.rho * np.cos(self.theta[frame])
                y2 = self.rho * np.sin(self.theta[frame])

                kint.set_offsets(np.column_stack((x2, y2)))

                return bent, kint

        lepeskoz = 100
        frame_lista = range(0, self.phi.shape[0], lepeskoz)
        
        ani = animation.FuncAnimation(
            fig=fig, 
            func=update, 
            frames=tqdm(frame_lista, desc="Animáció"),
            interval=120,
            blit=True                               
        )
        
        display(HTML(ani.to_jshtml()))
        plt.close(fig)
