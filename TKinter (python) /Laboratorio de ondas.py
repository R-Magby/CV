import tkinter as tk
from tkinter import ttk
from tkinter import Button, Frame, Tk, Entry,ttk,Label,DoubleVar,Spinbox,IntVar,StringVar,Radiobutton,Checkbutton
import numpy as np

########### GRAFICO #############################

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.pyplot import Figure, show
from matplotlib.animation import FuncAnimation
from numpy import zeros, amax
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch


class Grafico(tk.Frame):
    def __init__(self, parent,data,data_x, limite,clase,marked):
        self.si=tk.Frame.__init__(self, parent)
        self.lim=limite
        self.marked=marked
        self.datos=data
        self.posicion=data_x
        self.posicion=data_x
        self.clase=clase

        self.fig = Figure(figsize=(12,5), dpi=100)
        self.axl = self.fig.add_subplot(111)
        self.axl.set_ylim((-self.lim,self.lim))


        self.canvas=FigureCanvasTkAgg(self.fig,master=self.si)
        self.canvas.get_tk_widget().pack(pady=3,padx=3)
        self.ani=FuncAnimation(self.fig,self.animate,interval=20,frames=600,blit=False)
       

 
        self.limpiar()

    def pausar(self):
        self.ani.event_source.stop()
    def reanudar(self):
        self.ani.event_source.start()
    def limpiar(self):

        self.ani.event_source.stop()
        self.axl.clear()
        self.calcular()



    def animate(self,i):

        if self.clase.get()=="Transversal":
            if self.marked.get()==1:
                self.datos_blue=self.datos
                self.dato_red=[fila.tolist().pop(14) for fila in self.datos_blue]
                self.line_blue.set_ydata(self.datos_blue[i,:])
                self.line_red.set_ydata(self.dato_red[i])

            else:
                self.line.set_ydata(self.datos[i,:])

        elif self.clase.get()=='Longitudinal':
            if self.marked.get()==1:
                self.datos_blue=self.datos
                self.dato_red=[fila.tolist().pop(14) for fila in self.datos_blue]
                self.line_blue.set_xdata(self.datos_blue[i,:])
                self.line_red.set_xdata(self.dato_red[i])
            else:
                self.line.set_xdata(self.datos[i,:])

       
        elif self.clase.get()=="Pulso":
            if self.marked.get()==1:
                self.line.set_ydata(-self.datos[i,:])
            else:
                self.line.set_ydata(self.datos[i,:])

        elif  self.clase.get()=='Superposición':
            if self.marked.get()==1:
                self.line.set_ydata(-self.datos[i,:]+self.datos[i,::-1])
            else:
                self.line.set_ydata(self.datos[i,:]+self.datos[i,::-1])
        elif self.clase.get()=="Resorte":
            if self.marked.get()==1:
                self.line1.set_ydata(self.datos[:i])
                self.line1.set_xdata(self.posicion[:i])
                self.line2.set_ydata([0,self.datos[i]])
                self.line3.set_ydata(self.datos[i])       


        elif self.clase.get()=="2 Masas":
            if self.marked.get()==1:
                L=amax(abs(self.datos[:,0]))*4.0+amax(abs(self.datos[:,1]))*4.0
                self.line1.set_xdata(self.datos[i,0]+L/3)
                self.line3.set_xdata(self.datos[i,1]+L*2/3)       

        elif self.clase.get()=="3 Masas":
            if self.marked.get()==1:
                L=amax(abs(self.datos[:,0]))*4.0+amax(abs(self.datos[:,1]))*4.0+amax(abs(self.datos[:,2]))*4.0
                self.line1.set_xdata(self.datos[i,0]+amax(abs(self.datos[:,0]))*2)
                self.line3.set_xdata(self.datos[i,1]+amax(abs(self.datos[:,0]))*4+amax(abs(self.datos[:,1]))*2.0)   
                self.line4.set_xdata(self.datos[i,2]+amax(abs(self.datos[:,0]))*4+amax(abs(self.datos[:,1]))*4.0+amax(abs(self.datos[:,2]))*2.0)       
    

        elif self.clase.get()=="Cuerda":
            if self.marked.get()==1:
                self.line.set_ydata(self.datos[i,:])   
            else:
                self.line.set_ydata(self.datos[i,:])   

        elif self.clase.get()=="Pulsar" :
            if self.marked.get()==1:
                self.line1.set_ydata(self.datos[i,:,0]+self.datos[i,::-1,1]+self.datos[i,:,2])


        elif self.clase.get()=="Armonico":
                datos_r=self.datos[:,::-1,1]
                shape_x=len(self.datos[0,:,0])

                self.line1.set_ydata(self.datos[i,:,0]+datos_r[i,:]+self.datos[i,:,2])
                #self.line1.set_xdata(self.posicion[])


        else:
            self.ani.event_source.stop()
        return self.line,


    def calcular(self):

        if self.clase.get()=="Transversal":
            if self.marked.get()==1:
                self.datos_blue=self.datos
                self.posicion_blue=self.posicion
                self.dato_red=[fila.tolist().pop(14) for fila in self.datos_blue]
                

                self.line_blue,=self.axl.plot(self.posicion_blue,self.datos_blue[0,:],"ob")
                self.line_red,=self.axl.plot(self.posicion[14],self.dato_red[0],"or")
                
                self.axl.set_xlabel("Distancia horizontal");self.axl.set_ylabel("Amplitud")

            else:
                self.line,=self.axl.plot(self.posicion,self.datos[0,:],"ob")


                self.axl.set_xlabel("Distancia horizontal");self.axl.set_ylabel("Amplitud")

        elif self.clase.get()=="Longitudinal":
            if self.marked.get()==1:
                self.datos_blue=self.datos
                self.posicion_blue=self.posicion
                self.dato_red=[fila.tolist().pop(14) for fila in self.datos_blue]

                self.line_blue,=self.axl.plot(self.datos_blue[0,:],zeros(len(self.posicion_blue)),"ob")
                self.line_red,=self.axl.plot(self.dato_red[0],0.0,"or")
                
                self.axl.set_xlabel("Distancia horizontal");self.axl.set_ylabel("Amplitud")

            else:
                self.line,=self.axl.plot(self.datos[0,:],zeros(len(self.posicion)),"ob")
                

                self.axl.set_xlabel("Distancia horizontal");self.axl.set_ylabel("Amplitud")


        elif self.clase.get()=="Pulso":
            if self.marked.get()==1:
                self.line,=self.axl.plot(self.posicion,-self.datos[0,:],"-k")
                
                self.axl.set_xlabel("Distancia horizontal");self.axl.set_ylabel("Amplitud")
                self.axl.set_ylim((-self.lim*1.5,self.lim*1.5))

            else:
                self.line,=self.axl.plot(self.posicion,self.datos[0,:],"-k")

                self.axl.set_xlabel("Distancia horizontal");self.axl.set_ylabel("Amplitud")
                self.axl.set_ylim((-self.lim*1.5,self.lim*1.5))


        elif self.clase.get()=="Superposición":
            if self.marked.get()==1:
                self.line,=self.axl.plot(self.posicion,-self.datos[0,:]+self.datos[0,::-1],"-k")
                
                self.axl.set_xlabel("Distancia horizontal");self.axl.set_ylabel("Amplitud")
                self.axl.set_ylim((-self.lim*2,self.lim*2))

            else:
                self.line,=self.axl.plot(self.posicion,self.datos[0,:]+self.datos[0,::-1],"-k")

                self.axl.set_xlabel("Distancia horizontal");self.axl.set_ylabel("Amplitud")
                self.axl.set_ylim((-self.lim*2,self.lim*2))


        elif self.clase.get()=="Resorte":
            if self.marked.get()==1:
                self.line1,=self.axl.plot(self.posicion[0],self.datos[0],"-b")
                self.line2,=self.axl.plot([0,0,],[0,self.datos[0]],"-k")
                self.line3,=self.axl.plot(0,self.datos[0],"ro")

                self.axl.set_xlabel("Tiempo");self.axl.set_ylabel("Amplitud")

                self.axl.set_ylim((-amax(abs(self.datos)),amax(abs(self.datos))))     
                self.axl.set_xlim((-1.5,20))

        elif self.clase.get()=="2 Masas":
            if self.marked.get()==1:
                L=amax(abs(self.datos[:,0]))*4.0+amax(abs(self.datos[:,1]))*4.0

                self.line2,=self.axl.plot([0,L],[0,0],"-k")
                self.line1,=self.axl.plot(self.datos[0,0]+amax(abs(self.datos[:,0]))*2,0,"mo",markersize=30)
                self.line3,=self.axl.plot(self.datos[0,1]+amax(abs(self.datos[:,0]))*4+amax(abs(self.datos[:,1]))*2.0,0,"ro",markersize=30)

                self.axl.set_xlabel("Distancia horizontal (Amplitud)");self.axl.set_ylabel("Altura")
                self.axl.set_xlim((0,L))     


        elif self.clase.get()=="3 Masas":
            if self.marked.get()==1:
                L=amax(abs(self.datos[:,0]))*4.0+amax(abs(self.datos[:,1]))*4.0+amax(abs(self.datos[:,2]))*4.0

                self.line2,=self.axl.plot([0,L],[0,0],"-k")
                self.line1,=self.axl.plot(self.datos[0,0]+amax(abs(self.datos[:,0]))*2.0,0,"mo",markersize=30)
                self.line3,=self.axl.plot(self.datos[0,1]+amax(abs(self.datos[:,0]))*4.0+amax(abs(self.datos[:,1]))*2.0,0,"ro",markersize=30)
                self.line4,=self.axl.plot(self.datos[0,2]+amax(abs(self.datos[:,0]))*4.0+amax(abs(self.datos[:,1]))*4.0+amax(abs(self.datos[:,2]))*2.0,0,"bo",markersize=30)

                self.axl.set_xlabel("Distancia horizontal");self.axl.set_ylabel("Altura")
                self.axl.set_xlim((0,L))     

        elif self.clase.get()=="Cuerda":
            if self.marked.get()==1:
                self.line,=self.axl.plot(self.posicion,self.datos[0,:],"-b")
                self.axl.set_ylim((-amax(abs(self.datos))-0.5,amax(abs(self.datos))+0.5))  

            else:
                self.line,=self.axl.plot(self.posicion,self.datos[0,:],"-b")

                self.axl.set_xlabel("Distancia horizontal");self.axl.set_ylabel("Amplitud")
                self.axl.set_ylim((-amax(abs(self.datos)),amax(abs(self.datos))))  

        elif self.clase.get()=="Pulsar":
            if self.marked.get()==1:
                alltura=amax(abs(self.datos[:,:]))*2*1.2
                self.axl.add_patch(patches.Rectangle( (50, (-amax(abs(self.datos[:,:]))*1.2)),50,alltura,facecolor = '#83EDAB',fill=True) )
                self.line1,=self.axl.plot(self.posicion,self.datos[0,:,0]+self.datos[0,::-1,1]+self.datos[0,:,2],"-k",markersize=20)
                self.axl.set_xlim((0,100))
                self.axl.set_ylim((-amax(abs(self.datos[:,:]))*1.2,amax(abs(self.datos[:,:]))*1.2))  

                self.axl.set_xlabel("Distancia horizontal");self.axl.set_ylabel("Amplitud")

        elif self.clase.get()=="Armonico":
            if self.marked.get()==1:
                alltura=amax(abs(self.datos[:,:]))*4
                self.axl.add_patch(patches.Rectangle( (50, (-amax(abs(self.datos[:,:]))*2)),50,alltura,facecolor = '#83EDAB',fill=True) )
                self.line1,=self.axl.plot(self.posicion,self.datos[0,:,0]+self.datos[-1,::-1,1]+self.datos[0,:,2],"-k",markersize=20)
                self.axl.set_xlim((0,100))
                self.axl.set_ylim((-amax(abs(self.datos[:,:]))*2.0,amax(abs(self.datos[:,:]))*2.0))  

                self.axl.set_xlabel("Distancia horizontal");self.axl.set_ylabel("Amplitud")

        self.iniciar()




    def iniciar(self):
        self.ani=FuncAnimation(self.fig,self.animate,interval=20,frames=600,blit=False)
        self.canvas.draw()
        show()

##################################################

########################## CALCULADORA ######################

class calculadora():
    def ondaT(self,A,Londas,Fondas,t,x):
        return A*np.sin(2*np.pi*x/Londas +2*np.pi*t/Fondas)
    def ondaM(self,A,Londas,Fondas,t,x):
        return A*np.cos(2*np.pi*x/Londas +2*np.pi*t/Fondas)
    def ondaL(self,A,Londas,Fondas,t,x):
        return 3*A*x+A*np.cos(2*np.pi*x/Londas +2*np.pi*t/Fondas)
    

    def gauss(self,x,A,t,Londas,frec):
        return A*np.exp((-(x/Londas-frec*t)**2)/(2))
    
    def oscilador(self,masa,Amplitud,cte_elasticidad,Roce,t,amplitud_f,frecuencia_f):
        omega_cero=np.sqrt(cte_elasticidad/masa)
        coeff_rozamiento=Roce/(2*masa)
        omega_gamma=np.sqrt(-coeff_rozamiento**2+omega_cero**2)
        G=amplitud_f/(masa*((omega_cero**2-frecuencia_f**2)**2+(2*coeff_rozamiento*frecuencia_f)**2))
        
        C=Amplitud-G*((omega_cero**2-frecuencia_f**2))
        D=(coeff_rozamiento*C-frecuencia_f*G*2*coeff_rozamiento*frecuencia_f)/omega_gamma

        return np.exp(-coeff_rozamiento*(t))*(D*np.sin(omega_gamma*t)+C*np.cos(omega_gamma*(t)))+G*((omega_cero**2-frecuencia_f**2)*np.cos(frecuencia_f*t)+2*coeff_rozamiento*frecuencia_f*np.cos(frecuencia_f*t)) 
    
    def Cuerda(self,Amplitud,nodos,Tension,masa,t,x):
        L=100
        vel=np.sqrt(Tension*L/masa)
        omega=np.pi*(nodos)*vel/L
        return Amplitud*np.sin(omega*t)*np.sin((nodos)*np.pi*x/L)
    
    def acople_2_masas(self,masa,x_1,x_2,cte_elasticidad,cte_elasticidad_12,t):
        sol=np.array([0,0])
        omega_1=np.sqrt((cte_elasticidad+2*cte_elasticidad_12)/masa)
        omega_2=np.sqrt((cte_elasticidad)/masa)
        q1=(x_1)/2;q2=(x_2)/2
        x1=q1*(np.cos(omega_1*t)+np.cos(omega_2*t))+q2*(np.cos(omega_1*t)-np.cos(omega_2*t))
        x2=q1*(np.cos(omega_1*t)-np.cos(omega_2*t))+q2*(np.cos(omega_1*t)+np.cos(omega_2*t))
        sol[0]=x1
        sol[1]=x2
        return sol
    
    def acople_3_masas(self,masa,x_1,x_2,x_3,cte_elasticidad,t):
        sol=np.array([0,0,0])
        omega_0=2*cte_elasticidad/masa
        omega_1=omega_0*(1-np.sqrt(2)/2)
        omega_2=omega_0
        omega_3=omega_0*(1+np.sqrt(2)/2)
        
        A=(x_1+x_3+x_2*np.sqrt(2))/4 ; B=(x_1-x_3)/2 ; C=(x_1+x_3-x_2*np.sqrt(2))/4

        G=A;D=np.sqrt(2)*A
        E=0;H=-B
        F=-C*np.sqrt(2); I=C

        x1=A*np.cos(omega_1*t)+B*np.cos(omega_2*t)+C*np.cos(omega_3*t)
        x2=D*np.cos(omega_1*t)+E*np.cos(omega_2*t)+F*np.cos(omega_3*t)
        x3=G*np.cos(omega_1*t)+H*np.cos(omega_2*t)+I*np.cos(omega_3*t)

        sol[0]=x1
        sol[1]=x2
        sol[2]=x3

        return sol

    def impedancia_pulso(self,Amplitud_indicente,frecuencia,Tension,m_1,m_2,t,x):
        L=100
        sol=np.array([0,0,0])


        mu1=m_1*2/L
        v1=np.sqrt(Tension/(mu1))

        mu2=m_2*2/L
        v2=np.sqrt(Tension/(mu2))

        R=(v2-v1)/(v1+v2)
        T=2*v2/(v2+v1)   

        k1=frecuencia/v1;k2=frecuencia/v2       

        fase=x*(k1-k2)/k2

        if x>=L/2 :
            Amplotud_reflejado=Amplitud_indicente*R
            phi_r=Amplotud_reflejado*np.exp((-(x*k1-frecuencia*t)**2)/(4))

            Amplitud_transimitido=Amplitud_indicente*T
            phi_t=Amplitud_transimitido*np.exp((-((x+fase)*k2-frecuencia*t)**2)/(4))

            sol[1]=phi_r;sol[2]=phi_t
        else:
            phi_i=Amplitud_indicente*np.exp((-(x*k1-frecuencia*t)**2)/(4))
            sol[0]=phi_i
        return sol
    
    def impedancia_armonico(self,Amplitud_indicente,frecuencia,Tension,m_1,m_2,t,x):
        L=100
        sol=np.array([0,0,0])


        mu1=m_1*2/L
        v1=np.sqrt(Tension/(mu1))

        mu2=m_2*2/L
        v2=np.sqrt(Tension/(mu2))

        R=(v2-v1)/(v1+v2)
        T=2*v2/(v2+v1)

        k1=frecuencia/(v1);k2=frecuencia/(v2)       

        fase=x*(k1-k2)/k2

        if  x>=L/2:
            Amplotud_reflejado=Amplitud_indicente*R
            phi_r=Amplotud_reflejado*np.cos(((x*k1-frecuencia*t)))

            Amplitud_transimitido=Amplitud_indicente*T
            phi_t=Amplitud_transimitido*np.cos((((x+fase)*k2-frecuencia*t)))
            sol[1]=phi_r;sol[2]=phi_t
        else:
            phi_i=Amplitud_indicente*np.cos(((x*k1-frecuencia*t)))
            sol[0]=phi_i
        return sol
    
    def __init__(self,onda):
        self.t=np.linspace(0,20,500)
        self.clase_onda=onda.var[4]
        if self.clase_onda.get()=="Transversal" or self.clase_onda.get()=="Longitudinal" :

            self.periodo_espacial=onda.var[0]
            self.periodo_temporal=onda.var[1]
            self.amplitud=onda.var[2]
            self.velocidad=onda.var[3]
            self.posicion=np.linspace(0,20,30)
            self.datos_onda=np.zeros((len(self.t),len(self.posicion)))

        elif self.clase_onda.get()=="Pulso" or self.clase_onda.get()=="Superposición":

            self.amplitud=onda.var[2]
            self.periodo_temporal=onda.var[1]
            self.periodo_espacial=onda.var[0]

            self.velocidad=onda.var[3]
            self.posicion=np.linspace(0,100,250)
            self.datos_onda=np.zeros((len(self.t),len(self.posicion)))

        elif self.clase_onda.get()=="Resorte":
            self.amplitud=onda.var[3]
            self.masa=onda.var[0]
            self.elasticidad=onda.var[1]
            self.roce=onda.var[2]
            self.posicion=np.linspace(0,100,110)
            self.datos_onda=np.zeros((len(self.t)))
            self.amplitud_f=onda.var[6]
            self.frecuencia_f=onda.var[7]

            self.velocidad=onda.var[8]

        elif self.clase_onda.get()=="2 Masas":
            self.x_1=onda.var[3]
            self.x_2=onda.var[5]
            self.masa=onda.var[0]
            self.elasticidad_1=onda.var[1]
            self.elasticidad_12=onda.var[2]

            self.nodo1=onda.Nodos[0]
            self.nodo2=onda.Nodos[1]
            self.nodo3=onda.Nodos[2]

            self.posicion=np.linspace(0,100,110)
            self.datos_onda=np.zeros((len(self.t),2))

        elif self.clase_onda.get()=="3 Masas":
            self.x_3=onda.var[7]
            self.x_1=onda.var[3]
            self.x_2=onda.var[5]
            self.masa=onda.var[0]
            self.elasticidad_1=onda.var[1]

            self.nodo1=onda.Nodos[0]
            self.nodo2=onda.Nodos[1]
            self.nodo3=onda.Nodos[2]

            self.posicion=np.linspace(0,100,110)
            self.datos_onda=np.zeros((len(self.t),3))

        elif self.clase_onda.get()=="Cuerda":
            self.amplitud=onda.var[3]
            self.masa=onda.var[0]
            self.Tension=onda.var[1]
            self.nodos=onda.var[6]
            self.posicion=np.linspace(0,100,200)
            self.datos_onda=np.zeros((len(self.t),len(self.posicion)))

            self.longitud_de_onda=onda.var[7]

        elif self.clase_onda.get()=="Pulsar":
            self.amplitud_I=onda.var[3]
            self.masa_1=onda.var[0]
            self.masa_2=onda.var[2]

            self.frecuencia=onda.var[6]
            self.Tension=onda.var[1]

            self.posicion=np.linspace(0,100,200)
            self.datos_onda=np.zeros((len(self.t),len(self.posicion),3))

            self.velocidad_1=onda.var[7]
            self.velocidad_2=onda.var[8]
        elif self.clase_onda.get()=="Armonico":
            self.amplitud_I=onda.var[3]
            self.masa_1=onda.var[0]
            self.masa_2=onda.var[2]

            self.frecuencia=onda.var[6]
            self.Tension=onda.var[1]

            self.posicion=np.linspace(0,100,500)
            self.datos_onda=np.zeros((len(self.t),len(self.posicion),3))

            self.velocidad_1=onda.var[7]
            self.velocidad_2=onda.var[8]
        else:
            self.posicion=np.linspace(0,20,30)
            self.datos_onda=np.zeros((len(self.t)))


        self.Procesar()
    def Procesar(self):

        self.Estados()
        self.calcular()

    def Estados(self):
        tipo_onda = str(self.clase_onda.get())
        if tipo_onda=='Transversal' or tipo_onda=='Longitudinal' or tipo_onda=="Pulso":
            self.velocidad.set(self.periodo_espacial.get()*self.periodo_temporal.get())
        elif tipo_onda=='Resorte':
            self.velocidad.set(round(np.sqrt(self.elasticidad.get()/self.masa.get()),5))

        elif tipo_onda=="2 Masas":
            self.nodo1.set("x1 = x2")
            self.nodo2.set("x1 = -x2")
            self.nodo3.set("--")

        elif tipo_onda=="3 Masas":
            self.nodo1.set('x3 = x1 y x2 = '+str(round(np.sqrt(2)*self.x_1.get(),4)))
            self.nodo2.set('x3 = -x1 y x2 = 0')
            self.nodo3.set('x3 = x1 y x2 = '+str(-round(np.sqrt(2)*self.x_1.get(),4)))
        
        elif tipo_onda	=="Cuerda":
            self.longitud_de_onda.set(2*100/self.nodos.get())

        elif tipo_onda	=='Pulsar'or self.clase_onda.get()=="Armonico":
            self.velocidad_1.set(self.Tension.get()*100/(self.masa_1.get()*2))
            self.velocidad_2.set(self.Tension.get()*100/(self.masa_2.get()*2))

        else:
            pass


    def calcular(self):

        if self.clase_onda.get()=='Transversal':
            for t in list(range(len(self.t))):
                for x in list(range(len(self.posicion))):
                    self.datos_onda[t,x]=self.ondaT(float(self.amplitud.get()),int(self.periodo_espacial.get()),float(self.periodo_temporal.get()),self.t[t],self.posicion[x])

        elif self.clase_onda.get()=="Longitudinal":
            for t in list(range(len(self.t))):
                for x in list(range(len(self.posicion))):
                    self.datos_onda[t,x]=self.ondaL(float(self.amplitud.get()),int(self.periodo_espacial.get()),float(self.periodo_temporal.get()),self.t[t],self.posicion[x])


        elif self.clase_onda.get()=="Resorte":
            for t in list(range(len(self.t))):
                self.datos_onda[t]=self.oscilador(float(self.masa.get()),float(self.amplitud.get()),float(self.elasticidad.get()),float(self.roce.get()),self.t[t],float(self.amplitud_f.get()),float(self.frecuencia_f.get()))

        elif self.clase_onda.get()=="2 Masas":
            for t in list(range(len(self.t))):
                fun=self.acople_2_masas(float(self.masa.get()),float(self.x_1.get()),float(self.x_2.get()),float(self.elasticidad_1.get()),float(self.elasticidad_12.get()),self.t[t])
                for i in [0,1]:
                    self.datos_onda[t,i]=fun[i]

        elif self.clase_onda.get()=="3 Masas":
            for t in list(range(len(self.t))):
                fun=self.acople_3_masas(float(self.masa.get()),float(self.x_1.get()),float(self.x_2.get()),float(self.x_3.get()),float(self.elasticidad_1.get()),self.t[t])
                for i in [0,1,2]:
                    self.datos_onda[t,i]=fun[i]

        elif self.clase_onda.get()=='Pulso' or self.clase_onda.get()=='Superposición':
            for t in list(range(len(self.t))):
                for x in list(range(len(self.posicion))):
                    self.datos_onda[t,x]= self.gauss(self.posicion[x],float(self.amplitud.get()),self.t[t],float(self.periodo_espacial.get()),float(self.periodo_temporal.get()))

        elif self.clase_onda.get()=='Pulsar':
            for t in list(range(len(self.t))):
                for x in list(range(len(self.posicion))):
                    fun=self.impedancia_pulso(float(self.amplitud_I.get()),float(self.frecuencia.get()),float(self.Tension.get()),float(self.masa_1.get()),float(self.masa_2.get()),self.t[t],self.posicion[x])
                    for i in [0,1,2]:
                        self.datos_onda[t,x,i]= fun[i]

        elif self.clase_onda.get()=='Armonico':
            for t in list(range(len(self.t))):
                for x in list(range(len(self.posicion))):
                    fun=self.impedancia_armonico(float(self.amplitud_I.get()),float(self.frecuencia.get()),float(self.Tension.get()),float(self.masa_1.get()),float(self.masa_2.get()),self.t[t],self.posicion[x])
                    for i in [0,1,2]:
                        self.datos_onda[t,x,i]= fun[i]
       

        elif self.clase_onda.get()=='Cuerda':
            for t in list(range(len(self.t))):
                for x in list(range(len(self.posicion))):
                    self.datos_onda[t,x]=self.Cuerda(float(self.amplitud.get()),int(self.nodos.get()),float(self.Tension.get()),float(self.masa.get()),self.t[t],self.posicion[x])







###############################
LARGEFONT =("Verdana", 12)


class tkinterApp(tk.Tk):


    def __init__(self):
        super().__init__( )

        self.title("Laboratorio de ondas")

        self.geometry("1250x750")
        self.config(bg="black")

        self.icono_chico = tk.PhotoImage(file="icon-16.png")
        self.icono_grande = tk.PhotoImage(file="icon-32.png")
        self..iconphoto(False, self.icono_grande, self.icono_chico)

        self.frame1=Frame(self)
        self.frame1.pack(fill="x")
        self.frame1.config(width=250,height=150)

        self.frames = {}

        for F in (StartPage, Oscilador_armonico, Pulso,Resorte,Cuerda,Sist_acoplado,Reflexion_y_Transmision):
            frame = F(self.frame1, self)
            self.frames[F] = frame
            frame.grid(row = 0, column = 0, sticky ="nsew")


        self.show_frame(StartPage)

#---------------Variable De Onda--------------------

        self.var=[DoubleVar(value=1),DoubleVar(value=1),DoubleVar(value=1),DoubleVar(value=0.0),StringVar(value="Transversal"),IntVar(value=0)]


#--------------------Visual--------------------
        self.muchu=Grafico(self,np.zeros((600,30)),np.linspace(0,20,30),1.2,self.var[4],self.var[5])

#--------------funciones--------------
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


    def change(self,lista,posicion,limites,clase,marked):
        self.muchu.datos=lista
        self.muchu.posicion=posicion
        self.muchu.clase=clase
        self.muchu.marked=marked
        self.muchu.lim=limites
        self.muchu.limpiar()


    def pausar(self):
        self.muchu.pausar()
    def reanudar(self):
        self.muchu.reanudar()

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        label = ttk.Label(self, text ="Inicio", font = LARGEFONT,anchor="center")
        label.grid(row = 0, column = 4, padx = 10, pady = 10)

        button1 = ttk.Button(self, text ="Oscilador armonico",command = lambda : controller.show_frame(Oscilador_armonico))
        button1.grid(row = 1, column = 1, padx = 10, pady = 10)

        button2 = ttk.Button(self, text ="Pulso",command = lambda : controller.show_frame(Pulso))
        button2.grid(row = 1, column = 2, padx = 10, pady = 10)

        button2 = ttk.Button(self, text ="Resorte",command = lambda : controller.show_frame(Resorte))
        button2.grid(row = 1, column = 3, padx = 10, pady = 10)

        button3 = ttk.Button(self, text ="Cuerda",command = lambda : controller.show_frame(Cuerda))
        button3.grid(row = 1, column = 4, padx = 10, pady = 10)
        
        button3 = ttk.Button(self, text ="Reflexion y Transmision",command = lambda : controller.show_frame(Reflexion_y_Transmision))
        button3.grid(row = 1, column = 5, padx = 10, pady = 10)

        button4 = ttk.Button(self, text ="Sistema acoplado",command = lambda : controller.show_frame(Sist_acoplado))
        button4.grid(row = 1, column = 6, padx = 10, pady = 10)

        
class Oscilador_armonico(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text ="Oscilador armonico", font = LARGEFONT)
        label.grid(row = 0, column = 1, padx = 10, pady = 10)


        button1 = ttk.Button(self, text ="atras",command = lambda : controller.show_frame(StartPage))
        button1.grid(row = 0, column = 0, padx = 10, pady = 10)

#---------------Variable De Onda--------------------
        self.var=[DoubleVar(value=10),DoubleVar(value=0.5),DoubleVar(value=1.0),DoubleVar(value=0.0),StringVar(value="Transversal"),IntVar(value=0)]
        self.lim=[-self.var[2].get(),self.var[2].get(),0,30]
        self.estadito=StringVar(value="...")

#---------------Tipo De Onda--------------------

        self.clase1 = ttk.Combobox(self,state="readonly",textvariable=self.var[4],values=["Transversal", "Longitudinal"],postcommand=lambda:controller.pausar())
        self.clase1.grid(row=1,column=4)
#---------------longitud De Onda--------------------
        self.text_long=Entry(self,textvariable=self.var[0])
        self.text_long.grid(row=2,column=1)
        self.var_long=Label(self,text="Longitud de onda :")
        self.var_long.grid(row=2,column=0,sticky="e",pady=3,padx=2)

#---------------Frecuencia--------------------
        self.Frecuencia=Label(self,text=" Frecuencia :")
        self.Frecuencia.grid(row=3,column=0,sticky="e",pady=3,padx=2)
        self.text_fre=Entry(self,textvariable=self.var[1])
        self.text_fre.grid(row=3,column=1)

#---------------Amplitud--------------------
        self.Amplitud=Label(self,text="Amplitud :")
        self.Amplitud.grid(row=1,column=0,sticky="e",pady=3,padx=2)
        self.text_amp=Entry(self,textvariable=self.var[2])
        self.text_amp.grid(row=1,column=1)

#---------------bolita roja--------------------
        self.sentido=Checkbutton(self,text="Marcar en rojo",variable=self.var[5],command=lambda:controller.pausar())
        self.sentido.grid(row=2,column=4)
#---------------Botones--------------------
        self.clase1.bind("<<ComboboxSelected>>", self.check)



        self.Simula=ttk.Button(self,text="Simular",command=lambda:controller.change(calculadora(self).datos_onda,calculadora(self).posicion,self.lim,self.var[4],self.var[5])) 
        self.Simula.grid(row=1,column=5,sticky="w")
        self.pausar=ttk.Button(self,text="Pausar",command=lambda:controller.pausar())
        self.pausar.grid(row=2,column=5,sticky="w")
        self.pausar=ttk.Button(self,text="Reanudar",command=lambda:controller.reanudar())
        self.pausar.grid(row=3,column=5,sticky="w")
#---------------Estado-------------------
        self.Velocidad=Label(self,text="Velocidad de Propagacion :")
        self.Velocidad.grid(row=4,column=0,sticky="e",pady=3,padx=2)
        self.entry_vel=Entry(self,state="readonly",readonlybackground="black",textvariable=calculadora(self).velocidad)
        self.entry_vel.grid(row=4,column=1)
        self.entry_vel.config(background="black",fg="#03f943",justify="right")

#---------------Texto--------------------
        self.texto_texto=StringVar(value="Onda transversal:\n Es una onda en la que la vibración o desplazamiento de las partículas del medio es perpendicular \n a la dirección de propagación de la onda. Ejemplo: ondas electromagnéticas como la luz.")

        self.texto_transversal=Label(self,textvariable=self.texto_texto)
        self.texto_transversal.grid(row=0,column=6,sticky=tk.W)
    def check(self,event):

        if self.var[4].get()=="Transversal":
            self.texto_texto.set("Onda transversal:\n Es una onda en la que la vibración o desplazamiento de las partículas del medio es perpendicular \n a la dirección de propagación de la onda. Ejemplo: ondas electromagnéticas como la luz.")

        elif self.var[4].get()=="Longitudinal":
            self.texto_texto.set("Onda longitudinal:\n Es una onda en la que las partículas del medio vibran en la misma dirección en la que se propaga \nla onda. Ejemplo: ondas sonoras en el aire.")
class Pulso(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text ="Pulso", font = LARGEFONT)
        label.grid(row = 0, column = 4, padx = 10, pady = 10)

        button1 = ttk.Button(self, text ="atras",command = lambda : controller.show_frame(StartPage))
        button1.grid(row = 0, column = 0, padx = 10, pady = 10)


#---------------Variable De Onda--------------------
        self.var=[DoubleVar(value=2.0),DoubleVar(value=10.0),DoubleVar(value=1.0),DoubleVar(value=1.0),StringVar(value="Pulso"),IntVar(value=0)]
        self.estadito=StringVar(value="...")

#---------------Tipo De Onda--------------------

        self.clase1 = ttk.Combobox(self,state="readonly",textvariable=self.var[4],values=["Pulso", "Superposición"],postcommand=lambda:controller.pausar())
        self.clase1.grid(row=1,column=4)

#---------------longitud De Onda--------------------
        self.text_long=Entry(self,textvariable=self.var[0])
        self.text_long.grid(row=2,column=1)
        self.var_long=Label(self,text="Longitud de onda :")
        self.var_long.grid(row=2,column=0,sticky="e",pady=3,padx=2)

#---------------Frecuencia--------------------
        self.Frecuencia=Label(self,text=" Frecuencia :")
        self.Frecuencia.grid(row=3,column=0,sticky="e",pady=3,padx=2)
        self.text_fre=Entry(self,textvariable=self.var[1])
        self.text_fre.grid(row=3,column=1)

#---------------Amplitud--------------------
        self.Amplitud=Label(self,text="Amplitud :")
        self.Amplitud.grid(row=1,column=0,sticky="e",pady=3,padx=2)
        self.text_amp=Entry(self,textvariable=self.var[2])
        self.text_amp.grid(row=1,column=1)
#---------------Sentido--------------------
        self.sentido=Checkbutton(self,text="Invertir",variable=self.var[5])
        self.sentido.grid(row=2,column=4)
#---------------Botones--------------------

        self.Simula=ttk.Button(self,text="Simular",command=lambda:controller.change(calculadora(self).datos_onda,calculadora(self).posicion,calculadora(self).amplitud.get(),self.var[4],self.var[5]))
        self.Simula.grid(row=1,column=5,sticky="w")
        self.pausar=ttk.Button(self,text="Pausar",command=lambda:controller.pausar())
        self.pausar.grid(row=2,column=5,sticky="w")
        self.pausar=ttk.Button(self,text="Reanudar",command=lambda:controller.reanudar())
        self.pausar.grid(row=3,column=5,sticky="w")
#---------------Estado--------------------


        self.var[3]=calculadora(self).velocidad
        self.Velocidad=Label(self,text="Velocidad de Propagacion :")
        self.Velocidad.grid(row=4,column=0,sticky="e",pady=3,padx=2)
        self.entry_vel=Entry(self,state="readonly",readonlybackground="black",textvariable=self.var[3])
        self.entry_vel.grid(row=4,column=1)
        self.entry_vel.config(background="black",fg="#03f943",justify="right")

#---------------Texto--------------------
        
        self.texto_interferencia=Label(self,text="Interferencia:\n Fenómeno en el cual dos o más ondas se combinan y pueden reforzarse (constructiva)\n o anularse (destructiva) cuando se superponen en un punto del espacio")
        self.texto_interferencia.grid(row=0,column=6,sticky=tk.W)

class Resorte(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text ="Resorte", font = LARGEFONT)
        label.grid(row = 0, column = 4, padx = 10, pady = 10)



        button1 = ttk.Button(self, text ="atras",command = lambda : controller.show_frame(StartPage))
        button1.grid(row = 0, column = 0, padx = 10, pady = 10)



#---------------Variable De Onda--------------------
        self.var=[DoubleVar(value=0.3),DoubleVar(value=9),DoubleVar(value=0.22),DoubleVar(value=1.0),StringVar(value="Resorte"),IntVar(value=1),DoubleVar(value=0.3),DoubleVar(value=6.0),DoubleVar(value=1)]
        self.lim=[-self.var[3].get(),self.var[3].get(),0,30]
        self.estadito=StringVar(value="...")

#---------------Tipo De Onda--------------------

        self.clase1 = ttk.Combobox(self,state="readonly",textvariable=self.var[4],values=["Resorte"])
        self.clase1.grid(row=1,column=5)

#---------------Subtitulo--------------------
        self.resorte=Label(self,text="Resorte :")
        self.resorte.grid(row=1,column=1)
#---------------Masa--------------------
        self.masa=Label(self,text="Masa :")
        self.masa.grid(row=3,column=0,sticky="e",pady=3,padx=2)
        self.text_masa=Entry(self,textvariable=self.var[0])
        self.text_masa.grid(row=3,column=1)

#---------------Cte_elasticidad--------------------
        self.Cte_elasticidad=Label(self,text=" Constante de elasticidad :")
        self.Cte_elasticidad.grid(row=4,column=0,sticky="e",pady=3,padx=2)
        self.text_Cte_elasticidad=Entry(self,textvariable=self.var[1])
        self.text_Cte_elasticidad.grid(row=4,column=1)

#---------------roce--------------------
        self.Roce=Label(self,text="Roce :")
        self.Roce.grid(row=5,column=0,sticky="e",pady=3,padx=2)
        self.text_roce=Entry(self,textvariable=self.var[2])
        self.text_roce.grid(row=5,column=1)
#---------------Amplitud--------------------
        self.Amplitud=Label(self,text="Amplitud :")
        self.Amplitud.grid(row=2,column=0,sticky="e",pady=3,padx=2)
        self.text_amp=Entry(self,textvariable=self.var[3])
        self.text_amp.grid(row=2,column=1)

#---------------Subtitulo--------------------
        self.Externo=Label(self,text="Fuerza externa :")
        self.Externo.grid(row=1,column=3)
#---------------Amplitud_f--------------------
        self.Amplitud=Label(self,text="Amplitud  :")
        self.Amplitud.grid(row=2,column=2,sticky="e",pady=3,padx=2)
        self.text_amp=Entry(self,textvariable=self.var[6])
        self.text_amp.grid(row=2,column=3)
#---------------Frecuencia_f--------------------
        self.Amplitud=Label(self,text="Frecuencia :")
        self.Amplitud.grid(row=3,column=2,sticky="e",pady=3,padx=2)
        self.text_amp=Entry(self,textvariable=self.var[7])
        self.text_amp.grid(row=3,column=3)

#---------------Botones--------------------

        self.Simula=ttk.Button(self,text="Simular",command=lambda:controller.change(calculadora(self).datos_onda,calculadora(self).t,self.var[3].get()+self.var[6].get()+0.5,self.var[4],self.var[5]))
        self.Simula.grid(row=2,column=5,sticky="w")
        self.pausar=ttk.Button(self,text="Pausar",command=lambda:controller.pausar())
        self.pausar.grid(row=3,column=5,sticky="w")
        self.pausar=ttk.Button(self,text="Reanudar",command=lambda:controller.reanudar())
        self.pausar.grid(row=4,column=5,sticky="w")
#---------------Frecuencia natural--------------------
        self.Externo=Label(self,text="Frecuencia natural :")
        self.Externo.grid(row=4,column=3)

        self.entry_vel=Entry(self,state="readonly",readonlybackground="black",textvariable=calculadora(self).velocidad)
        self.entry_vel.grid(row=5,column=3)
        self.entry_vel.config(background="black",fg="#03f943",justify="right")

#---------------Texto--------------------

        self.texto_resorte=Label(self,text="Resonancia: \n Es el efecto de fortalecimiento y amplificación de una vibración o onda en un objeto o sistema \n cuando la frecuencia de la excitación coincide con su frecuencia natural de oscilación.")
        self.texto_resorte.grid(row=0,column=6,sticky=tk.W)


#---------------Estado--------------------
class Cuerda(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text ="Cuerda", font = LARGEFONT)
        label.grid(row = 0, column = 4, padx = 10, pady = 10)



        button1 = ttk.Button(self, text ="atras",command = lambda : controller.show_frame(StartPage))
        button1.grid(row = 0, column = 0, padx = 10, pady = 10)



#---------------Variable De Onda--------------------
        self.var=[DoubleVar(value=0.3),DoubleVar(value=9),DoubleVar(value=15),DoubleVar(value=1.0),StringVar(value="Cuerda"),IntVar(value=1),IntVar(value=1),DoubleVar(value=1)]
        self.lim=[-self.var[3].get(),self.var[3].get(),0,30]
        self.estadito=StringVar(value="...")

#---------------Tipo De Onda--------------------

        self.clase1 = ttk.Combobox(self,state="readonly",textvariable=self.var[4],values=["Cuerda"])
        self.clase1.grid(row=1,column=2)


#---------------Masa 1--------------------
        self.masa1=Label(self,text="Masa 1 :")
        self.masa1.grid(row=2,column=0,sticky="e",pady=3,padx=2)
        self.text_masa1=Entry(self,textvariable=self.var[0])
        self.text_masa1.grid(row=2,column=1)


#---------------Tension--------------------
        self.Tension=Label(self,text="Tension :")
        self.Tension.grid(row=4,column=0,sticky="e",pady=3,padx=2)
        self.txt_Tension=Entry(self,textvariable=self.var[1])
        self.txt_Tension.grid(row=4,column=1)
#---------------Nodos--------------------
        self.nodos=Label(self,text="Nodos :")
        self.nodos.grid(row=5,column=0,sticky="e",pady=3,padx=2)
        self.txt_nodos=tk.Spinbox(self, from_ = 1, to = 14,increment = 1,textvariable=self.var[6])
        self.txt_nodos.grid(row=5,column=1)

#---------------Amplitud--------------------
        self.Amplitud=Label(self,text="Amplitud :")
        self.Amplitud.grid(row=1,column=0,sticky="e",pady=3,padx=2)
        self.text_amp=Entry(self,textvariable=self.var[3])
        self.text_amp.grid(row=1,column=1)


#---------------Botones--------------------


        self.Simula=ttk.Button(self,text="Simular",command=lambda:controller.change(calculadora(self).datos_onda,calculadora(self).posicion,self.var[3].get()*2,self.var[4],self.var[5]))
        self.Simula.grid(row=1,column=3,sticky="w")
        self.pausar=ttk.Button(self,text="Pausar",command=lambda:controller.pausar())
        self.pausar.grid(row=2,column=3,sticky="w")
        self.pausar=ttk.Button(self,text="Reanudar",command=lambda:controller.reanudar())
        self.pausar.grid(row=3,column=3,sticky="w")
#---------------Longitud de onda--------------------
        self.Externo=Label(self,text="Longitud de onda :")
        self.Externo.grid(row=3,column=2)
        self.entry_vel=Entry(self,state="readonly",readonlybackground="black",textvariable=calculadora(self).longitud_de_onda)
        self.entry_vel.grid(row=4,column=2)
        self.entry_vel.config(background="black",fg="#03f943",justify="right")
#---------------Texto--------------------

        self.texto_resorte=Label(self,text="Nodos:\n Son puntos específicos en una cuerda vibrante donde no se produce movimiento durante su vibración.\n Permanecen fijos, sin oscilación alguna, debido a la interferencia constructiva y destructiva de las ondas reflejadas \n en los extremos sujetos de la cuerda.",anchor="center")
        self.texto_resorte.grid(row=0,column=6,sticky=tk.W)

class Reflexion_y_Transmision(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text ="Reflexion y Transmision", font = LARGEFONT)
        label.grid(row = 0, column = 4, padx = 10, pady = 10)



        button1 = ttk.Button(self, text ="atras",command = lambda : controller.show_frame(StartPage))
        button1.grid(row = 0, column = 0, padx = 10, pady = 10)



#---------------Variable De Onda--------------------
        self.var=[DoubleVar(value=10),DoubleVar(value=10),DoubleVar(value=40),DoubleVar(value=1000),StringVar(value="Pulsar"),IntVar(value=1),DoubleVar(value=1),DoubleVar(value=1),DoubleVar(value=1)]
        self.lim=[-self.var[3].get(),self.var[3].get(),0,30]
        self.estadito=StringVar(value="...")

#---------------Tipo De Onda--------------------

        self.clase1 = ttk.Combobox(self,state="readonly",textvariable=self.var[4],values=["Pulsar","Armonico"],postcommand=lambda:controller.pausar())
        self.clase1.grid(row=1,column=2)


#---------------Masa 1--------------------
        self.masa1=Label(self,text="Masa medio 1 :")
        self.masa1.grid(row=2,column=0,sticky="e",pady=3,padx=2)
        self.text_masa1=Entry(self,textvariable=self.var[0])
        self.text_masa1.grid(row=2,column=1)

#---------------Masa 2--------------------
        self.masa2=Label(self,text="Masa medio 2 :")
        self.masa2.grid(row=3,column=0,sticky="e",pady=3,padx=2)
        self.text_masa2=Entry(self,textvariable=self.var[2])
        self.text_masa2.grid(row=3,column=1)

#---------------Tension--------------------
        self.Tension=Label(self,text="Tension :")
        self.Tension.grid(row=4,column=0,sticky="e",pady=3,padx=2)
        self.txt_Tension=Entry(self,textvariable=self.var[1])
        self.txt_Tension.grid(row=4,column=1)

#---------------Frecuencia--------------------
        self.Amplitud=Label(self,text="Frecuencia :")
        self.Amplitud.grid(row=5,column=0,sticky="e",pady=3,padx=2)
        self.text_amp=Entry(self,textvariable=self.var[6])
        self.text_amp.grid(row=5,column=1)
#---------------Amplitud--------------------
        self.Amplitud=Label(self,text="Amplitud :")
        self.Amplitud.grid(row=1,column=0,sticky="e",pady=3,padx=2)
        self.text_amp=Entry(self,textvariable=self.var[3])
        self.text_amp.grid(row=1,column=1)


#---------------Botones--------------------


        self.Simula=ttk.Button(self,text="Simular",command=lambda:controller.change(calculadora(self).datos_onda,calculadora(self).posicion,self.var[3].get()*2,self.var[4],self.var[5]))
        self.Simula.grid(row=1,column=3,sticky="w")
        self.pausar=ttk.Button(self,text="Pausar",command=lambda:controller.pausar())
        self.pausar.grid(row=2,column=3,sticky="w")
        self.pausar=ttk.Button(self,text="Reanudar",command=lambda:controller.reanudar())
        self.pausar.grid(row=3,column=3,sticky="w")
#---------------Longitud de onda--------------------
        self.Externo=Label(self,text="Velocidad medio 1 :")
        self.Externo.grid(row=2,column=2)
        self.entry_vel=Entry(self,state="readonly",readonlybackground="black",textvariable=calculadora(self).velocidad_1)
        self.entry_vel.grid(row=3,column=2)
        self.entry_vel.config(background="black",fg="#03f943",justify="right")


        self.Externo=Label(self,text="Velocidad medio 2 :")
        self.Externo.grid(row=4,column=2)
        self.entry_vel=Entry(self,state="readonly",readonlybackground="black",textvariable=calculadora(self).velocidad_2)
        self.entry_vel.grid(row=5,column=2)
        self.entry_vel.config(background="black",fg="#03f943",justify="right")
#---------------Texto--------------------
        
        self.texto_resorte=Label(self,text=" ")
        self.texto_resorte.grid(row=2,column=6,sticky=tk.W)





class Sist_acoplado(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text ="Sist.Acoplado", font = LARGEFONT)
        label.grid(row = 0, column = 4, padx = 10, pady = 10)



        button1 = ttk.Button(self, text ="atras",command = lambda : controller.show_frame(StartPage))
        button1.grid(row = 0, column = 0, padx = 10, pady = 10)



#---------------Variable De Onda--------------------
        self.var=[DoubleVar(value=1.0),DoubleVar(value=3.0),DoubleVar(value=3.0),DoubleVar(value=100.0),StringVar(value="2 Masas"),DoubleVar(value=50.0),IntVar(value=1),DoubleVar(value=175.0)]
        self.lim=[-self.var[3].get(),self.var[5].get(),0,30]

        self.Nodos=[StringVar(value="1"),StringVar(value="2"),StringVar(value="3")]
        self.estadito=StringVar(value="...")

#---------------Tipo De Onda--------------------

        self.clase1 = ttk.Combobox(self,state="readonly",textvariable=self.var[4],values=["2 Masas","3 Masas"],postcommand=lambda:controller.pausar())
        self.clase1.grid(row=1,column=2)


#---------------Masa--------------------
        self.masa=Label(self,text="Masa :")
        self.masa.grid(row=3,column=0,sticky="e",pady=3,padx=2)
        self.text_masa=Entry(self,textvariable=self.var[0])
        self.text_masa.grid(row=3,column=1)

#---------------posicion 1--------------------
        self.posicion_1=Label(self,text="Posicion incial 1 (magenta) :")
        self.posicion_1.grid(row=4,column=0,sticky="e",pady=3,padx=2)
        self.txt_posicion_1=Entry(self,textvariable=self.var[3])
        self.txt_posicion_1.grid(row=4,column=1)
#---------------posicion 2--------------------
        self.posicion_2=Label(self,text="Posicion inicial 2 (rojo) :")
        self.posicion_2.grid(row=5,column=0,sticky="e",pady=3,padx=2)
        self.txt_posicion_2=Entry(self,textvariable=self.var[5])
        self.txt_posicion_2.grid(row=5,column=1)
#---------------posicion 3--------------------
        self.posicion_3=Label(self,text="Posicion inicial 3 (azul) :")
        self.posicion_3.grid(row=6,column=0,sticky="e",pady=3,padx=2)
        self.txt_posicion_3=Entry(self,textvariable=self.var[7])
        self.txt_posicion_3.grid(row=6,column=1)

#---------------elasticidad k--------------------
        self.elasticidad=Label(self,text="Elasticidad k :")
        self.elasticidad.grid(row=1,column=0,sticky="e",pady=3,padx=2)
        self.text_elasticidad=Entry(self,textvariable=self.var[1])
        self.text_elasticidad.grid(row=1,column=1)
#---------------elasticidad k12--------------------
        self.elasticidad_12=Label(self,text="Elasticidad k12 :")
        self.elasticidad_12.grid(row=2,column=0,sticky="e",pady=3,padx=2)
        self.text_elasticidad_12=Entry(self,textvariable=self.var[2])
        self.text_elasticidad_12.grid(row=2,column=1)

#---------------Botones--------------------
        self.clase1.bind("<<ComboboxSelected>>", self.check)

        self.Simula=ttk.Button(self,text="Simular",command=lambda:controller.change(calculadora(self).datos_onda,calculadora(self).posicion,self.var[3].get()*2,self.var[4],self.var[6]))
        self.Simula.grid(row=1,column=3,sticky="w")
        self.pausar=ttk.Button(self,text="Pausar",command=lambda:controller.pausar())
        self.pausar.grid(row=2,column=3,sticky="w")
        self.pausar=ttk.Button(self,text="Reanudar",command=lambda:controller.reanudar())
        self.pausar.grid(row=3,column=3,sticky="w")
#---------------Nodos--------------------

        self.nodos=Label(self,text="Condiciones para los nodos :")
        self.nodos.grid(row=3,column=2)
        
        self.entry_nodo_l=Entry(self,state="readonly",readonlybackground="black",textvariable=calculadora(self).nodo1)
        self.entry_nodo_l.grid(row=4,column=2)
        self.entry_nodo_l.config(background="black",fg="#03f943",justify="right")

        self.entry_nodo_2=Entry(self,state="readonly",readonlybackground="black",textvariable=calculadora(self).nodo2)
        self.entry_nodo_2.grid(row=5,column=2)
        self.entry_nodo_2.config(background="black",fg="#03f943",justify="right")


        self.entry_nodo_3=Entry(self,state="readonly",readonlybackground="black",textvariable=calculadora(self).nodo3)
        self.entry_nodo_3.grid(row=6,column=2)
        self.entry_nodo_3.config(background="black",fg="#03f943",justify="right")
#---------------Texto--------------------
        
        self.texto_resorte=Label(self,text="acople: masa iguales, fase o desfase, nodos")
        self.texto_resorte.grid(row=2,column=6,sticky=tk.W)
        self.check(1)

    def check(self,event):
        if self.var[4].get()=="2 Masas":
            self.txt_posicion_3.configure(state="disabled")
            self.text_elasticidad_12.configure(state="normal")
        elif self.var[4].get()=="3 Masas":
            self.text_elasticidad_12.configure(state="disabled")
            self.txt_posicion_3.configure(state="normal")


app = tkinterApp()
app.mainloop()
