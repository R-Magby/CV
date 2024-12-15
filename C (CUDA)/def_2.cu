#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <time.h> 
#include <string.h>


    
__global__ void masa_grilla(float *C, float *rho, float *r,float step_grid,int N_body,int N_g,float m_p,int t){

        int i =threadIdx.x + blockDim.x*blockIdx.x;

        if(i<N_body){
           float gridx=r[t*N_body*2  +i*2]/step_grid;
           float gridy=r[t*N_body*2  +i*2+1]/step_grid;

           if (gridx<N_g-1 && gridy<N_g-1 && gridx>0 && gridy>0 ){

        int idx_grid= (int)round(gridx);
        int idy_grid= (int)round(gridy);
        
        int id_grid=idy_grid*N_g + idx_grid;

        float d_x = fabs(C[id_grid*2]-r[t*N_body*2  +i*2]) , d_y = fabs(C[id_grid*2+1]-r[t*N_body*2  +i*2+1]);
        float t_x = fabs(step_grid  -  d_x),  t_y = fabs(step_grid  -  d_y);
        int signox=1;
        int signoy=1;

        rho[id_grid] = (rho[id_grid] + m_p * t_x * t_y) /(step_grid*step_grid);

        if (C[id_grid*2]-r[t*N_body*2  +i*2]>0 ){signox=-1;}if( C[id_grid*2+1]-r[t*N_body*2  +i*2+1]>0){signoy=-1;}

        int celda_horizontal =  id_grid  + signox;
        int celda_vertical = id_grid +  signoy*N_g;
        int celda_inversa =  id_grid +  signox +  signoy*N_g;


        if (idx_grid >0  &&  idx_grid  < N_g-1){
            rho[celda_horizontal] = (rho[celda_horizontal] + m_p * d_x * t_y)/(step_grid*step_grid) ;} 

        if (idy_grid >0  &&  idy_grid < N_g-1){
            rho[celda_vertical] = (rho[celda_vertical] + m_p * t_x * d_y )/(step_grid*step_grid);}

        if (idx_grid >0  &&  idx_grid  < N_g-1  &&  idy_grid >0  &&  idy_grid < N_g-1){

            rho[celda_inversa] = (rho[celda_inversa] + m_p * d_x * d_y)/(step_grid*step_grid) ;}

        }
    }
            __syncthreads();


}

__global__ void potencial(float *phi_n, float *masa, float *err,int N, float delta, int inter){

	int idx=threadIdx.x + blockDim.x*blockIdx.x  ;
	int idy=threadIdx.y + blockDim.y*blockIdx.y  ;

    int i=1;
    float err_min=0.000001;
    float thread_error;
    float temp;
    float phi_n1;

    err[0]=1.0;
    err[1]=0.0;

    while ( err[0] >= err_min){
        thread_error=0.0;        
        err[1]=0.0;

        if (idx<N-1 && idy<N-1 && idx>0 && idy>0){
            phi_n1 = (0.25)*(phi_n[(idy+1)*N+idx]+phi_n[(idy-1)*N+idx]+phi_n[(idy)*N+idx+1]+phi_n[(idy)*N+idx-1]
                -4.0*3.1415*delta*delta*masa[idy*N+idx] );

            thread_error += (phi_n1-phi_n[idy*N+idx])*(phi_n1-phi_n[idy*N+idx]);

        }
        err[1]+=thread_error;

    __syncthreads();



    temp=phi_n[idy*N+idx];
    phi_n[idy*N+idx]=phi_n1;
    phi_n1=temp;
    //phi_n[idy*N+idx]=phi_n1[idy*N+idx];
    err[0]=err[1];

    i++;
    __syncthreads();

    
    }


}

__global__ void gravedad(float *phi_n, float *gravity,int N, float delta){

	int idx=threadIdx.x + blockDim.x*blockIdx.x +1 ;
	int idy=threadIdx.y + blockDim.y*blockIdx.y  +1 ;

    if (idx>0 && idx<N-1 && idy<N-1 && idy>0){

        gravity[idy*N+idx]= -(phi_n[idy*N+idx+1]-phi_n[idy*N+idx-1])/(2*delta);
        gravity[N*N+idy*N+idx]= -(phi_n[(idy+1)*N+idx]-phi_n[(idy-1)*N+idx])/(2*delta);
    }

    __syncthreads();

}

__global__ void actualizacion(float *C, float *r, float *v, float *gravity ,
    int N_body,int N_g, float *G, float  step_grid, float m_p, float dt, int t){

    int i =threadIdx.x + blockDim.x*blockIdx.x;
    float v_leap_x,v_leap_y;

    if(i<N_body){
        float gridx=r[t*N_body*2  +i*2]/step_grid;
        float gridy=r[t*N_body*2  +i*2+1]/step_grid;

    if (gridx<N_g-1 && gridy<N_g-1 && gridx>0 && gridy>0 ){

     int idx_grid= (int)round(gridx);
     int idy_grid= (int)round(gridy);
     
    int id_grid=idy_grid*N_g + idx_grid;

    float d_x = fabs(C[id_grid*2]-r[t*N_body*2+i*2]) , d_y = fabs(C[id_grid*2+1]-r[t*N_body*2+i*2+1]);
    float t_x = fabs(step_grid  -  d_x),  t_y = fabs(step_grid  -  d_y);


    int signox=1;
    int signoy=1;
    if (C[id_grid*2]-r[t*N_body*2  +i*2]>0.0 ){signox=-1;}if( C[id_grid*2+1]-r[t*N_body*2  +i*2+1]>0.0){signoy=-1;}

    v_leap_x   =   v[i*2] + G[i*2+0]*dt/(2.0*m_p);
    r[(t+1)*N_body*2+i*2]= r[t*N_body*2+i*2] + v_leap_x*dt;

    v_leap_y  =  v[i*2+1] + G[i*2+1]*dt/(2.0*m_p);
    r[(t+1)*N_body*2+i*2+1]= r[t*N_body*2+i*2+1] + v_leap_y*dt;


    G[i*2+0]= gravity[id_grid]*t_x * t_y;
    G[i*2+1]= gravity[N_g*N_g + id_grid]*t_x * t_y;





    int celda_horizontal =  id_grid  + signox;
    int celda_vertical = id_grid +  signoy*N_g;
    int celda_inversa =  id_grid +  signox +  signoy*N_g;


    if (idx_grid >0  &&  idx_grid  < N_g-1){
        G[i*2+0] += gravity[celda_horizontal]*d_x * t_y;
        G[i*2+1] += (gravity[N_g*N_g + celda_horizontal]* d_x * t_y );} 

    if (idy_grid >0  &&  idy_grid < N_g-1){
        G[i*2+0] += gravity[celda_vertical]*t_x * d_y;
        G[i*2+1] += (gravity[N_g*N_g + celda_vertical]* t_x * d_y );}

    if (idx_grid >0  &&  idx_grid  < N_g-1  &&  idy_grid >0  &&  idy_grid < N_g-1){
        G[i*2+0] += gravity[celda_inversa]*d_x * d_y;
        G[i*2+1] += (gravity[N_g*N_g + celda_inversa]* d_x * d_y );}

    }

//    act coordenda x        
        v[i*2]  =  v_leap_x+G[i*2+0]*dt/(2.0*m_p);

//    act coordenda y        
        v[i*2+1]   =   v_leap_y + G[i*2+1]*dt/(2.0*m_p);
    }
        __syncthreads();

}

void guardar_particulas(float *data,int size) {

    FILE *fp = fopen("n_body12.dat", "wb");
	
	fwrite(&(data[0]),sizeof(float),size,fp);
	fclose(fp);
}
void guardar_phi(float *data,int size) {


    FILE *fp = fopen("phi.dat", "wb");
    //printf("%s",arch);
	
	fwrite(&(data[0]),sizeof(float),size,fp);
	fclose(fp);
}
void guardar_masas(float *data, int size) {

    FILE *fp = fopen("masas.dat", "wb");
	
	fwrite(&(data[0]),sizeof(float),size,fp);
	fclose(fp);
}
void rellenar_r(float *r,int size){

    for(int i=0;i<size;i++){
        r[i*2]=rand()%(1024);
        r[i*2+1]=rand()%(1024);
        
    }
}
void rellenar_v(float *r,int size){

    for(int i=0;i<size;i++){
        r[i*2]=(float)rand()/(float)(RAND_MAX)*0.0 ;
        r[i*2+1]=(float)rand()/(float)(RAND_MAX)*0.0;
        
    }
}
void rellenar_rho(float *masa,int size){

    for(int i=0;i<size;i++){

            masa[i]=0.0;

        
    }
}
void rellenar_grilla(float *grilla,int size,float step_grid){

    for(int i=0;i<size;i++){
        for(int j=i*size;j<size+i*size;j++){

            grilla[j*2+1]=i*step_grid;
            grilla[j*2]=(j-i*size)*step_grid;

        }
    }
}
void imprimir_archivo(float *v2,int size){
    FILE *arch; 
    arch=fopen("n_body.dat","rb");
    int numElem = fgetc(arch);
    fread(&v2, sizeof(float), numElem, arch);
    fclose(arch);
}


int main(int argc, char *argv[]){


    float *dot_r;
    float *cuda_dot_r;

    float *r_tn;
    float *cuda_r_tn;

    float *grilla;
    float *cuda_grilla;
    float *cuda_grilla_G;
    float *G,*g;

    float *rho,*cuda_rho;

    float *phi,*cuda_phi_n;

    float *cuda_err;

    int N_grilla=512;
    int N_b=5000;
    int L_size = 1024;

    int size_cuerpos = 2*N_b*sizeof(float);
    int size_grilla = 2*N_grilla*N_grilla*sizeof(float);
    int size_time=30000;

    float dt=0.005;

    double time_spent = 0.0;
    clock_t begin = clock();


    g=(float *)malloc(2*N_b*sizeof(float));

    r_tn=(float *)malloc(size_time*size_cuerpos);
    dot_r=(float *)malloc(size_cuerpos);
    grilla=(float *)malloc(size_grilla);
    rho=(float  *)malloc(N_grilla*N_grilla*sizeof(float));

    phi=(float  *)malloc(N_grilla*N_grilla*sizeof(float));
    

    rellenar_rho(r_tn,N_b*size_time*2);

    rellenar_r(r_tn,N_b);

    rellenar_v(dot_r,N_b);
    rellenar_rho(g,2*N_b);

    rellenar_grilla(grilla,N_grilla,(float)L_size/N_grilla);
    rellenar_rho(rho,N_grilla*N_grilla);
    rellenar_rho(phi,N_grilla*N_grilla);








    cudaMalloc((void **)&cuda_r_tn, size_time*size_cuerpos);
    cudaMalloc((void **)&cuda_dot_r, size_cuerpos);
    cudaMalloc((void **)&cuda_grilla, size_grilla);
    cudaMalloc((void **)&cuda_grilla_G, size_grilla);
    cudaMalloc((void **)&G, 2*N_b*sizeof(float));

    cudaMalloc((void **)&cuda_phi_n, N_grilla*N_grilla*sizeof(float));
    cudaMalloc((void **)&cuda_rho, N_grilla*N_grilla*sizeof(float));

    cudaMemcpy(cuda_dot_r, dot_r, size_cuerpos, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_grilla, grilla, size_grilla, cudaMemcpyHostToDevice);

    cudaMemcpy(cuda_rho, rho, N_grilla*N_grilla*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_phi_n, phi, N_grilla*N_grilla*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(cuda_r_tn, r_tn, size_time*2*N_b*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(G, g, 2*N_b*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&cuda_err, sizeof(float)*2);
    cudaMalloc((void **)&G, sizeof(float)*2);


   int thread=32;
   dim3 bloque(thread,thread);
   dim3 grid((int)ceil((float)(N_grilla)/thread),(int)ceil((float)(N_grilla)/thread));

   
   printf("%f \n", dt);

    for (int t=0;t<size_time;t++){
        if (t%100==0){
            printf("%d \n",t);
        }
    
        masa_grilla<<<(int)ceil((float)N_b/1024),1024>>>(cuda_grilla,cuda_rho,cuda_r_tn,(float) L_size/N_grilla ,N_b,N_grilla,1,t);




        potencial<<<grid,bloque>>>(cuda_phi_n,cuda_rho,cuda_err,N_grilla,(float) L_size/N_grilla,100000);


        gravedad<<<grid,bloque>>>(cuda_phi_n,cuda_grilla_G,N_grilla,(float) L_size/N_grilla);


        actualizacion<<<(int)ceil((float)N_b/1024),1024>>>(cuda_grilla,cuda_r_tn,cuda_dot_r,cuda_grilla_G, N_b, N_grilla, G, (float)L_size/N_grilla,  1, dt,  t );
        if(t==0){
            cudaMemcpy(rho,cuda_rho, N_grilla*N_grilla*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(phi,cuda_phi_n, N_grilla*N_grilla*sizeof(float), cudaMemcpyDeviceToHost);
        }
        }
        cudaMemcpy(r_tn,cuda_r_tn, size_time*size_cuerpos, cudaMemcpyDeviceToHost);




        clock_t end = clock();
        
        time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
        printf("\nTiempo de ejecucion : %f seconds\n", time_spent);
		

        guardar_particulas(r_tn,2*N_b*size_time);
        guardar_phi(phi,N_grilla*N_grilla);
        guardar_masas(rho,N_grilla*N_grilla);

    cudaError_t err = cudaGetLastError();
    printf("Error: %s\n",cudaGetErrorString(err));
    free(g); free(dot_r); free(grilla); free(r_tn); free(phi); free(rho);

    cudaFree(G); cudaFree(cuda_dot_r); cudaFree(cuda_grilla);cudaFree(cuda_rho); cudaFree(cuda_err); cudaFree(cuda_grilla_G);
    cudaFree(cuda_phi_n); cudaFree(cuda_r_tn);


}
