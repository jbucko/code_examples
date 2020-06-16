#include <iostream>
using namespace std;

//prints 2D array of length dim
void pArray(int dim,int** arr)
{
int k,l;
for(k=1;k<=dim;k++)
    {
    for(l=1;l<=dim;l++) if (arr[k][l]>=0) cout<<" "<<arr[k][l];
                        else              cout<<arr[k][l];
    cout<<"\n";
    }
}

//creates empty 2D array of size n
int **createLattice(int n)
{
int ii,jj;
int** a = new int*[n+2];
for (int ii = 0; ii < n+2; ++ii)
    a[ii] = new int[n+2];
for (ii=1;ii<=n;ii++)
    {
    for (jj=1;jj<=n;jj++)
        a[ii][jj]=0;
    }

return a;
}

//random walker function on lattice arr, seed at xc,yc number of walkers k and lattice size n
void rWalker(int **arr,int yc,int xc,int k,int n)
{
int iter,posI,posJ,start,ii,jj,time=0;
double p,q;
bool c1,c2,extend;
int rmin=10, rmax=20;
//initialization of walker
for (iter=0;iter<k;iter++)
{
extend=false;
start=rand()%(2*rmin)-rmin;
p=rand()%1000/(double)1000;
if (p<0.25) {
            posI=xc-rmin;
            posJ=yc+start;
            }
    else if (p<0.5)
            {
            posJ=yc+rmin;
            posI=xc+start;
            }
        else if (p<0.75)
            {
            posI=xc+rmin;
            posJ=yc+start;
            }
            else
            {
            posJ=yc-rmin;
            posI=xc+start;
            }


//walk 
ii=posI;
jj=posJ;

do
    {
    q=rand()%1000/(double)1000;
    if (q<0.25) {
                ii-=1;
                }
        else if (q<0.5)
                {
                jj+=1;
                }
            else if (q<0.75)
                {
                ii+=1;
                }
                else
                {
                jj+=-1;
                }
    c1=((ii<xc+rmax)&&(ii>xc-rmax)&&(jj<yc+rmax)&&(jj>yc-rmax)); // particle is still within the allowed region
    c2=(arr[ii-1][jj]==0) && (arr[ii+1][jj]==0) && (arr[ii][jj-1]==0) && (arr[ii][jj+1]==0); // particle does not have neighbor 
    }while((c1==true)&&(c2==true));
//time increase and controll of need to extend generation region
if (c2==false)
    {
    time+=1;
    arr[ii][jj]=time/1000+1;
    if ((abs(ii-xc)>rmin*0.7)||(abs(jj-yc)>rmin*0.7))   {
                                                        extend=true;
                                                        }
    }
//pushing generation region of walkers
if (extend==true)
    {
    rmin+=10;
    rmax=2*rmin;
    if (rmax>=n*0.5) rmax=n/2-3;

    if (rmin>=n*0.5)
        {
        break;
        }
    }

}
}

//sandbox method
int* sandbox(int **arr,int n,int xc,int yc)
{
int* a=new int[51];
int rad,ii,jj,r;
r=0;
do  {
    rad=5+r*5;
    for (ii=1;ii<=n;ii++)
    for (jj=1;jj<=n;jj++)
        {
        if (((ii>xc-rad)&&(ii<xc+rad)&&(jj>yc-rad)&&(jj<yc+rad)) && arr[ii][jj]!=0) a[r]+=1;
        }
    r+=1;
    }while(rad<0.5*n);
for (r=0;r<51;r++) cout<<5+r*5<<" "<<a[r]<<"\n";
cout<<"\n";
return a;
}

int main()
{
srand(time(NULL)+rand());
int N=500;
int **lattice=createLattice(N);
int xC=N/2+1,yC=N/2+1;
int K=1000000,o;

lattice[xC][yC]=1;


rWalker(lattice,yC,xC,K,N);
sandbox(lattice,N,xC,yC);

return 0;
}
