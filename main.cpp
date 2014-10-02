#include <stdio.h>
#include <stdlib.h>
#include "src/shared_var.h"
#include "src/config.hpp"
#include "src/CSRdouble.hpp"
#include "src/IO.hpp"
#include "src/ParDiSO.hpp"
#include "src/RealMath.hpp"
#include "src/smat.h"
#include <cassert>

extern "C" {
    int MPI_Init(int *, char ***);
    int MPI_Finalize(void);
    int MPI_Dims_create(int, int, int *);
    int MPI_Barrier( MPI_Comm comm );
    void blacs_pinfo_ ( int *mypnum, int *nprocs );
    void blacs_get_ ( int *ConTxt, int *what, int *val );
    void blacs_gridinit_ ( int *ConTxt, char *order, int *nprow, int *npcol );
    void blacs_gridexit_ ( int *ConTxt );
    void blacs_pcoord_ ( int *ConTxt, int *nodenum, int *prow, int *pcol );
    void descinit_ ( int*, int*, int*, int*, int*, int*, int*, int*, int*, int* );
    void pdpotrf_ ( char *uplo, int *n, double *a, int *ia, int *ja, int *desca, int *info );
    void pdpotri_ ( char *uplo, int *n, double *a, int *ia, int *ja, int *desca, int *info );
    void pdsymm_( char *side, char *uplo, int *m, int *n, double *alpha, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb,
                  int *descb, double *beta, double *c, int *ic, int *jc, int *descc );
    void pddot_( int *n, double *dot, double *x, int *ix, int *jx, int *descx, int *incx, double *y, int *iy, int *jy, int *descy, int *incy );
    void dgesd2d_ ( int *ConTxt, int *m, int *n, double *A, int *lda, int *rdest, int *cdest );
    void dgerv2d_ ( int *ConTxt, int *m, int *n, double *A, int *lda, int *rsrc, int *csrc );
}

double d_one = 1.0, d_zero = 0.0, d_negone = -1.0;
int DLEN_=9, i_negone=-1, i_zero=0, i_one=1; // some many used constants
int Ddim, Adim, blocksize; //dimensions of different matrices
int lld_D, Dblocks, Drows, Dcols;
int size, *dims, * position, ICTXT2D, iam;
char *filenameD, *filenameA, *filenameB, *filenameC;
double lambda;
bool printsparseC_bool;
MPI_Status status;
int Bassparse_bool;

int main(int argc, char **argv) {
    int info, i, j, pcol;
    double *D, *AB_sol, *InvD_T_Block, *XSrow;
    int *DESCD, *DESCAB_sol, *DESCXSROW;
    CSRdouble BT_i, B_j;
    CSRdouble Asparse, Btsparse;

    //Initialise MPI and some MPI-variables
    info = MPI_Init ( &argc, &argv );
    if ( info != 0 ) {
        printf ( "Error in MPI initialisation: %d\n",info );
        return info;
    }

    position= ( int* ) calloc ( 2,sizeof ( int ) );
    if ( position==NULL ) {
        printf ( "unable to allocate memory for processor position coordinate\n" );
        return EXIT_FAILURE;
    }

    dims= ( int* ) calloc ( 2,sizeof ( int ) );
    if ( dims==NULL ) {
        printf ( "unable to allocate memory for grid dimensions coordinate\n" );
        return EXIT_FAILURE;
    }

    //BLACS is the interface used by PBLAS and ScaLAPACK on top of MPI

    blacs_pinfo_ ( &iam,&size ); 				//determine the number of processes involved
    info=MPI_Dims_create ( size, 2, dims );			//determine the best 2D cartesian grid with the number of processes
    if ( info != 0 ) {
        printf ( "Error in MPI creation of dimensions: %d\n",info );
        return info;
    }
//Until now the code can only work with square process grids
    //So we try to get the biggest square grid possible with the number of processes involved
    if (*dims != *(dims+1)) {
        while (*dims * *dims > size)
            *dims -=1;
        *(dims+1)= *dims;
        if (iam==0)
            printf("WARNING: %d processor(s) unused due to reformatting to a square process grid\n", size - (*dims * *dims));
        size = *dims * *dims;
        //cout << "New size of process grid: " << size << endl;
    }

    blacs_get_ ( &i_negone,&i_zero,&ICTXT2D );

    //Initialisation of the BLACS process grid, which is referenced as ICTXT2D
    blacs_gridinit_ ( &ICTXT2D,"R",dims, dims+1 );

    if (iam < size) {

        //The rank (iam) of the process is mapped to a 2D grid: position= (process row, process column)
        blacs_pcoord_ ( &ICTXT2D,&iam,position, position+1 );
        if ( *position ==-1 ) {
            printf ( "Error in proces grid\n" );
            return -1;
        }

        //Filenames, dimensions of all matrices and other important variables are read in as global variables (see src/readinput.cpp)
        info=read_input ( *++argv );
        if ( info!=0 ) {
            printf ( "Something went wrong when reading input file for processor %d\n",iam );
            return -1;
        }

        //blacs_barrier is used to stop any process of going beyond this point before all processes have made it up to this point.
        blacs_barrier_ ( &ICTXT2D,"ALL" );
        if ( * ( position+1 ) ==0 && *position==0 )
            printf ( "Reading of input-file succesful\n" );

        if ( * ( position+1 ) ==0 && *position==0 ) {
            printf("\nA sparse square matrix of dimension %d with a dense square submatrix with dimension %d \n", Adim+Ddim,Ddim);
            printf("was analyzed using %d (%d x %d) processors\n",size,*dims,*(dims+1));
        }

        pcol= * ( position+1 );

        //Define number of blocks needed to store a complete column/row of D
        Dblocks= Ddim%blocksize==0 ? Ddim/blocksize : Ddim/blocksize +1;

        //Define the number of rowblocks needed by the current process to store its part of the dense matrix D
        Drows= ( Dblocks - *position ) % *dims == 0 ? ( Dblocks- *position ) / *dims : ( Dblocks- *position ) / *dims +1;
        Drows= Drows<1? 1 : Drows;

        //Define the number of columnblocks needed by the current process to store its part of the dense matrix D
        Dcols= ( Dblocks - pcol ) % * ( dims+1 ) == 0 ? ( Dblocks- pcol ) / * ( dims+1 ) : ( Dblocks- pcol ) / * ( dims+1 ) +1;
        Dcols=Dcols<1? 1 : Dcols;

        //Define the local leading dimension of D (keeping in mind that matrices are always stored column-wise)
        lld_D=Drows*blocksize;

        //Initialise the descriptor of the dense distributed matrix
        DESCD= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
        if ( DESCD==NULL ) {
            printf ( "unable to allocate memory for descriptor for C\n" );
            return -1;
        }

        //D with dimensions (Ddim,Ddim) is distributed over all processes in ICTXT2D, with the first element in process (0,0)
        //D is distributed into blocks of size (blocksize,blocksize), having a local leading dimension lld_D in this specific process
        descinit_ ( DESCD, &Ddim, &Ddim, &blocksize, &blocksize, &i_zero, &i_zero, &ICTXT2D, &lld_D, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of matrix C returns info: %d\n",info );
            return info;
        }

        //Allocate the space necessary to store the part of D that is held into memory of this process.
        D = ( double* ) calloc ( Drows * blocksize * Dcols * blocksize,sizeof ( double ) );
        if ( D==NULL ) {
            printf ( "unable to allocate memory for Matrix D  (required: %ld bytes)\n", Drows * blocksize * Dcols * blocksize * sizeof ( double ) );
            return EXIT_FAILURE;
        }

        read_in_BD ( DESCD,D, BT_i, B_j, Btsparse ) ;

        blacs_barrier_ ( &ICTXT2D,"ALL" );
        if (iam==0)
            printf ( "Matrices B & D set up\n" );

        //Now every matrix has to read in the sparse matrix A

        Asparse.loadFromFile(filenameA);
	assert(Asparse.nrows == Adim);
	assert(Asparse.ncols == Adim);

        blacs_barrier_ ( &ICTXT2D,"ALL" );
	
	if(printsparseC_bool) {
            CSRdouble Dmat, Dblock, Csparse;
            Dblock.nrows=Dblocks * blocksize;
            Dblock.ncols=Dblocks * blocksize;
            Dblock.allocate(Dblocks * blocksize, Dblocks * blocksize, 0);
            Dmat.allocate(0,0,0);
            for (i=0; i<Drows; ++i) {
                for(j=0; j<Dcols; ++j) {
                    dense2CSR_sub(D + i * blocksize + j * lld_D * blocksize,blocksize,blocksize,lld_D,Dblock,( * ( dims) * i + *position ) *blocksize,
                                  ( * ( dims+1 ) * j + pcol ) *blocksize);
                    if ( Dblock.nonzeros>0 ) {
                        if ( Dmat.nonzeros==0 ) {
                            Dmat.make2 ( Dblock.nrows,Dblock.ncols,Dblock.nonzeros,Dblock.pRows,Dblock.pCols,Dblock.pData );
                        }
                        else {
                            Dmat.addBCSR ( Dblock );
                        }
                    }

                    Dblock.clear();
                }
            }
            blacs_barrier_(&ICTXT2D,"A");
            if ( iam!=0 ) {
                //Each process other than root sends its Dmat to the root process.
                MPI_Send ( & ( Dmat.nonzeros ),1, MPI_INT,0,iam,MPI_COMM_WORLD );
                MPI_Send ( & ( Dmat.pRows[0] ),Dmat.nrows + 1, MPI_INT,0,iam+size,MPI_COMM_WORLD );
                MPI_Send ( & ( Dmat.pCols[0] ),Dmat.nonzeros, MPI_INT,0,iam+2*size,MPI_COMM_WORLD );
                MPI_Send ( & ( Dmat.pData[0] ),Dmat.nonzeros, MPI_DOUBLE,0,iam+3*size,MPI_COMM_WORLD );
                Dmat.clear();
		Btsparse.clear();
            }
            else {
                for ( i=1; i<size; ++i ) {
                    // The root process receives parts of Dmat sequentially from all processes and directly adds them together.
                    int nonzeroes, count;
                    MPI_Recv ( &nonzeroes,1,MPI_INT,i,i,MPI_COMM_WORLD,&status );
                    /*MPI_Get_count(&status, MPI_INT, &count);
                    printf("Process 0 received %d elements of process %d\n",count,i);*/
                    if(nonzeroes>0) {
                        printf("Nonzeroes : %d\n ",nonzeroes);
                        Dblock.allocate ( Dblocks * blocksize,Dblocks * blocksize,nonzeroes );
                        MPI_Recv ( & ( Dblock.pRows[0] ), Dblocks * blocksize + 1, MPI_INT,i,i+size,MPI_COMM_WORLD,&status );
                        /*MPI_Get_count(&status, MPI_INT, &count);
                        printf("Process 0 received %d elements of process %d\n",count,i);*/
                        MPI_Recv ( & ( Dblock.pCols[0] ),nonzeroes, MPI_INT,i,i+2*size,MPI_COMM_WORLD,&status );
                        /*MPI_Get_count(&status, MPI_INT, &count);
                        printf("Process 0 received %d elements of process %d\n",count,i);*/
                        MPI_Recv ( & ( Dblock.pData[0] ),nonzeroes, MPI_DOUBLE,i,i+3*size,MPI_COMM_WORLD,&status );
                        /*MPI_Get_count(&status, MPI_DOUBLE, &count);
                        printf("Process 0 received %d elements of process %d\n",count,i);*/
                        Dmat.addBCSR ( Dblock );
			Dblock.clear();
                    }
                }
                //Dmat.writeToFile("D_sparse.csr");
                Dmat.reduceSymmetric();
                Btsparse.transposeIt(1);
                create2x2SymBlockMatrix(Asparse,Btsparse, Dmat, Csparse);
                Btsparse.clear();
                Dmat.clear();
                Csparse.writeToFile(filenameC);
                Csparse.clear();
                if(filenameC != NULL)
                    free(filenameC);
                filenameC=NULL;
            }
        }
        
        blacs_barrier_(&ICTXT2D,"A");

        //AB_sol will contain the solution of A*X=B, distributed across the process rows. Processes in the same process row possess the same part of AB_sol
        DESCAB_sol= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
        if ( DESCAB_sol==NULL ) {
            printf ( "unable to allocate memory for descriptor for AB_sol\n" );
            return -1;
        }
        //AB_sol (Adim, Ddim) is distributed across all processes in ICTXT2D starting from process (0,0) into blocks of size (Adim, blocksize)
        descinit_ ( DESCAB_sol, &Adim, &Ddim, &Adim, &blocksize, &i_zero, &i_zero, &ICTXT2D, &Adim, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of matrix C returns info: %d\n",info );
            return info;
        }

        AB_sol=(double *) calloc(Adim * Dcols*blocksize,sizeof(double));

        // Each process calculates the Schur complement of the part of D at its disposal. (see src/schur.cpp)
        // The solution of A * X = B_j is stored in AB_sol (= A^-1 * B_j)
        make_Sij_parallel_denseB ( Asparse, BT_i, B_j, D, lld_D, AB_sol );
	
	if(iam !=0)
	  Asparse.clear();
	
	BT_i.clear();
	B_j.clear();

        blacs_barrier_ ( &ICTXT2D,"ALL" );

        //The Schur complement is factorised (by ScaLAPACK)
        pdpotrf_ ( "U",&Ddim,D,&i_one,&i_one,DESCD,&info );
        if ( info != 0 ) {
            printf ( "Cholesky decomposition of D was unsuccessful, error returned: %d\n",info );
            return -1;
        }

        blacs_barrier_ ( &ICTXT2D,"ALL" );

        //The Schur complement is inverteded (by ScaLAPACK)
        pdpotri_ ( "U",&Ddim,D,&i_one,&i_one,DESCD,&info );
        if ( info != 0 ) {
            printf ( "Inverse of D was unsuccessful, error returned: %d\n",info );
            return -1;
        }

        blacs_barrier_(&ICTXT2D,"A");

        InvD_T_Block = ( double* ) calloc ( Dblocks * blocksize + Adim ,sizeof ( double ) );

        //Diagonal elements of the (1,1) block of C^-1 are still distributed and here they are gathered in InvD_T_Block in the root process.
        if(*position == pcol) {
            for (i=0; i<Ddim; ++i) {
                if (pcol == (i/blocksize) % *dims) {
                    int Dpos = i%blocksize + ((i/blocksize) / *dims) * blocksize ;
                    *(InvD_T_Block + Adim +i) = *( D + Dpos + lld_D * Dpos);
                }
            }
            for ( i=0,j=0; i<Dblocks; ++i,++j ) {
                if ( j==*dims )
                    j=0;
                if ( *position==j ) {
                    dgesd2d_ ( &ICTXT2D,&blocksize,&i_one,InvD_T_Block + Adim + i * blocksize,&blocksize,&i_zero,&i_zero );
                }
                if ( *position==0 ) {
                    dgerv2d_ ( &ICTXT2D,&blocksize,&i_one,InvD_T_Block + Adim + blocksize*i,&blocksize,&j,&j );
                }
            }
        }
        
        if(position != NULL){
	  free(position);
	  position=NULL;
	}
	if(dims != NULL){
	  free(dims);
	  dims=NULL;
	}

        //Only the root process performs a selected inversion of A.
        if (iam==0) {

            int pardiso_message_level = 1;

            int pardiso_mtype=-2;

            ParDiSO pardiso ( pardiso_mtype, pardiso_message_level );
            int number_of_processors = 1;
            char* var = getenv("OMP_NUM_THREADS");
            if(var != NULL)
                sscanf( var, "%d", &number_of_processors );
            else {
                printf("Set environment OMP_NUM_THREADS to 1");
                exit(1);
            }

            pardiso.iparm[2]  = 2;
            pardiso.iparm[3]  = number_of_processors;
            pardiso.iparm[8]  = 0;
            pardiso.iparm[11] = 1;
            pardiso.iparm[13]  = 0;
            pardiso.iparm[28]  = 0;

            //This function calculates the factorisation of A once again so this might be optimized.
            pardiso.findInverseOfA ( Asparse );

            printf("Processor %d inverted matrix A\n",iam);
        }
        blacs_barrier_(&ICTXT2D,"A");

        // To minimize memory usage, and because only the diagonal elements of the inverse are needed, X' * S is calculated row by row
        // the diagonal element is calculated as the dot product of this row and the corresponding column of X. (X is solution of AX=B)
        XSrow= ( double* ) calloc ( Dcols * blocksize,sizeof ( double ) );
        DESCXSROW= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
        if ( DESCXSROW==NULL ) {
            printf ( "unable to allocate memory for descriptor for AB_sol\n" );
            return -1;
        }
        //XSrow (1,Ddim) is distributed acrros processes of ICTXT2D starting from process (0,0) into blocks of size (1,blocksize)
        descinit_ ( DESCXSROW, &i_one, &Ddim, &i_one,&blocksize, &i_zero, &i_zero, &ICTXT2D, &i_one, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of matrix C returns info: %d\n",info );
            return info;
        }

        blacs_barrier_(&ICTXT2D,"A");

        //Calculating diagonal elements 1 by 1 of the (0,0)-block of C^-1.
        for (i=1; i<=Adim; ++i) {
            pdsymm_ ("R","U",&i_one,&Ddim,&d_one,D,&i_one,&i_one,DESCD,AB_sol,&i,&i_one,DESCAB_sol,&d_zero,XSrow,&i_one,&i_one,DESCXSROW);
            pddot_(&Ddim,InvD_T_Block+i-1,AB_sol,&i,&i_one,DESCAB_sol,&Adim,XSrow,&i_one,&i_one,DESCXSROW,&i_one);
        }
        blacs_barrier_(&ICTXT2D,"A");
	
	if(D!=NULL){
	  free(D);
	  D=NULL;
	}
	if(AB_sol!=NULL){
	  free(AB_sol);
	  AB_sol=NULL;
	}
	if(XSrow !=NULL){
	  free(XSrow);
	  XSrow=NULL;
	}
	if(DESCD!=NULL){
	  free(DESCD);
	  DESCD=NULL;
	}
	if(DESCAB_sol!=NULL){
	  free(DESCAB_sol);
	  DESCAB_sol=NULL;
	}
	if(DESCXSROW!=NULL){
	  free(DESCXSROW);
	  DESCXSROW=NULL;
	}

        //Only in the root process we add the diagonal elements of A^-1
        if (iam ==0) {
            for(i=0; i<Adim; ++i) {
                j=Asparse.pRows[i];
                *(InvD_T_Block+i) += Asparse.pData[j];
            }
            Asparse.clear();
            printdense ( Adim+Ddim,1,InvD_T_Block,"diag_inverse_C_parallel.txt" );
        }
        
        if(InvD_T_Block !=NULL){
	  free(InvD_T_Block);
	  InvD_T_Block=NULL;
	}

        blacs_barrier_(&ICTXT2D,"A");
	blacs_gridexit_(&ICTXT2D);
    }
    //cout << iam << " reached end before MPI_Barrier" << endl;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}
