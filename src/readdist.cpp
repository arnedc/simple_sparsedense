#include <stdio.h>
#include <stdlib.h>
#include "shared_var.h"
#include "RealMath.hpp"
#include "CSRdouble.hpp"
#include <cassert>

extern "C" {
    int MPI_Get_count(MPI_Status *, MPI_Datatype, int *);
    void blacs_barrier_ ( int*, char* );
    int blacs_pnum_ ( int *ConTxt, int *prow, int *pcol );
}

int read_in_BD ( int * DESCD, double * Dmat, CSRdouble& BT_i, CSRdouble& B_j, CSRdouble& Btsparse ) {

    FILE *fD;
    int ni, i,j, info;
    int nstrips, pcol, colcur,rowcur;

    MPI_Status status;
    
    pcol= * ( position+1 );

    fD=fopen ( filenameD,"rb" );
    if ( fD==NULL ) {
        printf ( "Error opening file\n" );
        return -1;
    }
    
    nstrips= Ddim % ( blocksize * * ( dims+1 ) ) ==0 ?  Ddim / ( blocksize * * ( dims+1 ) ) : ( Ddim / ( blocksize * * ( dims+1 ) ) ) +1;

    // Set up of matrix D and B per strip of T'

    for ( ni=0; ni< nstrips; ++ni ) {
       if ( ( Dblocks-1 ) % *dims == *position && Ddim%blocksize !=0 ) {
            if ( ni==0 ) {
                info=fseek ( fD, ( long ) ( pcol * blocksize * ( Ddim ) * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading D file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            } else {
                info=fseek ( fD, ( long ) ( blocksize * ( * ( dims+1 )-1 ) * ( Ddim ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading D file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<blocksize; ++i ) {
                info=fseek ( fD, ( long ) ( blocksize * *position * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading D file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
                for ( j=0; j < Drows-1; ++j ) {
                    fread ( Dmat + i*Drows*blocksize + j*blocksize + ni*blocksize*Drows*blocksize,sizeof ( double ),blocksize,fD );
                    info=fseek ( fD, ( long ) ( ( ( *dims ) -1 ) * blocksize * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading D file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                }
                fread ( Dmat + i*Drows*blocksize + j*blocksize + ni*blocksize*Drows*blocksize,sizeof ( double ),Ddim%blocksize,fD );
            }
            //Normal read-in of the strips of T from a binary file (each time blocksize elements are read in)
        } else {
            if ( ni==0 ) {
                info=fseek ( fD, ( long ) ( pcol * blocksize * ( Ddim ) * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading D file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            } else {
                info=fseek ( fD, ( long ) ( blocksize * ( * ( dims+1 )-1 ) * ( Ddim ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading D file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<blocksize; ++i ) {
                info=fseek ( fD, ( long ) ( blocksize * *position * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading D file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
                for ( j=0; j < Drows-1; ++j ) {
                    fread ( Dmat + i*Drows*blocksize + j*blocksize + ni*blocksize*Drows*blocksize,sizeof ( double ),blocksize,fD );
                    info=fseek ( fD, ( long ) ( ( * ( dims )-1 ) * blocksize * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                }
                fread ( Dmat + i*Drows*blocksize + j*blocksize + ni*blocksize*Drows*blocksize,sizeof ( double ),blocksize,fD );
                info=fseek ( fD, ( long ) ( ( Ddim - blocksize * ( ( Drows-1 ) * *dims + *position +1 ) ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
        }

        blacs_barrier_ ( &ICTXT2D,"A" );
    }
    
    fclose(fD);

        // End of read-in

        // Matrix B 
    if ( iam!=0 ) {
       
        // Each process receives the necessary BT_i and B_j
        // Blocking sends are used, which is why the order of the receives is critical depending on the coordinates of the process
        int nonzeroes;
        if (*position >= pcol) {
            MPI_Recv ( &nonzeroes,1,MPI_INT,0,iam,MPI_COMM_WORLD,&status );
            BT_i.allocate ( blocksize*Drows,Adim,nonzeroes );
            MPI_Recv ( & ( BT_i.pRows[0] ),blocksize*Drows + 1, MPI_INT,0,iam + size,MPI_COMM_WORLD,&status );
            int count;
            MPI_Get_count(&status,MPI_INT,&count);
            BT_i.nrows=count-1;
            MPI_Recv ( & ( BT_i.pCols[0] ),nonzeroes, MPI_INT,0,iam+2*size,MPI_COMM_WORLD,&status );
            MPI_Recv ( & ( BT_i.pData[0] ),nonzeroes, MPI_DOUBLE,0,iam+3*size,MPI_COMM_WORLD,&status );

            MPI_Recv ( &nonzeroes,1, MPI_INT,0,iam+4*size,MPI_COMM_WORLD,&status );

            B_j.allocate ( blocksize*Dcols,Adim,nonzeroes );

            MPI_Recv ( & ( B_j.pRows[0] ),blocksize*Dcols + 1, MPI_INT,0,iam + 5*size,MPI_COMM_WORLD,&status );
            MPI_Get_count(&status,MPI_INT,&count);
            B_j.nrows=count-1;
            MPI_Recv ( & ( B_j.pCols[0] ),nonzeroes, MPI_INT,0,iam+6*size,MPI_COMM_WORLD,&status );
            MPI_Recv ( & ( B_j.pData[0] ),nonzeroes, MPI_DOUBLE,0,iam+7*size,MPI_COMM_WORLD,&status );

            //Actually BT_j is sent, so it still needs to be transposed
            B_j.transposeIt ( 1 );
        }
        else {
            MPI_Recv ( &nonzeroes,1, MPI_INT,0,iam+4*size,MPI_COMM_WORLD,&status );

            B_j.allocate ( blocksize*Dcols,Adim,nonzeroes );

            MPI_Recv ( & ( B_j.pRows[0] ),blocksize*Dcols + 1, MPI_INT,0,iam + 5*size,MPI_COMM_WORLD,&status );
            int count;
            MPI_Get_count(&status,MPI_INT,&count);
            B_j.nrows=count-1;

            MPI_Recv ( & ( B_j.pCols[0] ),nonzeroes, MPI_INT,0,iam+6*size,MPI_COMM_WORLD,&status );

            MPI_Recv ( & ( B_j.pData[0] ),nonzeroes, MPI_DOUBLE,0,iam+7*size,MPI_COMM_WORLD,&status );

            B_j.transposeIt ( 1 );

            MPI_Recv ( &nonzeroes,1,MPI_INT,0,iam,MPI_COMM_WORLD,&status );
            BT_i.allocate ( blocksize*Drows,Adim,nonzeroes );
            MPI_Recv ( & ( BT_i.pRows[0] ),blocksize*Drows + 1, MPI_INT,0,iam + size,MPI_COMM_WORLD,&status );
            MPI_Get_count(&status,MPI_INT,&count);
            BT_i.nrows=count-1;
            MPI_Recv ( & ( BT_i.pCols[0] ),nonzeroes, MPI_INT,0,iam+2*size,MPI_COMM_WORLD,&status );
            MPI_Recv ( & ( BT_i.pData[0] ),nonzeroes, MPI_DOUBLE,0,iam+3*size,MPI_COMM_WORLD,&status );
        }
    }
    else {
            
        Btsparse.loadFromFile(filenameB);
	assert(Btsparse.nrows == Adim);
	assert(Btsparse.ncols == Ddim);
	Btsparse.transposeIt(1);

        // For each process row i BT_i is created which is also sent to processes in column i to become B_j.
        for ( int rowproc= *dims - 1; rowproc>= 0; --rowproc ) {
            BT_i.ncols=Btsparse.ncols;
            BT_i.nrows=0;
            BT_i.nonzeros=0;
            int Drows_rowproc;
            if (rowproc!=0) {
                Drows_rowproc= ( Dblocks - rowproc ) % *dims == 0 ? ( Dblocks- rowproc ) / *dims : ( Dblocks- rowproc ) / *dims +1;
                Drows_rowproc= Drows_rowproc<1? 1 : Drows_rowproc;
            }
            else
                Drows_rowproc=Drows;
            for ( i=0; i<Drows_rowproc; ++i ) {
                //Each process in row i can hold several blocks of contiguous rows of D for which we need the corresponding rows of B_T
                // Therefore we use the function extendrows to create BT_i (see src/tools.cpp)
                BT_i.extendrows ( Btsparse, ( i * *dims + rowproc ) * blocksize,blocksize );
            }
            for ( int colproc= ( rowproc==0 ? 1 : 0 ); colproc < * ( dims+1 ); ++colproc ) {
                int *curpos, rankproc;
                rankproc= blacs_pnum_ (&ICTXT2D, &rowproc,&colproc);

                MPI_Send ( & ( BT_i.nonzeros ),1, MPI_INT,rankproc,rankproc,MPI_COMM_WORLD );
                MPI_Send ( & ( BT_i.pRows[0] ),BT_i.nrows + 1, MPI_INT,rankproc,rankproc+size,MPI_COMM_WORLD );
                MPI_Send ( & ( BT_i.pCols[0] ),BT_i.nonzeros, MPI_INT,rankproc,rankproc+2*size,MPI_COMM_WORLD );
                MPI_Send ( & ( BT_i.pData[0] ),BT_i.nonzeros, MPI_DOUBLE,rankproc,rankproc+3*size,MPI_COMM_WORLD );

                //printf("BT_i's sent to processor %d\n",rankproc);

                rankproc= blacs_pnum_ (&ICTXT2D, &colproc,&rowproc);
                MPI_Send ( & ( BT_i.nonzeros ),1, MPI_INT,rankproc,rankproc+4*size,MPI_COMM_WORLD );
                MPI_Send ( & ( BT_i.pRows[0] ),BT_i.nrows + 1, MPI_INT,rankproc,rankproc+5*size,MPI_COMM_WORLD );
                MPI_Send ( & ( BT_i.pCols[0] ),BT_i.nonzeros, MPI_INT,rankproc,rankproc+6*size,MPI_COMM_WORLD );
                MPI_Send ( & ( BT_i.pData[0] ),BT_i.nonzeros, MPI_DOUBLE,rankproc,rankproc+7*size,MPI_COMM_WORLD );

                //printf("B_j's sent to processor %d\n",rankproc);
            }
        }
        B_j.make ( BT_i.nrows,BT_i.ncols,BT_i.nonzeros,BT_i.pRows,BT_i.pCols,BT_i.pData );
        B_j.transposeIt ( 1 );
    }
    return 0;
}
