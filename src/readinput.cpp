#include <stdio.h>
#include <stdlib.h>
#include <iosfwd>
#include <string>
#include "shared_var.h"
#include <iostream>
#include <fstream>
using namespace std;

int read_input ( char * filename ) {
    std::ifstream inputfile ( filename );
    string line;
    bool dense_bool=false, sparse_bool=false, Afile_bool=false, Dfile_bool=false, Bfile_bool=false;
    bool blocksize_bool=false;
    int Bassparse_bool;

    filenameD= ( char* ) calloc ( 100,sizeof ( char ) );
    filenameA= ( char* ) calloc ( 100,sizeof ( char ) );
    filenameB= ( char* ) calloc ( 100,sizeof ( char ) );
    filenameC= ( char* ) calloc ( 100,sizeof ( char ) );

    lambda=0;
    blocksize=64;
    printsparseC_bool=false;
    Bassparse_bool=0;


    while ( std::getline ( inputfile,line ) ) {
        if ( line=="#DimensionD" ) {
            std::getline ( inputfile,line );
            Ddim=atoi ( line.c_str() );
            dense_bool=true;
        } else if ( line=="#DimensionA" ) {
            std::getline ( inputfile,line );
            Adim=atoi ( line.c_str() );
            sparse_bool=true;
        } else if ( line=="#SparseFileA" ) {
            std::getline ( inputfile,line );
            line.copy ( filenameA,100 );
            Afile_bool=true;
        } else if ( line=="#DenseFileD" ) {
            std::getline ( inputfile,line );
            line.copy ( filenameD,100 );
            Dfile_bool=true;
        } else if ( line=="#SparseFileB" ) {
            std::getline ( inputfile,line );
            line.copy ( filenameB,100 );
            Bfile_bool=true;
        } else if ( line=="#BlockSize" ) {
            std::getline ( inputfile,line );
            blocksize=atoi ( line.c_str() );
            blocksize_bool=true;
	    } else if ( line=="#Bsparse" ) {
            std::getline ( inputfile,line );
            Bassparse_bool=atoi ( line.c_str() );
        } else if ( line=="#OutputFileSparseC" ) {
            std::getline ( inputfile,line );
            line.copy ( filenameC,100 );;
            printsparseC_bool=true;
        } else if ( line[0]=='/' || line.size() ==0 ) {}
        else {
            printf ( "Unknown parameter in inputfile, the following line was ignored: \n" );
            printf ( "%s\n",line.c_str() );
        }
    }
        if ( dense_bool ) {
            if ( sparse_bool ) {
                if ( Dfile_bool ) {
                    if ( Afile_bool ) {
                            if ( Bfile_bool ) {
                                if ( *position==0 && * ( position+1 ) ==0 ) {
                                    printf ( "Dimension of dense matrix:   \t %d\n", Ddim );
                                    printf ( "Dimension of sparse matrix:  \t %d\n", Adim );
                                    printf ( "filename of dense matrix:    \t %s\n", filenameD );
                                    printf ( "filename of sparse matrix A: \t %s\n", filenameA );
                                    printf ( "filename of sparse matrix B: \t %s\n", filenameB );
                                }
                            } else {
                                printf ( "ERROR: filename of matrix B was not in input file or not read correctly\n" );
                                return -1;
                            }
                    } else {
                        printf ( "ERROR: filename of matrix A was not in input file or not read correctly\n" );
                        return -1;
                    }
                } else {
                    printf ( "ERROR: filename of dense matrix D was not in input file or not read correctly\n" );
                    return -1;
                }
            } else {
                printf ( "ERROR: dimension of sparse matrix A was not in input file or not read correctly\n" );
                return -1;
            }
        } else {
            printf ( "ERROR: dimension of dense matrix D was not in input file or not read correctly\n" );
            return -1;
        }
    if ( *position==0 && * ( position+1 ) ==0 ) {
        if ( blocksize_bool ) {
            printf ( "Blocksize of %d was used to distribute matrices across processes\n", blocksize );
        } else {
            printf ( "Default blocksize of %d was used to distribute matrices across processes\n", blocksize );
        }
        if ( printsparseC_bool )
	  printf("Sparse matrix C will be written in CSR format to text file %s \n", filenameC);
	if ( Bassparse_bool )
	  printf("B will be treated as a sparse matrix \n");
	else
	  printf("B will be treated as a dense matrix\n");
    }
    return 0;
}
