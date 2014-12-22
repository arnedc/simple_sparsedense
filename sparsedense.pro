TEMPLATE	= app
LANGUAGE	= C++

LIBS	        = -L${HOME}/Libraries/linuxAMD64 -lpardiso500-GNU472-X86-64
LIBS           += -L/opt/intel/mkl/lib/intel64 \
                  -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lmkl_blacs_intelmpi_lp64 \
                  -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lmkl_blacs_intelmpi_lp64 \
                  -lmkl_blas95_lp64 -lmkl_scalapack_lp64 -fopenmp -lgfortran

CCX              = /opt/intel/impi/5.0.2.044/intel64/bin/mpicxx
QMAKE_CXX        = $$CCX
QMAKE_CC         = $$CCX
QMAKE_LINK       = $$CCX
QMAKE_LFLAGS     = -dynamic

DEFINES          = 

INCLUDEPATH	+= src 


CONFIG  -= qt
CONFIG  += debug
SOURCES += src/CSRdouble.cpp   \
	   src/CSRcomplex.cpp  \
	   src/RealMath.cpp    \
	   src/ParDiSO.cpp     \
	   src/tools.cpp       \
	   src/schur.cpp       \
	   src/IO.cpp          \
	   src/readinput.cpp   \
	   src/readdist.cpp    \
	   src/main.cpp        \

DESTDIR     = bin
OBJECTS_DIR = obj


unix {
  OBJECTS_DIR = obj
}



