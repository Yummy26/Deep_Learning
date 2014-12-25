G=g++
OBJECTS=utils.o
INCLUDE=-I./include/
BLASLIB=../lib-blas/libblas.a
CBLASLIB=../lib-blas/libcblas.a
GFORTRAN=gfortran

all: LogisticRegression RBM

logistic_sgd: logistic_sgd.cpp $(CBLASLIB) $(BLASLIB)
	$(G) -c -g logistic_sgd.cpp $(INCLUDE) -std=c++0x
	$(G) logistic_sgd.o $(CBLASLIB) $(BLASLIB) -o $@ -lgfortran -std=c++0x 

mlp: mlp.cpp $(CBLASLIB) $(BLASLIB)
	$(G) -c -g mlp.cpp $(INCLUDE) -std=c++0x
	$(G) mlp.o $(CBLASLIB) $(BLASLIB) -o $@ -lgfortran -std=c++0x 

testhd: testhd.cpp $(CBLASLIB) $(BLASLIB)
	$(G) -c -g testhd.cpp $(INCLUDE) -std=c++0x
	$(G) testhd.o $(CBLASLIB) $(BLASLIB) -o $@ -lgfortran -std=c++0x 
clean:
	rm -rf *.o logistic_sgd mlp testhd
