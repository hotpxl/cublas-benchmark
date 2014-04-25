a.out: main.cpp image-mat.o image-mat.h
	g++ main.cpp image-mat.o \
	  -I/usr/local/cuda/include \
	  -L/usr/local/cuda/lib64 \
	  -lcublas \
	  -lcudart

image-mat.o: image-mat.cu
	/usr/local/cuda/bin/nvcc -c image-mat.cu \
	  -arch sm_35
