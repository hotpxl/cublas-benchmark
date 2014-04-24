a.out: main.cpp
	g++ main.cpp \
	  -I/usr/local/cuda/include \
	  -L/usr/local/cuda/lib64 \
	  -lcublas \
	  -lcudart

image-mat.o: image-mat.cu
	/usr/local/cuda/bin/nvcc -c image-mat.cu
