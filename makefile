all:
	g++ main.cpp \
	  -I/usr/local/cuda/include \
	  -L/usr/local/cuda/lib64 \
	  -lcublas \
	  -lcudart
