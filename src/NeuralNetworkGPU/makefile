NVCC = nvcc
NVCCFLAGS = -std=c++17 -O2

SRCS_CU  = neuronLayerGPU.cu neuralNetworkGPU.cu main.cu
SRCS_CPP = mnist_images.cpp
OBJS_CU  = $(SRCS_CU:.cu=.obj)
OBJS_CPP = $(SRCS_CPP:.cpp=.obj)
OBJS     = $(OBJS_CU) $(OBJS_CPP)

TARGET = main

ifeq ($(OS),Windows_NT)
    RM = del /Q
    EXE = .exe
else
    RM = rm -f
    EXE =
endif

all: $(TARGET)$(EXE)

$(TARGET)$(EXE): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $(TARGET)$(EXE)

%.obj: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.obj: %.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	-$(RM) $(OBJS)

cleanall: clean
	-$(RM) $(TARGET)$(EXE)