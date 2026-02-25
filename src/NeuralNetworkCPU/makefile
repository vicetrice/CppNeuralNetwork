CXX = g++
CXXFLAGS = -Wall -Werror -Wextra -std=c++17

SRCS = main.cpp neuralNetwork.cpp neuronLayer.cpp mnist_images.cpp utils.cpp
OBJS = $(SRCS:.cpp=.o)

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
	$(CXX) $(CXXFLAGS) -o $(TARGET)$(EXE) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJS)

cleanall:
	$(RM) $(OBJS) $(TARGET)$(EXE)
