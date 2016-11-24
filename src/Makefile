CXX = g++
CXXFLAGS = -O3 -std=c++11 -pthread 

all: train_g2p apply_g2p

train_g2p: g2p_io.o g2p_utils.o g2p_model.o train_g2p.cc
	$(CXX) $(CXXFLAGS) $^ -o $@

apply_g2p: g2p_io.o g2p_utils.o g2p_model.o apply_g2p.cc
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -rf *.o train_g2p apply_g2p
