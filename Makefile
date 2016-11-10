
all:
	make -C src

clean:
	rm -rf src/*.o src/train_g2p src/apply_g2p
