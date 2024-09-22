CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -pedantic -lm
DEPS = diffusion.h tensor.h normalization.h unet_blocks.h
OBJ = diffusion.o main.o tensor.o normalization.o unet_blocks.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

unet_test: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -f *.o unet_test

