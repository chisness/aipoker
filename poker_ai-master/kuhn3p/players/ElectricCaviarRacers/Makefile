CC = gcc
CFLAGS = -O3 -Wall

PROGRAMS = electric_caviar

all: $(PROGRAMS)

clean:
	rm -f $(PROGRAMS)

electric_caviar: game.c game.h evalHandTables rng.c rng.h electric_caviar.c net.c net.h
	$(CC) $(CFLAGS) -o $@ game.c rng.c electric_caviar.c net.c
