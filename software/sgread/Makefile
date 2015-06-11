CC=gcc
CFLAGS=
LFLAGS=-lrt
EXEC=sgtool
OBJS=sgtool.o sg_access.o

$(EXEC): $(OBJS)
	$(CC) -o $@ $^ $(LFLAGS)

%.o: %.c
	$(CC) -c -o $@ $<

.PHONY: clean

clean:
	rm -f *.o $(EXEC)
