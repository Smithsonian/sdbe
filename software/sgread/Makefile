CC=gcc
CFLAGS=
LFLAGS=-lrt -lpthread
EXEC_READ=sgtool
OBJS_READ=sgtool.o sg_access.o print_rusage.o
EXEC_WRITE=sgwrite
OBJS_WRITE=sgwrite.o

$(EXEC_READ): $(OBJS_READ)
	$(CC) -o $@ $^ $(LFLAGS)

$(EXEC_WRITE): $(OBJS_WRITE)
	$(CC) -o $@ $^ $(LFLAGS)
	
%.o: %.c
	$(CC) -c -o $@ $<

.PHONY: clean

clean:
	rm -f *.o $(EXEC_READ) $(EXEC_WRITE)
