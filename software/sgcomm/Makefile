CC=gcc
OPTIM=-O3

# set include directory for external dependencies
ifdef EXTERN_I
EXTERN_INCLUDES=-I$(EXTERN_I)
else
EXTERN_INCLUDES=
endif

# set path for external library dependencies
ifdef EXTERN_L
EXTERN_LINKS=-L$(EXTERN_L)
else
EXTERN_LINKS=-L./
endif

CFLAGS=-std=gnu99 -fPIC $(OPTIM) -g $(EXTERN_INCLUDES)
LIBS=-lrt -lpthread -lscatgat
LFLAGS=$(LIBS) $(EXTERN_LINKS)
DEPS=sgcomm_report.o sgcomm_net.o sgcomm_threads.o
EXEC_TRANSCEIVER=sgtxrx
OBJS_TRANSCEIVER=sgcomm_tx_rx_test.o $(DEPS)

EXEC_TRANSMITTER=sgtx
OBJS_TRANSMITTER=sgcomm_tx_test.o $(DEPS)

EXEC_RECEIVER=sgrx
OBJS_RECEIVER=sgcomm_rx_test.o $(DEPS)

$(EXEC_TRANSCEIVER): $(OBJS_TRANSCEIVER)
	$(CC) -o $@ $^ $(LFLAGS)

$(EXEC_TRANSMITTER): $(OBJS_TRANSMITTER)
	$(CC) -o $@ $^ $(LFLAGS)

$(EXEC_RECEIVER): $(OBJS_RECEIVER)
	$(CC) -o $@ $^ $(LFLAGS)

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY : all
all : $(EXEC_TRANSCEIVER) $(EXEC_RECEIVER) $(EXEC_TRANSMITTER)

.PHONY: clean
clean:
	rm -f *.o $(EXEC_TRANSCEIVER) $(EXEC_TRANSMITTER) $(EXEC_RECEIVER)
