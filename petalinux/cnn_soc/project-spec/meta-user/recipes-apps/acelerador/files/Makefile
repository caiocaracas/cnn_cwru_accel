APP = acelerador
SRCS = acelerador.c mapa_linux.c inferencia_sw.c fluxo_sw.c pacote.c
OBJS = $(SRCS:.c=.o)

CFLAGS  += -std=c99 -Wall -Wextra -O3 -ftree-vectorize
LDLIBS  += -lm

all: $(APP)
$(APP): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<
clean:
	rm -f $(APP) $(OBJS)
