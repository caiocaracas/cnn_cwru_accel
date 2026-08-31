// mapeia enderecos fisicos no espaco do processo.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

volatile uint32_t *mapeia(uint32_t base, uint32_t tam)
{
    static int fd = -1;

    if (fd < 0) {
        fd = open("/dev/mem", O_RDWR | O_SYNC);
        if (fd < 0) {
            perror("/dev/mem");
            exit(1);
        }
    }

    long pg = sysconf(_SC_PAGESIZE);
    uint32_t base_pg = base & ~(uint32_t)(pg - 1);
    uint32_t desloc  = base - base_pg;

    void *p = mmap(NULL, tam + desloc, PROT_READ | PROT_WRITE, MAP_SHARED,
                   fd, base_pg);
    if (p == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    return (volatile uint32_t *)((char *)p + desloc);
}
