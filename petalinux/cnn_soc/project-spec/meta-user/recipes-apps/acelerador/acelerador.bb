SUMMARY = "Orquestrador do acelerador CNN 1D na PL"
SECTION = "PETALINUX/apps"
LICENSE = "MIT"
FILES:${PN} += "${datadir}/acelerador"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = "file://acelerador.c \
           file://mapa_linux.c \
           file://inferencia_sw.c \
           file://fluxo_sw.c \
           file://pacote.c \
           file://pacote.h \
           file://entrada_ps.bin \
           file://pesos.h \
           file://Makefile \
          "
S = "${WORKDIR}"

do_compile() {
    oe_runmake
}
do_install() {
    install -d ${D}${bindir}
    install -m 0755 acelerador ${D}${bindir}
    install -d ${D}${datadir}/acelerador
    install -m 0644 ${UNPACKDIR}/entrada_ps.bin ${D}${datadir}/acelerador/ || \
        install -m 0644 entrada_ps.bin ${D}${datadir}/acelerador/
}
