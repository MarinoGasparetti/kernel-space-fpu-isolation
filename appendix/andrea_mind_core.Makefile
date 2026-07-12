obj-m += andrea_mind_core.o

ifeq ($(ARCH),x86_64)
CFLAGS_andrea_mind_core.o += -msse -msse2
endif
ifeq ($(ARCH),arm64)
CFLAGS_REMOVE_andrea_mind_core.o += -mgeneral-regs-only
endif

KDIR_X86_64 ?= /work/buildroot/output/build/linux-6.1.24
KDIR_ARM64 ?= /work/output-arm64/build/linux-6.1.24
HOST_BIN_X86_64 ?= /work/buildroot/output/host/bin
HOST_BIN_ARM64 ?= /work/output-arm64/host/bin

all: x86_64 arm64

x86_64:
	PATH="$(HOST_BIN_X86_64):$$PATH" $(MAKE) -C $(KDIR_X86_64) M=$(PWD) ARCH=x86_64 CROSS_COMPILE=x86_64-buildroot-linux-gnu- modules

arm64:
	PATH="$(HOST_BIN_ARM64):$$PATH" $(MAKE) -C $(KDIR_ARM64) M=$(PWD) ARCH=arm64 CROSS_COMPILE=aarch64-buildroot-linux-gnu- modules

clean:
	rm -f *.o *.ko *.mod *.mod.c *.symvers *.order
