/* Hand-written linker script for STM32H753ZI (NUCLEO-H753ZI), single-core
 * Cortex-M7. CMSIS's cmsis-device-h7 pack ships ST linker templates only for
 * the dual-core H745/747/755/757 parts, so a hand-written script is the only
 * option for the single-core H753.
 *
 * Layout choice for an RT-determinism, allocation-free proof: place
 * .data/.bss/stack and the small bounded heap in DTCM (zero-wait-state,
 * CPU-local, no AXI/cache arbitration jitter). Code + read-only data live in
 * flash. The heap is deliberately tiny and fixed-ceilinged (_eheap): the
 * fixed-N solve() loop must never allocate, and the blindness canary's single
 * deliberate allocation is the only heap traffic expected.
 *
 * The symbol names below (_estack, _sdata/_edata/_sidata, _sbss/_ebss) are
 * dictated verbatim by Source/Templates/gcc/startup_stm32h753xx.s from the
 * cmsis-device-h7 pack; end/_end and _eheap are consumed by _sbrk in
 * wrap_malloc.cpp. Renaming any of them silently breaks the startup
 * copy/zero loops or the heap bound.
 */

MEMORY
{
  FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 2048K
  DTCM  (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
}

ENTRY(Reset_Handler)

/* Initial stack pointer = top of DTCM; the stack grows down from here. */
_estack = ORIGIN(DTCM) + LENGTH(DTCM);

/* Minimum sizes reserved and checked below. The heap is intentionally small:
 * the only heap allocation expected is the mandatory blindness canary. */
_Min_Heap_Size  = 0x6000;   /* 24 KiB: the RT claim is zero allocs PER STEP, not
                               zero total heap -- each policy's one-time setup
                               legitimately uses the heap (measured on-device
                               peak: 13136 bytes across the four policies).
                               _sbrk still fails past _eheap as a runaway guard;
                               the high-water reports the real peak. */
_Min_Stack_Size = 0x10000;  /* 64 KiB main stack reserve: must hold both the
                               ~19888-byte worst-case solve() call frame AND
                               kraft_slsqp's large Eigen stack temporary (kept
                               off the heap by EIGEN_STACK_ALLOCATION_LIMIT).
                               DTCM budget: 24K heap + 64K stack + bss << 128K. */

SECTIONS
{
  /* Vector table must be first in flash (CPU reads _estack + Reset_Handler
   * from the first two words at 0x08000000 at reset). */
  .isr_vector :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector))
    . = ALIGN(4);
  } >FLASH

  .text :
  {
    . = ALIGN(4);
    *(.text)
    *(.text*)
    *(.glue_7)
    *(.glue_7t)
    *(.eh_frame)
    KEEP(*(.init))
    KEEP(*(.fini))
    . = ALIGN(4);
    _etext = .;
  } >FLASH

  .rodata :
  {
    . = ALIGN(4);
    *(.rodata)
    *(.rodata*)
    . = ALIGN(4);
  } >FLASH

  .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } >FLASH
  .ARM :
  {
    __exidx_start = .;
    *(.ARM.exidx*)
    __exidx_end = .;
  } >FLASH

  .preinit_array :
  {
    PROVIDE_HIDDEN(__preinit_array_start = .);
    KEEP(*(.preinit_array*))
    PROVIDE_HIDDEN(__preinit_array_end = .);
  } >FLASH
  .init_array :
  {
    PROVIDE_HIDDEN(__init_array_start = .);
    KEEP(*(SORT(.init_array.*)))
    KEEP(*(.init_array*))
    PROVIDE_HIDDEN(__init_array_end = .);
  } >FLASH
  .fini_array :
  {
    PROVIDE_HIDDEN(__fini_array_start = .);
    KEEP(*(SORT(.fini_array.*)))
    KEEP(*(.fini_array*))
    PROVIDE_HIDDEN(__fini_array_end = .);
  } >FLASH

  /* .data: loaded from flash (LMA), resident in DTCM (VMA); startup copies it
   * from _sidata to [_sdata,_edata). */
  _sidata = LOADADDR(.data);
  .data :
  {
    . = ALIGN(4);
    _sdata = .;
    *(.data)
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } >DTCM AT>FLASH

  /* .bss: zeroed by startup over [_sbss,_ebss). */
  .bss :
  {
    . = ALIGN(4);
    _sbss = .;
    __bss_start__ = _sbss;
    *(.bss)
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
    __bss_end__ = _ebss;
  } >DTCM

  /* Heap + stack reservation. end/_end mark the heap start (consumed by
   * _sbrk); _eheap is the fixed heap ceiling the instrument enforces. The
   * stack reserve below it guarantees the linker errors out if DTCM cannot
   * hold data+bss+heap+stack. */
  ._user_heap_stack :
  {
    . = ALIGN(8);
    PROVIDE(end = .);
    PROVIDE(_end = .);
    . = . + _Min_Heap_Size;
    _eheap = .;
    . = . + _Min_Stack_Size;
    . = ALIGN(8);
  } >DTCM

  /DISCARD/ : { libc.a(*) libm.a(*) libgcc.a(*) }

  .ARM.attributes 0 : { *(.ARM.attributes) }
}
