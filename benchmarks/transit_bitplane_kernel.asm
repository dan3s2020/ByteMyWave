BITS 64
DEFAULT REL

; Transit direct bitplane kernel, Windows x64 ABI.
;
; Function:
;   int64_t transit_kernel(
;       const uint8_t *weight_planes,   ; RCX
;       const uint8_t *activation_planes, ; RDX
;       int32_t *out                    ; R8
;   );
;
; Fixed to the current real Ollama tensor:
;   rows = 640
;   cols = 2048
;   weight format = signed INT4 two's complement, 4 packed bitplanes
;   activation format = signed INT8 two's complement, 8 packed bitplanes
;
; Layout:
;   each weight plane = 640 * 2048 / 8 = 163840 bytes = 0x28000
;   each activation plane = 2048 / 8 = 256 bytes = 0x100
;
; Math performed exactly:
;   q = b0 + 2*b1 + 4*b2 - 8*b3
;   x = a0 + 2*a1 + 4*a2 + 8*a3 + 16*a4 + 32*a5 + 64*a6 - 128*a7
;
; For every row:
;   dot(q, x) = sum_{i,j} coeffW[i]*coeffX[j]*popcnt(W_i & X_j)
;
; No general multiply is used in the matrix datapath.
; The kernel uses AND, POPCNT, shifts, adds/subtracts.

%define WPLANE_SIZE 0x28000
%define XPLANE_SIZE 0x100
%define ROW_BYTES   0x100
%define ROWS        640

; Compute signed INT8 sum selected by mask in R10.
; X planes are at RSI, current 64-bit chunk offset in R11.
; Returns signed sum in RCX.
%macro DOT_X_SIGNED8 0
    ; bit 0: +1
    mov     rdx, r10
    and     rdx, [rsi + r11 + 0*XPLANE_SIZE]
    popcnt  rdx, rdx
    mov     rcx, rdx

    ; bit 1: +2
    mov     rdx, r10
    and     rdx, [rsi + r11 + 1*XPLANE_SIZE]
    popcnt  rdx, rdx
    shl     rdx, 1
    add     rcx, rdx

    ; bit 2: +4
    mov     rdx, r10
    and     rdx, [rsi + r11 + 2*XPLANE_SIZE]
    popcnt  rdx, rdx
    shl     rdx, 2
    add     rcx, rdx

    ; bit 3: +8
    mov     rdx, r10
    and     rdx, [rsi + r11 + 3*XPLANE_SIZE]
    popcnt  rdx, rdx
    shl     rdx, 3
    add     rcx, rdx

    ; bit 4: +16
    mov     rdx, r10
    and     rdx, [rsi + r11 + 4*XPLANE_SIZE]
    popcnt  rdx, rdx
    shl     rdx, 4
    add     rcx, rdx

    ; bit 5: +32
    mov     rdx, r10
    and     rdx, [rsi + r11 + 5*XPLANE_SIZE]
    popcnt  rdx, rdx
    shl     rdx, 5
    add     rcx, rdx

    ; bit 6: +64
    mov     rdx, r10
    and     rdx, [rsi + r11 + 6*XPLANE_SIZE]
    popcnt  rdx, rdx
    shl     rdx, 6
    add     rcx, rdx

    ; bit 7: -128
    mov     rdx, r10
    and     rdx, [rsi + r11 + 7*XPLANE_SIZE]
    popcnt  rdx, rdx
    shl     rdx, 7
    sub     rcx, rdx
%endmacro

section .text
global transit_kernel

transit_kernel:
    ; Preserve Win64 nonvolatile registers.
    push    rbx
    push    rsi
    push    rdi
    push    r12
    push    r13
    push    r14
    push    r15

    ; Four weight-plane row pointers.
    mov     r12, rcx
    lea     r13, [rcx + WPLANE_SIZE]
    lea     r14, [rcx + 2*WPLANE_SIZE]
    lea     r15, [rcx + 3*WPLANE_SIZE]

    mov     rsi, rdx                ; activation planes
    mov     rdi, r8                 ; output int32[640]

    xor     ebx, ebx                ; row = 0

.row_loop:
    xor     r8, r8                  ; signed 64-bit accumulator
    xor     r11d, r11d              ; chunk byte offset = 0..248

.chunk_loop:
    ; Weight bit 0, coefficient +1
    mov     r10, [r12 + r11]
    DOT_X_SIGNED8
    add     r8, rcx

    ; Weight bit 1, coefficient +2
    mov     r10, [r13 + r11]
    DOT_X_SIGNED8
    shl     rcx, 1
    add     r8, rcx

    ; Weight bit 2, coefficient +4
    mov     r10, [r14 + r11]
    DOT_X_SIGNED8
    shl     rcx, 2
    add     r8, rcx

    ; Weight sign bit 3, coefficient -8
    mov     r10, [r15 + r11]
    DOT_X_SIGNED8
    shl     rcx, 3
    sub     r8, rcx

    add     r11, 8
    cmp     r11, ROW_BYTES
    jb      .chunk_loop

    mov     [rdi + rbx*4], r8d

    add     r12, ROW_BYTES
    add     r13, ROW_BYTES
    add     r14, ROW_BYTES
    add     r15, ROW_BYTES

    inc     ebx
    cmp     ebx, ROWS
    jb      .row_loop

    xor     eax, eax

    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    pop     rbx
    ret
