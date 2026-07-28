"""Find the code that touches a given address range in a MIPS ELF.

static analysis's base analysis leaves this binary with no code-to-data references at all,
because MIPS builds an address from a lui/addiu pair or an offset off $gp and the
analyzer that resolves those did not run. This does the same job directly: walk
every instruction, track what each register holds when it is a known constant, and
report the instructions whose effective address lands in the range asked for.

Usage: mipsref.py <elf> <lo-hex> <hi-hex> [gp-hex]
"""
import struct
import sys

LUI, ADDIU, ORI = 0x0F, 0x09, 0x0D
LOADS = {0x20: "lb", 0x21: "lh", 0x23: "lw", 0x24: "lbu", 0x25: "lhu", 0x37: "ld"}
STORES = {0x28: "sb", 0x29: "sh", 0x2B: "sw", 0x3F: "sd"}


def segments(blob):
    """ELF32 program headers -> [(vaddr, file_off, size)] for loadable code/data."""
    phoff = struct.unpack_from("<I", blob, 0x1C)[0]
    phentsize = struct.unpack_from("<H", blob, 0x2A)[0]
    phnum = struct.unpack_from("<H", blob, 0x2C)[0]
    out = []
    for i in range(phnum):
        o = phoff + i * phentsize
        p_type, p_off, p_vaddr, _pa, p_filesz = struct.unpack_from("<5I", blob, o)
        if p_type == 1 and p_filesz:
            out.append((p_vaddr, p_off, p_filesz))
    return out


def scan(blob, lo, hi, gp):
    hits = []
    for vaddr, off, size in segments(blob):
        reg = [None] * 32                      # last known constant per register
        for i in range(0, size - 3, 4):
            w = struct.unpack_from("<I", blob, off + i)[0]
            pc = vaddr + i
            op = w >> 26
            rs = (w >> 21) & 31
            rt = (w >> 16) & 31
            imm = w & 0xFFFF
            simm = imm - 0x10000 if imm & 0x8000 else imm

            if op == LUI:
                reg[rt] = imm << 16
                continue
            if op in (ADDIU, ORI):
                base = reg[rs]
                if rs == 28 and gp is not None:
                    base = gp                  # $gp is fixed for the whole program
                if base is not None:
                    val = (base + simm) if op == ADDIU else (base | imm)
                    reg[rt] = val & 0xFFFFFFFF
                    if lo <= reg[rt] <= hi:
                        hits.append((pc, "addr", reg[rt]))
                else:
                    reg[rt] = None
                continue
            if op in LOADS or op in STORES:
                base = reg[rs]
                if rs == 28 and gp is not None:
                    base = gp
                if base is not None:
                    ea = (base + simm) & 0xFFFFFFFF
                    if lo <= ea <= hi:
                        hits.append((pc, LOADS.get(op) or STORES[op], ea))
                if op in LOADS:
                    reg[rt] = None             # loaded value is not a known constant
                continue
            # anything else that writes a register invalidates what we knew
            if op == 0:                        # SPECIAL: rd is the destination
                rd = (w >> 11) & 31
                if rd:
                    reg[rd] = None
            elif op in (0x02, 0x03):           # j / jal - registers survive
                pass
            elif rt:
                reg[rt] = None
    return hits


def main():
    blob = open(sys.argv[1], "rb").read()
    lo, hi = int(sys.argv[2], 16), int(sys.argv[3], 16)
    gp = int(sys.argv[4], 16) if len(sys.argv) > 4 else None
    hits = scan(blob, lo, hi, gp)
    print("instructions touching %08x..%08x: %d" % (lo, hi, len(hits)))
    for pc, kind, ea in hits[:60]:
        print("   %08x  %-5s -> %08x" % (pc, kind, ea))


if __name__ == "__main__":
    main()
