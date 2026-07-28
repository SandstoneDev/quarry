#!/usr/bin/env python3
"""
scm_disasm.py - the source game main.scm opcode-usage disassembler / histogram.

Purpose (for the PSP demake SCM VM): find out exactly which opcodes the REAL
vanilla main.scm uses and how often, so we can prioritise implementing handlers
and build the arg-length table needed to skip unimplemented opcodes without
desyncing the instruction pointer.

How it works
------------
SA SCM bytecode is *self-describing per argument*: every argument is preceded by
a 1-byte parameter-type tag (eScriptParameterType, values 0..0x13). What the
stream does NOT encode is how many arguments each opcode takes - for that we use
an opcode table.

Arg-count table source: Sanny Builder Library `sa.json`
(https://github.com/sannybuilder/library) -> per-opcode `num_params`
(= inputs + outputs). This is a complete, authoritative, machine-readable table
for all ~3700 known SA/CLEO opcodes. We only need the vanilla subset.

Fallback: for any opcode missing from the table we greedily consume args while
the next byte is a valid type tag, and FLAG the instruction as heuristic.

Variadic opcodes (START_NEW_SCRIPT 0x004F etc.) end their arg list with an
END_OF_ARGUMENTS (0x00) tag; handled explicitly.

Usage:  python scm_disasm.py [path-to-main.scm] [path-to-sa.json]
"""
import sys, os, struct, json
from collections import Counter, defaultdict

# ---- parameter-type tag -> byte length consumed AFTER the 1-byte tag ----------
# Values are eScriptParameterType from the analysed SA parser (contiguous 0..0x13)
# 0x00 END_OF_ARGS | 0x01 int32 | 0x02 gvar | 0x03 lvar | 0x04 int8 | 0x05 int16
# 0x06 float | 0x07 gArray | 0x08 lArray | 0x09 str8 | 0x0A gStrVar | 0x0B lStrVar
# 0x0C gStrArray | 0x0D lStrArray | 0x0E pascalStr(len-prefixed) | 0x0F str16
# 0x10 gLongStrVar | 0x11 lLongStrVar | 0x12 gLongStrArray | 0x13 lLongStrArray
TAGLEN = {
    0x00: 0,
    0x01: 4, 0x02: 2, 0x03: 2, 0x04: 1, 0x05: 2, 0x06: 4,
    0x07: 6, 0x08: 6,
    0x09: 8, 0x0A: 2, 0x0B: 2, 0x0C: 6, 0x0D: 6,
    0x0E: None,   # pascal string: 1 length byte then N chars (special)
    0x0F: 16, 0x10: 2, 0x11: 2, 0x12: 6, 0x13: 6,
}
MAX_TAG = 0x13

MAIN_OFFSET = 55976   # confirmed by header GOTO-chain trace

def load_table(sajson_path):
    d = json.load(open(sajson_path, encoding='utf-8'))
    table = {}   # id_int -> dict(name, num_params, types[list], vararg)
    for ext in d['extensions']:
        for c in ext['commands']:
            cid = int(c['id'], 16)
            types = [p['type'] for p in c.get('input', [])] + \
                    [p['type'] for p in c.get('output', [])]
            vararg = 'arguments' in types
            table[cid] = {
                'name': c['name'],
                'num_params': c.get('num_params', 0),
                'types': types,
                'vararg': vararg,
                'ext': ext['name'],
            }
    return table

def consume_one_arg(mm, ip):
    """Consume one tagged argument starting at ip. Return new ip, or None on bad tag."""
    if ip >= len(mm):
        return None
    tag = mm[ip]
    if tag > MAX_TAG:
        return None
    ip += 1
    if tag == 0x0E:            # pascal string: length byte + chars
        if ip >= len(mm): return None
        n = mm[ip]; ip += 1 + n
        return ip
    return ip + TAGLEN[tag]

def try_decode_one(mm, table, ip, end):
    """Tentatively decode a single instruction. Return (next_ip, op) or (None, op).
    Used by the resync scanner - strict: opcode must be in table and in valid range."""
    if ip + 2 > end:
        return None, None
    op = struct.unpack_from('<H', mm, ip)[0] & 0x7FFF
    if op > 0x0EFF or op not in table:
        return None, op
    info = table[op]
    ip += 2
    types = info['types']
    if info['vararg']:
        fixed = 0
        for t in types:
            if t == 'arguments':
                break
            fixed += 1
        for _ in range(fixed):
            nip = consume_one_arg(mm, ip)
            if nip is None:
                return None, op
            ip = nip
        while ip < end:
            tag = mm[ip]
            nip = consume_one_arg(mm, ip)
            if nip is None:
                return None, op
            ip = nip
            if tag == 0x00:
                break
        return ip, op
    for pi in range(info['num_params']):
        pt = types[pi] if pi < len(types) else None
        if pt == 'string128':
            z = mm.find(b'\x00', ip)
            if z < 0 or z >= end:
                return None, op
            ip = z + 1
            continue
        nip = consume_one_arg(mm, ip)
        if nip is None:
            return None, op
        ip = nip
    return ip, op

def resync(mm, table, ip, end, window=6):
    """Scan forward from ip byte-by-byte until `window` consecutive instructions
    decode cleanly (all in-range, in-table). Return the locked ip, or end."""
    pos = ip
    while pos < end:
        cur = pos
        ok = True
        for _ in range(window):
            nip, _op = try_decode_one(mm, table, cur, end)
            if nip is None or nip <= cur:
                ok = False
                break
            cur = nip
        if ok:
            return pos
        pos += 1
    return end

def disasm(mm, table, start, end, hist, unknown_ids, heuristic_sites, mission_end=None):
    ip = start
    n_instr = 0
    n_heur = 0
    n_resync = 0
    resync_gap = 0
    while ip < end:
        op_ip = ip
        if ip + 2 > end:
            break
        raw = struct.unpack_from('<H', mm, ip)[0]
        op = raw & 0x7FFF          # strip the NOT flag (high bit)
        info = table.get(op)
        if info is None or op > 0x0EFF:
            # desync (garbage / untracked debug data). Resync onto the stream.
            unknown_ids[op] += 1
            newip = resync(mm, table, ip + 1, end)
            resync_gap += (newip - ip)
            n_resync += 1
            ip = newip
            continue
        ip += 2

        hist[op] += 1
        n_instr += 1
        nparams = info['num_params']

        if info['vararg']:
            # read the fixed leading (non-'arguments') params, then consume until END_OF_ARGS
            types = info['types']
            fixed = 0
            for t in types:
                if t == 'arguments':
                    break
                fixed += 1
            for _ in range(fixed):
                nip = consume_one_arg(mm, ip)
                if nip is None: break
                ip = nip
            # variadic tail terminated by 0x00 tag
            while ip < end:
                tag = mm[ip]
                nip = consume_one_arg(mm, ip)
                if nip is None: break
                ip = nip
                if tag == 0x00:
                    break
            continue

        # fixed-arity opcode: consume exactly num_params args.
        # Most args are type-tagged; the sole exception is the untagged,
        # null-terminated 'string128' arg (opcode 0x05B6 SAVE_STRING_TO_DEBUG_FILE).
        types = info['types']
        ok = True
        for pi in range(nparams):
            ptype = types[pi] if pi < len(types) else None
            if ptype == 'string128':
                z = mm.find(b'\x00', ip)
                if z < 0 or z >= end:
                    ok = False; break
                ip = z + 1
                continue
            nip = consume_one_arg(mm, ip)
            if nip is None:
                ok = False
                break
            ip = nip
        if not ok:
            # arg consumption hit invalid tag -> desync; resync onto the stream
            heuristic_sites.append((op_ip, op))
            n_heur += 1
            newip = resync(mm, table, op_ip + 2, end)
            resync_gap += (newip - (op_ip + 2))
            n_resync += 1
            ip = newip
    return n_instr, n_heur, ip, n_resync, resync_gap

def main():
    scm = sys.argv[1] if len(sys.argv) > 1 else \
        ""
    here = os.path.dirname(os.path.abspath(__file__))
    sajson = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "sa.json")
    if not os.path.exists(sajson):
        # fallback: scratchpad copy
        alt = ""
        if os.path.exists(alt): sajson = alt

    mm = open(scm, "rb").read()
    table = load_table(sajson)
    print("main.scm size: %d bytes" % len(mm))
    print("opcode table: %d entries (sa.json)" % len(table))
    print("MAIN thread offset: %d\n" % MAIN_OFFSET)

    hist = Counter()
    unknown_ids = Counter()
    heuristic_sites = []
    n_instr, n_heur, endip, n_resync, resync_gap = disasm(
        mm, table, MAIN_OFFSET, len(mm), hist, unknown_ids, heuristic_sites)

    distinct = len(hist)
    print("Instructions decoded (linear sweep MAIN..EOF): %d" % n_instr)
    print("Distinct opcodes used: %d" % distinct)
    print("Heuristic/uncertain sites: %d" % n_heur)
    print("Resync events: %d, total bytes skipped as untracked-data: %d (%.3f%% of body)" %
          (n_resync, resync_gap, 100.0*resync_gap/(len(mm)-MAIN_OFFSET)))
    print("Final IP reached: %d / %d\n" % (endip, len(mm)))

    print("=== TOP 60 OPCODES BY FREQUENCY ===")
    print("%-6s %-38s %8s" % ("op", "name", "count"))
    for op, cnt in hist.most_common(60):
        info = table.get(op)
        name = info['name'] if info else "??? (unknown)"
        print("0x%04X %-38s %8d" % (op, name, cnt))

    if unknown_ids:
        print("\n=== UNKNOWN OPCODE IDS (not in table) ===")
        for op, cnt in unknown_ids.most_common(30):
            print("0x%04X  x%d" % (op, cnt))

    # dump full histogram to file for downstream roadmap use
    out = os.path.join(here, "scm_histogram.txt")
    with open(out, "w", encoding='utf-8') as f:
        for op, cnt in hist.most_common():
            info = table.get(op)
            name = info['name'] if info else "UNKNOWN"
            np = info['num_params'] if info else -1
            f.write("0x%04X\t%d\t%d\t%s\n" % (op, cnt, np, name))
    print("\nFull histogram (%d rows) -> %s" % (distinct, out))

if __name__ == "__main__":
    main()
