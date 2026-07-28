import struct, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from scm_opcodes import OPCODES, TAG_INT8, TAG_INT16, TAG_INT32, TAG_GVAR, TAG_STR8, GLOBAL_BASE
import scm_asm

def assemble(text):
    return scm_asm.assemble_text(text)

def test_header_chunk():
    scm = assemble("GLOBALS 2\nMAIN\nTERMINATE\n")
    # chunk 's' header: 02 00 01, nextOffset, 's'
    assert scm[0:3] == bytes([0x02, 0x00, 0x01])
    next_off = struct.unpack_from("<I", scm, 3)[0]
    assert scm[7] == ord('s')
    assert next_off == GLOBAL_BASE + 2 * 4          # 2 int globals
    # MAIN code begins at next_off; SCRIPT_NAME opcode 03A4 emitted by MAIN directive
    assert struct.unpack_from("<H", scm, next_off)[0] == 0x03A4

def test_wait_encoding():
    scm = assemble("GLOBALS 0\nMAIN\nWAIT 500\nTERMINATE\n")
    off = struct.unpack_from("<I", scm, 3)[0]
    # skip SCRIPT_NAME: opcode(2) + tag(1) + 8 bytes
    off += 2 + 1 + 8
    assert struct.unpack_from("<H", scm, off)[0] == 0x0001      # WAIT
    off += 2
    assert scm[off] == TAG_INT16                                 # 500 -> int16 tag 5
    assert struct.unpack_from("<h", scm, off + 1)[0] == 500

def test_gvar_and_label():
    scm = assemble("GLOBALS 1\nMAIN\n:top\nADD_INT_VAR $0 1\nGOTO @top\nTERMINATE\n")
    base = struct.unpack_from("<I", scm, 3)[0]
    off = base + 2 + 1 + 8      # past SCRIPT_NAME
    # ADD_INT_VAR: op 0008, gvar arg (tag 2, offset 8), int8 arg (tag 4, 1)
    assert struct.unpack_from("<H", scm, off)[0] == 0x0008
    off += 2
    assert scm[off] == TAG_GVAR and struct.unpack_from("<H", scm, off + 1)[0] == GLOBAL_BASE
    off += 3
    assert scm[off] == TAG_INT8 and scm[off + 1] == 1
    off += 2
    # GOTO @top: op 0002 + tag1 int32 == absolute offset of :top (== base+SCRIPT_NAME size)
    assert struct.unpack_from("<H", scm, off)[0] == 0x0002
    assert scm[off + 2] == TAG_INT32
    top_off = struct.unpack_from("<i", scm, off + 3)[0]
    assert top_off == base + 11        # :top is right after SCRIPT_NAME (2+1+8=11 bytes)

def test_gvar_bounds():
    try:
        assemble("GLOBALS 1\nMAIN\nSET_VAR_INT $5 1\nTERMINATE\n")
        assert False, "expected ValueError for out-of-range global"
    except ValueError:
        pass

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
    print("ALL PASS")
