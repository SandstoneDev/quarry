#!/usr/bin/env python3
"""scm_roadmap - turn the on-device unknown-opcode counters into a system-level roadmap.

Reads a session.log, picks up every "scm top-unk XXXX:N ..." line, resolves the opcode
names from sa.json, groups them by engine subsystem and prints what each missing system
is costing. The head of that table is the next handoff.

The device zeroes a counter slot once it has printed it, so the numbers on successive
top-unk lines are increments, not running totals - they are summed here.

Usage: python tools/scm_roadmap.py <session.log> [sa.json]
"""
import json, os, re, sys
from collections import Counter, defaultdict

SYSTEMS = [
    ("PED/TASK",  r'^(TASK_|PERFORM_SEQUENCE|OPEN_SEQUENCE|CLOSE_SEQUENCE|CLEAR_SEQUENCE|SET_SEQUENCE|.*DECISION_MAKER|ATTRACTOR)'),
    ("CHAR",      r'^(SET_CHAR|GET_CHAR|IS_CHAR|CREATE_CHAR|CLEAR_CHAR|REMOVE_CHAR|HAS_CHAR|CHAR_)'),
    ("CAR/VEH",   r'^(SET_CAR|GET_CAR|IS_CAR|CREATE_CAR|CLEAR_CAR|MARK_CAR|HAS_CAR|CAR_|.*_VEHICLE|VEHICLE_)'),
    # Model/entity streaming lifecycle: the script telling the engine what to page
    # in and what it is done with. Our streaming is tile-based and has no per-model
    # request at all, so this whole family is unimplemented - and at full scale it is
    # the single largest group. It had no category here at first and hid inside OTHER.
    ("MODEL/LIFE", r'(REQUEST_MODEL|REQUEST_ANIMATION|REQUEST_COLLISION|_NO_LONGER_NEEDED|HAS_MODEL_LOADED|LOAD_ALL_MODELS)'),
    ("STREAMSCR", r'(STREAMED_SCRIPT|STREAM_SCRIPT|SCRIPT_BRAIN)'),
    ("MISSION",   r'^(LOAD_AND_LAUNCH|.*MISSION.*)'),
    ("OBJECT",    r'^(SET_OBJECT|GET_OBJECT|IS_OBJECT|CREATE_OBJECT|DONT_REMOVE_OBJECT|ROTATE_OBJECT|ADD_TO_OBJECT|SLIDE_OBJECT)'),
    ("TEXT/HUD",  r'^(PRINT|SET_TEXT|DISPLAY_|.*_TEXT_LABEL|LOAD_SPRITE|DRAW_|.*_HELP.*|.*MESSAGE.*)'),
    ("STATS",     r'(_STAT|STAT_)'),
    ("GARAGE",    r'GARAGE'),
    ("PICKUP",    r'PICKUP'),
    ("BLIP/MAP",  r'(BLIP|MARKER|SPHERE)'),
    ("ZONE/POP",  r'(ZONE|POPULATION|GANG)'),
    ("AUDIO",     r'(AUDIO|SOUND|MP3|RADIO)'),
    ("ANIM",      r'(ANIM)'),
    ("CAM",       r'(CAMERA|CAM_)'),
    ("WANTED",    r'(WANTED|POLICE)'),
    ("IPL/WORLD", r'(_IPL|INTERIOR|ENTRY_EXIT|CLEAR_AREA|EXPLOD|FIRE|WEATHER|TIME)'),
    ("VAR/MATH",  r'^(SET_|ADD_|SUB_|MULT|DIV|IS_INT|IS_FLOAT|IS_NUMBER|ABS_|CSET|GENERATE_RANDOM|SWITCH_)'),
]


def load_names(path):
    db = json.load(open(path, encoding="utf-8"))
    return {int(c["id"], 16): c["name"]
            for ext in db["extensions"] for c in ext.get("commands", [])}


def main():
    log = sys.argv[1]
    saj = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sa.json")
    names = load_names(saj)

    counts = Counter()
    lines = 0
    for line in open(log, encoding="utf-8", errors="replace"):
        if "top-unk" not in line:
            continue
        lines += 1
        for op, n in re.findall(r'([0-9A-Fa-f]{4}):(\d+)', line):
            counts[int(op, 16)] += int(n)
    if not counts:
        print("no 'scm top-unk' lines in %s - was the build instrumented and scm_main=1?" % log)
        return

    groups, other = defaultdict(lambda: [0, 0]), []
    for op, n in counts.items():
        nm = names.get(op, "?0x%04X" % op)
        for label, pat in SYSTEMS:
            if re.search(pat, nm):
                groups[label][0] += n
                groups[label][1] += 1
                break
        else:
            groups["OTHER"][0] += n
            groups["OTHER"][1] += 1
            other.append((n, nm))

    print("scm roadmap from %s (%d top-unk lines, %d distinct opcodes, %d calls)\n"
          % (log, lines, len(counts), sum(counts.values())))
    print("  %-11s %8s  %s" % ("system", "calls", "opcodes"))
    for label, (calls, ops) in sorted(groups.items(), key=lambda x: -x[1][0]):
        print("  %-11s %8d  %d" % (label, calls, ops))

    print("\ntop 20 individual opcodes:")
    for op, n in counts.most_common(20):
        print("  0x%04X %7d  %s" % (op, n, names.get(op, "?")))
    if other:
        other.sort(reverse=True)
        print("\nOTHER, top 10: " + ", ".join(nm for _, nm in other[:10]))


if __name__ == "__main__":
    main()
