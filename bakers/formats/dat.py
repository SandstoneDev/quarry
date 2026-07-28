"""the source game DAT + CFG config-family decoder (ASCII tables/sections + binary nodes).

The DATA/ config family is the source game's gameplay-tuning and world-config layer: small
data tables that drive vehicle dynamics (handling.cfg), water surfaces (water.dat),
dynamic-object physics (object.dat), day/night colour grading (timecyc.dat) and
population (popcycle.dat) - plus the one binary member, NODESnn.DAT, the path graph.

Almost every text member is parsed through ONE shared tokenizer,
, which replaces every control char (<0x20) AND every
comma with a space - so commas == whitespace. Each file then runs a dedicated
sscanf/strtok loop with an asserted field count. NODESnn.DAT is little-endian POD
read straight off RwStreamRead, validated by an exact file-size formula.

Public API:
 load_line(raw) -> tokenizer (commas == whitespace)
 parse_dat_table(text, ...) -> generic list[list[token]]
 parse_handling / parse_water / parse_object / parse_timecyc / parse_popcycle
 parse_nodes(data) -> binary path graph
 parse_dat(name, data) -> dispatch by filename
 to_json(model) -> JSON-serializable dict (web table view)

 (confirmed byte layout, v1.0 US)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# ======================================================================
# Shared ASCII tokenizer 
# ======================================================================

# Every byte < 0x20 and every ',' becomes a space, then split collapses runs.
_WS_TRANSLATE = {i: 0x20 for i in range(0x20)}
_WS_TRANSLATE[ord(",")] = 0x20


def load_line(raw: str) -> List[str]:
    """Tokenize one logical line exactly like CFileLoader::LoadLine.

 Replaces every control char (<0x20) AND every comma with a space, then splits
 on whitespace. Commas are therefore field separators just like spaces/tabs.
 """
    return raw.translate(_WS_TRANSLATE).split()


def _strip_inline_comment(raw: str, leaders: Sequence[str]) -> str:
    """Cut a trailing inline comment (popcycle uses ``... // label``)."""
    cut = len(raw)
    for lead in leaders:
        i = raw.find(lead)
        if i != -1 and i < cut:
            cut = i
    return raw[:cut]


def _is_comment(stripped: str, leaders: Sequence[str]) -> bool:
    s = stripped.lstrip()
    return any(s.startswith(lead) for lead in leaders)


def parse_dat_table(
    text: str,
    comment_leaders: Sequence[str] = (";", "#"),
    terminators: Sequence[str] = (),
    inline_comment_leaders: Sequence[str] = (),
) -> List[List[str]]:
    """Generic column-table reader -> a list of token-rows.

 Skips blank lines and any line whose first non-space char is a comment leader.
 Stops (exclusive) at the first line whose first token equals a terminator OR
 whose first char starts a terminator (handling.cfg ';the end', object '*').
 ``inline_comment_leaders`` trims a trailing comment from a data line first.
 """
    rows: List[List[str]] = []
    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        # terminator: tested on the raw, un-tokenized line (engine compares line[0]
        # / exact strings before tokenizing).
        ls = line.lstrip()
        if terminators:
            hit = False
            for term in terminators:
                if ls == term or ls.startswith(term):
                    hit = True
                    break
            if hit:
                break
        if not line.strip():
            continue
        if _is_comment(line, comment_leaders):
            continue
        if inline_comment_leaders:
            line = _strip_inline_comment(line, inline_comment_leaders)
        toks = load_line(line)
        if toks:
            rows.append(toks)
    return rows


# small numeric coercers (atof/atol are forgiving; mirror that, never crash a row)
def _f(tok: str, default: float = 0.0) -> float:
    try:
        return float(tok)
    except (ValueError, TypeError):
        return default


def _i(tok: str, default: int = 0) -> int:
    try:
        return int(tok)
    except (ValueError, TypeError):
        try:
            return int(float(tok))
        except (ValueError, TypeError):
            return default


def _hx(tok: str, default: int = 0) -> int:
    try:
        return int(tok, 16)
    except (ValueError, TypeError):
        return default


# ======================================================================
# (1) handling.cfg - 
# ======================================================================
# Lead char of the line selects the bank (ground-truth for the source game v1.0 US):
# alnum (default) = Car tHandlingData (name + 35 cols)
# '!' = bike-bank | '$' = second-bank | '%' = boat/flying | '^' = anim/special-flags
# Stop EXACTLY at ";the end". ';' otherwise = comment.

# Car column index -> (attr, kind). col0 = name. kind: f float, i int, c char, x hex.
_CAR_COLS = [
    ("fMass", "f"), ("fTurnMass", "f"), ("fDragMult", "f"),
    ("com_x", "f"), ("com_y", "f"), ("com_z", "f"),
    ("nPercentSubmerged", "i"),
    ("fTractionMultiplier", "f"), ("fTractionLoss", "f"), ("fTractionBias", "f"),
    ("nGears", "i"), ("fMaxVelocity", "f"), ("fEngineAccel", "f"),
    ("fEngineInertia", "f"), ("nDriveType", "c"), ("nEngineType", "c"),
    ("fBrakeDecel", "f"), ("fBrakeBias", "f"), ("bABS", "i"), ("fSteeringLock", "f"),
    ("fSuspForceLevel", "f"), ("fSuspDampingLevel", "f"), ("fSuspHighSpdComDamp", "f"),
    ("fSuspUpperLimit", "f"), ("fSuspLowerLimit", "f"), ("fSuspBias", "f"),
    ("fSuspAntiDiveMult", "f"), ("fSeatOffsetDist", "f"), ("fCollDamageMult", "f"),
    ("nMonetaryValue", "i"), ("ModelFlags", "x"), ("HandlingFlags", "x"),
    ("FrontLights", "i"), ("RearLights", "i"), ("AnimGroup", "i"),
]


@dataclass
class HandlingCar:
    """One Car tHandlingData row (sizeof 0xE0 in-engine). 36 tokens: name + 35 cols."""
    name: str
    tokens: List[str] = field(default_factory=list)  # raw tokens, column order preserved
    # decoded named fields (default-filled so a short row never crashes)
    fMass: float = 0.0
    fTurnMass: float = 0.0
    fDragMult: float = 0.0
    com_x: float = 0.0
    com_y: float = 0.0
    com_z: float = 0.0
    nPercentSubmerged: int = 0
    fTractionMultiplier: float = 0.0
    fTractionLoss: float = 0.0
    fTractionBias: float = 0.0
    nGears: int = 0
    fMaxVelocity: float = 0.0
    fEngineAccel: float = 0.0
    fEngineInertia: float = 0.0
    nDriveType: str = ""
    nEngineType: str = ""
    fBrakeDecel: float = 0.0
    fBrakeBias: float = 0.0
    bABS: int = 0
    fSteeringLock: float = 0.0
    fSuspForceLevel: float = 0.0
    fSuspDampingLevel: float = 0.0
    fSuspHighSpdComDamp: float = 0.0
    fSuspUpperLimit: float = 0.0
    fSuspLowerLimit: float = 0.0
    fSuspBias: float = 0.0
    fSuspAntiDiveMult: float = 0.0
    fSeatOffsetDist: float = 0.0
    fCollDamageMult: float = 0.0
    nMonetaryValue: int = 0
    ModelFlags: int = 0
    HandlingFlags: int = 0
    FrontLights: int = 0
    RearLights: int = 0
    AnimGroup: int = 0


@dataclass
class HandlingRow:
    """A non-Car bank record (bike '!', second '$', boat '%', anim '^').

 These are kept as the lead char + raw tokens (column order preserved for
 re-emit); the per-column physics meaning differs per bank and is not needed
 for the table view.
 """
    lead: str
    name: str           # token after the lead char (boat/bike name, or index for '^')
    tokens: List[str] = field(default_factory=list)


@dataclass
class Handling:
    cars: List[HandlingCar] = field(default_factory=list)
    bikes: List[HandlingRow] = field(default_factory=list)    # '!'
    flying: List[HandlingRow] = field(default_factory=list)   # '$' second-bank
    boats: List[HandlingRow] = field(default_factory=list)    # '%'
    anim_flags: List[HandlingRow] = field(default_factory=list)  # '^'


def _decode_car(toks: List[str]) -> HandlingCar:
    car = HandlingCar(name=toks[0], tokens=list(toks))
    for idx, (attr, kind) in enumerate(_CAR_COLS, start=1):
        if idx >= len(toks):
            break
        tok = toks[idx]
        if kind == "f":
            setattr(car, attr, _f(tok))
        elif kind == "i":
            setattr(car, attr, _i(tok))
        elif kind == "x":
            setattr(car, attr, _hx(tok))
        else:  # 'c' single char (drive/engine type)
            setattr(car, attr, tok[:1])
    return car


def parse_handling(text: str) -> Handling:
    out = Handling()
    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        ls = line.lstrip()
        if ls == ";the end":          # EXACT terminator, no trailing space
            break
        if not ls or ls.startswith(";"):
            continue
        lead = ls[0]
        try:
            if lead == "!":
                toks = load_line(ls[1:])
                if toks:
                    out.bikes.append(HandlingRow("!", toks[0], toks))
            elif lead == "$":
                toks = load_line(ls[1:])
                if toks:
                    out.flying.append(HandlingRow("$", toks[0], toks))
            elif lead == "%":
                toks = load_line(ls[1:])
                if toks:
                    out.boats.append(HandlingRow("%", toks[0], toks))
            elif lead == "^":
                toks = load_line(ls[1:])
                if toks:
                    out.anim_flags.append(HandlingRow("^", toks[0], toks))
            else:  # alnum default case -> Car
                toks = load_line(ls)
                if toks:
                    out.cars.append(_decode_car(toks))
        except Exception:
            # one malformed line must never abort the whole config
            continue
    return out


# ======================================================================
# (3) water.dat - 
# ======================================================================
# One polygon per line, whitespace-separated. Skip first char ';' '*' 'p' / blank.
# 29 tokens = 4 corners x 7f + flag (quad) | 28 = 4 corners x 7f (flag default 1)
# 22 tokens = 3 corners x 7f + flag (triangle). Corner = X Y Z f3 f4 f5 f6.

WaterCorner = Tuple[float, float, float, float, float, float, float]


@dataclass
class WaterQuad:
    corners: List[WaterCorner]   # 4 corners
    flag: int = 1


@dataclass
class WaterTriangle:
    corners: List[WaterCorner]   # 3 corners
    flag: int = 1


@dataclass
class Water:
    quads: List[WaterQuad] = field(default_factory=list)
    triangles: List[WaterTriangle] = field(default_factory=list)


def _corners(nums: List[float], count: int) -> List[WaterCorner]:
    out: List[WaterCorner] = []
    for k in range(count):
        b = k * 7
        out.append(tuple(nums[b:b + 7]))  # type: ignore[arg-type]
    return out


def parse_water(text: str) -> Water:
    out = Water()
    for raw in text.splitlines():
        line = raw.rstrip("\r\n").lstrip()
        if not line:
            continue
        c0 = line[0]
        # ';' comment, '*' terminator/marker, 'p' = the "processed" header line
        if c0 in ";*p":
            continue
        toks = load_line(line)
        n = len(toks)
        try:
            if n >= 29:
                nums = [_f(t) for t in toks[:28]]
                out.quads.append(WaterQuad(_corners(nums, 4), _i(toks[28])))
            elif n == 28:
                nums = [_f(t) for t in toks[:28]]
                out.quads.append(WaterQuad(_corners(nums, 4), 1))
            elif n >= 22:
                nums = [_f(t) for t in toks[:21]]
                out.triangles.append(WaterTriangle(_corners(nums, 3), _i(toks[21])))
            # anything else: not a polygon row, ignore
        except Exception:
            continue
    return out


# ======================================================================
# (4) object.dat - 
# ======================================================================
# ';'/'#' comment, '*' terminator. Commas == whitespace. >=13 fields, up to 24.
# 1 name | 2 mass | 3 turnMass | 4 airRes | 5 elasticity | 6 %submerged |
# 7 uprootLimit | 8 colDamageMult | 9 colDamageEffect | 10 specialColResp |
# 11 cameraAvoid | 12 causesExplosion | 13 fxType | 14-16 fxOffset | 17 effectName |
# 18 smashMult | 19-21 breakVel | 22 breakVelRand | 23 gunBreakMode | 24 sparks.


@dataclass
class ObjectRecord:
    name: str
    tokens: List[str] = field(default_factory=list)
    mass: float = 0.0
    turn_mass: float = 0.0
    air_resistance: float = 0.0
    elasticity: float = 0.0
    percent_submerged: float = 0.0
    uproot_limit: float = 0.0
    col_damage_mult: float = 0.0
    col_damage_effect: int = 0
    special_col_response: int = 0
    camera_avoid: int = 0
    causes_explosion: int = 0
    fx_type: int = 0
    fx_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    effect_name: str = ""
    # optional breakable tail (cols 18-24)
    smash_multiplier: Optional[float] = None
    break_velocity: Optional[Tuple[float, float, float]] = None
    break_velocity_rand: Optional[float] = None
    gun_break_mode: Optional[int] = None
    sparks_on_impact: Optional[int] = None


@dataclass
class ObjectData:
    objects: List[ObjectRecord] = field(default_factory=list)


def _decode_object(toks: List[str]) -> ObjectRecord:
    g = lambda i: toks[i] if i < len(toks) else None  # noqa: E731
    rec = ObjectRecord(name=toks[0], tokens=list(toks))
    rec.mass = _f(g(1))
    rec.turn_mass = _f(g(2))
    rec.air_resistance = _f(g(3))
    rec.elasticity = _f(g(4))
    rec.percent_submerged = _f(g(5))
    rec.uproot_limit = _f(g(6))
    rec.col_damage_mult = _f(g(7))
    rec.col_damage_effect = _i(g(8))
    rec.special_col_response = _i(g(9))
    rec.camera_avoid = _i(g(10))
    rec.causes_explosion = _i(g(11))
    rec.fx_type = _i(g(12))
    rec.fx_offset = (_f(g(13)), _f(g(14)), _f(g(15)))
    if g(16) is not None:
        rec.effect_name = g(16)
    # optional breakable physics tail
    if len(toks) >= 18:
        rec.smash_multiplier = _f(g(17))
    if len(toks) >= 21:
        rec.break_velocity = (_f(g(18)), _f(g(19)), _f(g(20)))
    if len(toks) >= 22:
        rec.break_velocity_rand = _f(g(21))
    if len(toks) >= 23:
        rec.gun_break_mode = _i(g(22))
    if len(toks) >= 24:
        rec.sparks_on_impact = _i(g(23))
    return rec


def parse_object(text: str) -> ObjectData:
    out = ObjectData()
    rows = parse_dat_table(text, comment_leaders=(";", "#"), terminators=("*",))
    for toks in rows:
        if len(toks) < 13:
            continue            # engine requires >=13 conversions
        try:
            out.objects.append(_decode_object(toks))
        except Exception:
            continue
    return out


# ======================================================================
# (5) timecyc.dat - 
# ======================================================================
# '/' comment leader, fixed 23 weather x 8 hour = 184 rows. Each row holds plain
# ints/floats; the byte/u16 quantization is the in-engine storage, the FILE holds
# the human values. We surface the leading colour groups + keep raw tokens.
# col(1-based): 1-3 AmbientRGB, 4-6 AmbientObjRGB, 7-9 DirRGB(discarded by engine),
# 10-12 SkyTopRGB, 13-15 SkyBotRGB, 16-18 SunCoreRGB, 19-21 SunCoronaRGB...


def _rgb(toks: List[str], base0: int) -> Tuple[int, int, int]:
    return (_i(_get(toks, base0)), _i(_get(toks, base0 + 1)), _i(_get(toks, base0 + 2)))


def _get(toks: List[str], i: int):
    return toks[i] if 0 <= i < len(toks) else None


@dataclass
class TimecycRow:
    tokens: List[str] = field(default_factory=list)
    ambient_rgb: Tuple[int, int, int] = (0, 0, 0)
    ambient_obj_rgb: Tuple[int, int, int] = (0, 0, 0)
    dir_rgb: Tuple[int, int, int] = (0, 0, 0)
    sky_top_rgb: Tuple[int, int, int] = (0, 0, 0)
    sky_bot_rgb: Tuple[int, int, int] = (0, 0, 0)
    sun_core_rgb: Tuple[int, int, int] = (0, 0, 0)
    sun_corona_rgb: Tuple[int, int, int] = (0, 0, 0)


@dataclass
class Timecyc:
    rows: List[TimecycRow] = field(default_factory=list)


def _decode_timecyc(toks: List[str]) -> TimecycRow:
    return TimecycRow(
        tokens=list(toks),
        ambient_rgb=_rgb(toks, 0),
        ambient_obj_rgb=_rgb(toks, 3),
        dir_rgb=_rgb(toks, 6),
        sky_top_rgb=_rgb(toks, 9),
        sky_bot_rgb=_rgb(toks, 12),
        sun_core_rgb=_rgb(toks, 15),
        sun_corona_rgb=_rgb(toks, 18),
    )


def parse_timecyc(text: str) -> Timecyc:
    out = Timecyc()
    # leader '/' covers both '//' and a bare '/'.
    rows = parse_dat_table(text, comment_leaders=("/",))
    for toks in rows:
        # a colour row carries the 21 leading colour ints at minimum; skip stray short lines
        if len(toks) < 21:
            continue
        try:
            out.rows.append(_decode_timecyc(toks))
        except Exception:
            continue
    return out


# ======================================================================
# (6) popcycle.dat - 
# ======================================================================
# '/' comment leader (plus inline '//' label after the numbers). Fixed grid
# 20 zones x 2 wktime x 12 daytime = 480 rows x 24 %hhu. cols: 1 MaxPeds 2 MaxCars
# 3 %Dealers 4 %Gang 5 %Cops 6 %Other 7-24 = 18 group percentages.


@dataclass
class PopcycleRow:
    values: List[int] = field(default_factory=list)   # all 24 ints
    max_peds: int = 0
    max_cars: int = 0
    pct_dealers: int = 0
    pct_gang: int = 0
    pct_cops: int = 0
    pct_other: int = 0
    groups: List[int] = field(default_factory=list)   # 18 group percentages


@dataclass
class Popcycle:
    rows: List[PopcycleRow] = field(default_factory=list)


def parse_popcycle(text: str) -> Popcycle:
    out = Popcycle()
    rows = parse_dat_table(
        text, comment_leaders=("/",), inline_comment_leaders=("//",)
    )
    for toks in rows:
        if len(toks) < 24:
            continue
        try:
            vals = [_i(t) for t in toks[:24]]
            out.rows.append(PopcycleRow(
                values=vals,
                max_peds=vals[0], max_cars=vals[1],
                pct_dealers=vals[2], pct_gang=vals[3],
                pct_cops=vals[4], pct_other=vals[5],
                groups=vals[6:24],
            ))
        except Exception:
            continue
    return out


# ======================================================================
# (21) NODESnn.DAT - CPathFind (BINARY)
# ======================================================================
# All little-endian POD, plain RwStreamRead (NO XOR). Read order:
# HEADER 5xu32: numNodes, numVehicleNodes, numPedNodes, numCarPathLinks, numAddresses
# CPathNode[numNodes] @ 28B
# CCarPathLink[numCarPathLinks] @ 14B
# CNodeAddress links[numToAdd] @ 4B (numToAdd = numAddresses + 192 slack)
# CCarPathLinkAddress nav[numAddresses] @ 2B
# u8 linkLengths[numToAdd] @ 1B
# CPathIntersectionInfo[numToAdd]@ 1B
# size = 20 + nodes*28 + carlinks*14 + numToAdd*4 + addr*2 + numToAdd + numToAdd.

_DYNAMIC_SLACK = 192   # NUM_DYNAMIC_LINKS_PER_AREA(16) * 12 - present on disk
_POS_SCALE = 1.0 / 8.0
_WIDTH_SCALE = 1.0 / 16.0
_DIR_SCALE = 1.0 / 100.0

# CPathNode head: m_next/m_prev are junk pointers on disk -> two i32 we skip,
# then s16 x,y,z, s16 totalDist, s16 baseLink, u16 area, u16 node. After node_id come
# the scalar/flag bytes:,,,,
#, (read explicitly below - they are NOT part of this struct).
_NODE = struct.Struct("<iihhhhhHH")     # 20 bytes (0x00..0x16); tail bytes read by index
_NODE_SIZE = 28
_CARLINK = struct.Struct("<hhHHbbB")    # 13 bytes -> + lane byte + u16 light = 14
_CARLINK_SIZE = 14


@dataclass
class PathNode:
    index: int
    x: float
    y: float
    z: float
    total_dist_from_origin: int
    base_link_id: int
    area_id: int
    node_id: int
    path_width: float
    flood_fill: int
    num_links: int
    on_dead_end: bool
    is_switched_off: bool
    road_blocks: bool
    water_node: bool
    spawn_probability: int
    behaviour_type: int


@dataclass
class CarPathLink:
    index: int
    x: float
    y: float
    area_id: int
    node_id: int
    dir_x: float
    dir_y: float
    path_node_width: float
    num_opposite_lanes: int
    num_same_lanes: int
    traffic_light_dir: int
    traffic_light_state: int
    bridge_lights: bool


@dataclass
class NodeAddress:
    area_id: int
    node_id: int

    @property
    def is_null(self) -> bool:
        return self.area_id == 0xFFFF and self.node_id == 0xFFFF


@dataclass
class CarPathLinkAddress:
    car_path_link_id: int      # 10 bits
    area_id: int               # 6 bits
    raw: int

    @property
    def is_invalid(self) -> bool:
        return self.raw == 0xFFFF


@dataclass
class IntersectionInfo:
    road_cross: bool
    ped_traffic_light: bool


@dataclass
class Nodes:
    num_nodes: int
    num_vehicle_nodes: int
    num_ped_nodes: int
    num_car_path_links: int
    num_addresses: int
    byte_size: int
    nodes: List[PathNode] = field(default_factory=list)
    car_path_links: List[CarPathLink] = field(default_factory=list)
    links: List[NodeAddress] = field(default_factory=list)
    navi_links: List[CarPathLinkAddress] = field(default_factory=list)
    link_lengths: List[int] = field(default_factory=list)
    intersections: List[IntersectionInfo] = field(default_factory=list)


def _decode_node(buf: bytes, off: int, index: int) -> PathNode:
    (_nxt, _prv, px, py, pz, total, base, area, node) = _NODE.unpack_from(buf, off)
    fb0 = buf[off + 0x18]   # flagbyte0: numLinks/deadEnd/switchedOff/roadBlocks/water
    fb2 = buf[off + 0x1A]   # flagbyte2: spawnProbability + behaviourType
    return PathNode(
        index=index,
        x=px * _POS_SCALE, y=py * _POS_SCALE, z=pz * _POS_SCALE,
        total_dist_from_origin=total,
        base_link_id=base,
        area_id=area, node_id=node,
        path_width=buf[off + 0x16] * _WIDTH_SCALE,
        flood_fill=buf[off + 0x17],
        num_links=fb0 & 0x0F,
        on_dead_end=bool(fb0 & 0x10),
        is_switched_off=bool(fb0 & 0x20),
        road_blocks=bool(fb0 & 0x40),
        water_node=bool(fb0 & 0x80),
        spawn_probability=fb2 & 0x0F,
        behaviour_type=(fb2 >> 4) & 0x0F,
    )


def _decode_carlink(buf: bytes, off: int, index: int) -> CarPathLink:
    (px, py, area, node, dx, dy, width) = _CARLINK.unpack_from(buf, off)
    lane = buf[off + 0x0B]
    light = struct.unpack_from("<H", buf, off + 0x0C)[0]
    return CarPathLink(
        index=index,
        x=px * _POS_SCALE, y=py * _POS_SCALE,
        area_id=area, node_id=node,
        dir_x=dx * _DIR_SCALE, dir_y=dy * _DIR_SCALE,
        path_node_width=width * _WIDTH_SCALE,
        num_opposite_lanes=lane & 0x07,
        num_same_lanes=(lane >> 3) & 0x07,
        traffic_light_dir=(lane >> 6) & 0x01,
        traffic_light_state=light & 0x03,
        bridge_lights=bool(light & 0x04),
    )


def parse_nodes(data: bytes) -> Nodes:
    if len(data) < 20:
        raise ValueError("NODES file too short for 20-byte header")
    num_nodes, num_veh, num_ped, num_links, num_addr = struct.unpack_from("<5I", data, 0)
    num_to_add = num_addr + _DYNAMIC_SLACK

    model = Nodes(
        num_nodes=num_nodes, num_vehicle_nodes=num_veh, num_ped_nodes=num_ped,
        num_car_path_links=num_links, num_addresses=num_addr, byte_size=len(data),
    )

    o = 20
    for i in range(num_nodes):
        if o + _NODE_SIZE > len(data):
            break
        try:
            model.nodes.append(_decode_node(data, o, i))
        except Exception:
            pass
        o += _NODE_SIZE

    for i in range(num_links):
        if o + _CARLINK_SIZE > len(data):
            break
        try:
            model.car_path_links.append(_decode_carlink(data, o, i))
        except Exception:
            pass
        o += _CARLINK_SIZE

    for _ in range(num_to_add):
        if o + 4 > len(data):
            break
        a, n = struct.unpack_from("<HH", data, o)
        model.links.append(NodeAddress(a, n))
        o += 4

    for _ in range(num_addr):
        if o + 2 > len(data):
            break
        raw = struct.unpack_from("<H", data, o)[0]
        model.navi_links.append(
            CarPathLinkAddress(car_path_link_id=raw & 0x3FF, area_id=(raw >> 10) & 0x3F, raw=raw)
        )
        o += 2

    for _ in range(num_to_add):
        if o + 1 > len(data):
            break
        model.link_lengths.append(data[o])
        o += 1

    for _ in range(num_to_add):
        if o + 1 > len(data):
            break
        b = data[o]
        model.intersections.append(IntersectionInfo(bool(b & 0x01), bool(b & 0x02)))
        o += 1

    return model


# ======================================================================
# Dispatch + JSON
# ======================================================================

def _looks_binary_nodes(name: str) -> bool:
    n = name.lower()
    base = n.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return base.startswith("nodes") and base.endswith(".dat")


def parse_dat(name: str, data: bytes):
    """Dispatch by filename (case-insensitive). ``data`` is raw bytes.

 NODES*.DAT -> binary path graph; everything else is decoded as latin-1 text
 and routed to its per-file parser (defaults to a generic token table).
 """
    base = name.lower().rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if _looks_binary_nodes(base):
        return parse_nodes(data)

    text = data.decode("latin-1") if isinstance(data, (bytes, bytearray)) else data
    if base == "handling.cfg":
        return parse_handling(text)
    if base.startswith("water") and base.endswith(".dat"):
        return parse_water(text)
    if base == "object.dat":
        return parse_object(text)
    if base.startswith("timecyc"):
        return parse_timecyc(text)
    if base == "popcycle.dat":
        return parse_popcycle(text)
    # generic fallback: a raw token table
    return parse_dat_table(text)


# ---- to_json --------------------------------------------------------------

def _car_json(c: HandlingCar) -> dict:
    return {
        "name": c.name,
        "fMass": c.fMass, "fTurnMass": c.fTurnMass, "fDragMult": c.fDragMult,
        "com": [c.com_x, c.com_y, c.com_z],
        "nPercentSubmerged": c.nPercentSubmerged,
        "fTractionMultiplier": c.fTractionMultiplier,
        "fTractionLoss": c.fTractionLoss, "fTractionBias": c.fTractionBias,
        "nGears": c.nGears, "fMaxVelocity": c.fMaxVelocity,
        "fEngineAccel": c.fEngineAccel, "fEngineInertia": c.fEngineInertia,
        "nDriveType": c.nDriveType, "nEngineType": c.nEngineType,
        "fBrakeDecel": c.fBrakeDecel, "fBrakeBias": c.fBrakeBias, "bABS": c.bABS,
        "fSteeringLock": c.fSteeringLock,
        "nMonetaryValue": c.nMonetaryValue,
        "ModelFlags": c.ModelFlags, "HandlingFlags": c.HandlingFlags,
        "FrontLights": c.FrontLights, "RearLights": c.RearLights,
        "AnimGroup": c.AnimGroup,
        "tokens": list(c.tokens),
    }


def _row_json(r: HandlingRow) -> dict:
    return {"lead": r.lead, "name": r.name, "tokens": list(r.tokens)}


def _obj_json(o: ObjectRecord) -> dict:
    d = {
        "name": o.name,
        "mass": o.mass, "turn_mass": o.turn_mass,
        "air_resistance": o.air_resistance, "elasticity": o.elasticity,
        "percent_submerged": o.percent_submerged, "uproot_limit": o.uproot_limit,
        "col_damage_mult": o.col_damage_mult,
        "col_damage_effect": o.col_damage_effect,
        "special_col_response": o.special_col_response,
        "camera_avoid": o.camera_avoid, "causes_explosion": o.causes_explosion,
        "fx_type": o.fx_type, "fx_offset": list(o.fx_offset),
        "effect_name": o.effect_name,
    }
    if o.smash_multiplier is not None:
        d["smash_multiplier"] = o.smash_multiplier
    if o.break_velocity is not None:
        d["break_velocity"] = list(o.break_velocity)
    if o.break_velocity_rand is not None:
        d["break_velocity_rand"] = o.break_velocity_rand
    if o.gun_break_mode is not None:
        d["gun_break_mode"] = o.gun_break_mode
    if o.sparks_on_impact is not None:
        d["sparks_on_impact"] = o.sparks_on_impact
    return d


def _node_json(n: PathNode) -> dict:
    return {
        "index": n.index, "x": n.x, "y": n.y, "z": n.z,
        "total_dist_from_origin": n.total_dist_from_origin,
        "base_link_id": n.base_link_id,
        "area_id": n.area_id, "node_id": n.node_id,
        "path_width": n.path_width, "flood_fill": n.flood_fill,
        "num_links": n.num_links, "on_dead_end": n.on_dead_end,
        "is_switched_off": n.is_switched_off, "road_blocks": n.road_blocks,
        "water_node": n.water_node,
        "spawn_probability": n.spawn_probability,
        "behaviour_type": n.behaviour_type,
    }


def _carlink_json(c: CarPathLink) -> dict:
    return {
        "index": c.index, "x": c.x, "y": c.y,
        "area_id": c.area_id, "node_id": c.node_id,
        "dir_x": c.dir_x, "dir_y": c.dir_y,
        "path_node_width": c.path_node_width,
        "num_opposite_lanes": c.num_opposite_lanes,
        "num_same_lanes": c.num_same_lanes,
        "traffic_light_dir": c.traffic_light_dir,
        "traffic_light_state": c.traffic_light_state,
        "bridge_lights": c.bridge_lights,
    }


def to_json(model) -> dict:
    """Render any parsed model as a JSON-serializable dict for the SAW table view."""
    if isinstance(model, Handling):
        return {
            "type": "handling",
            "cars": [_car_json(c) for c in model.cars],
            "bikes": [_row_json(r) for r in model.bikes],
            "flying": [_row_json(r) for r in model.flying],
            "boats": [_row_json(r) for r in model.boats],
            "anim_flags": [_row_json(r) for r in model.anim_flags],
        }
    if isinstance(model, Water):
        return {
            "type": "water",
            "quads": [{"corners": [list(c) for c in q.corners], "flag": q.flag}
                      for q in model.quads],
            "triangles": [{"corners": [list(c) for c in t.corners], "flag": t.flag}
                          for t in model.triangles],
        }
    if isinstance(model, ObjectData):
        return {"type": "object", "objects": [_obj_json(o) for o in model.objects]}
    if isinstance(model, Timecyc):
        return {
            "type": "timecyc",
            "rows": [{
                "ambient_rgb": list(r.ambient_rgb),
                "ambient_obj_rgb": list(r.ambient_obj_rgb),
                "dir_rgb": list(r.dir_rgb),
                "sky_top_rgb": list(r.sky_top_rgb),
                "sky_bot_rgb": list(r.sky_bot_rgb),
                "sun_core_rgb": list(r.sun_core_rgb),
                "sun_corona_rgb": list(r.sun_corona_rgb),
                "tokens": list(r.tokens),
            } for r in model.rows],
        }
    if isinstance(model, Popcycle):
        return {
            "type": "popcycle",
            "rows": [{
                "max_peds": r.max_peds, "max_cars": r.max_cars,
                "pct_dealers": r.pct_dealers, "pct_gang": r.pct_gang,
                "pct_cops": r.pct_cops, "pct_other": r.pct_other,
                "groups": list(r.groups),
                "values": list(r.values),
            } for r in model.rows],
        }
    if isinstance(model, Nodes):
        return {
            "type": "nodes",
            "num_nodes": model.num_nodes,
            "num_vehicle_nodes": model.num_vehicle_nodes,
            "num_ped_nodes": model.num_ped_nodes,
            "num_car_path_links": model.num_car_path_links,
            "num_addresses": model.num_addresses,
            "byte_size": model.byte_size,
            "nodes": [_node_json(n) for n in model.nodes],
            "car_path_links": [_carlink_json(c) for c in model.car_path_links],
            "links": [{"area_id": a.area_id, "node_id": a.node_id} for a in model.links],
            "navi_links": [{"car_path_link_id": a.car_path_link_id, "area_id": a.area_id}
                           for a in model.navi_links],
            "link_lengths": list(model.link_lengths),
            "intersections": [{"road_cross": x.road_cross,
                               "ped_traffic_light": x.ped_traffic_light}
                              for x in model.intersections],
        }
    if isinstance(model, list):  # generic token table
        return {"type": "table", "rows": [list(r) for r in model]}
    raise TypeError(f"to_json: unsupported model {type(model).__name__}")
