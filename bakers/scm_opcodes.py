# Opcode table shared by the assembler and its tests.
# Mirrors src/game_sa/Scripts/ScriptCmd.h (keep in sync).
# name -> (hex_opcode, arg_spec)
# arg_spec chars: v=var(dest, tag 2/3) i=int f=float s=string8 a=address(@label, int32)
OPCODES = {
    "NOP":                (0x0000, ""),
    "WAIT":               (0x0001, "i"),
    "GOTO":               (0x0002, "a"),
    "SET_VAR_INT":        (0x0004, "vi"),
    "SET_VAR_FLOAT":      (0x0005, "vf"),
    "SET_LVAR_INT":       (0x0006, "vi"),
    "SET_LVAR_FLOAT":     (0x0007, "vf"),
    "ADD_INT_VAR":        (0x0008, "vi"),
    "ADD_FLOAT_VAR":      (0x0009, "vf"),
    "SUB_INT_VAR":        (0x000C, "vi"),
    "SUB_FLOAT_VAR":      (0x000D, "vf"),
    "MULT_INT_VAR":       (0x0010, "vi"),
    "IS_INT_VAR_GREATER":       (0x0018, "vi"),
    "IS_INT_VAR_GREATER_EQUAL": (0x0028, "vi"),
    "IS_INT_VAR_EQUAL":         (0x0038, "vi"),
    "IS_FLOAT_VAR_GREATER":     (0x0020, "vf"),
    "GOTO_IF_TRUE":       (0x004C, "a"),
    "GOTO_IF_FALSE":      (0x004D, "a"),
    "TERMINATE":          (0x004E, ""),
    "GOSUB":              (0x0050, "a"),
    "RETURN":             (0x0051, ""),
    "SCRIPT_NAME":        (0x03A4, "s"),
    "ANDOR":              (0x00D6, "i"),
    "SET_TIME_OF_DAY":    (0x00C0, "ii"),
    "FORCE_WEATHER":      (0x01B6, "i"),   # b609: param = SA weather TYPE 0..22 (SET_WEATHER_NOW)
    "RELEASE_WEATHER":    (0x01B7, ""),    # b609: back to the region cycle
    "PRINT_NOW":          (0x00BC, "si"),
    # demake player-implicit opcodes (0x7F00 block)
    "TELEPORT_PLAYER":    (0x7F01, "fff"),
    "SET_PLAYER_HEADING": (0x7F02, "f"),
    "ADD_PLAYER_MONEY":   (0x7F03, "i"),
    "HEAL_PLAYER":        (0x7F04, ""),
    "GET_PLAYER_COORDS":  (0x7F05, "vvv"),   # 3 output vars
    "IS_PLAYER_NEAR_2D":  (0x7F06, "fff"),   # x y radius -> condition
    "SPAWN_VEHICLE":      (0x7F07, "i"),     # model id, at hero
    "FORCE_INTERIOR":     (0x7F08, "i"),     # world door index
    "ADD_BLIP_FOR_COORD_XY": (0x7F09, "ffv"), # demake 2-float x y -> blip var (legacy)
    "REMOVE_BLIP":        (0x7F0A, "i"),     # blip id
    "DO_FADE":            (0x7F0B, "ii"),    # time_ms, mode(0 in / 1 out)
    "IS_FADING":          (0x7F0C, ""),
    "CREATE_OBJECT":      (0x7F0D, "fffv"),  # x y z -> object-id out var
    "DELETE_OBJECT":      (0x7F0E, "i"),     # object id
    "PRINT_STRING":       (0x7F0F, "si"),    # GXT-key(str8) duration_ms -> table lookup + show
    "SET_CITY_BARRIERS":  (0x7F10, "i"),     # present(1)=blocked / 0=open
    "SET_TIME_SCALE":     (0x7F11, "f"),     # day/night clock speed multiplier
    "GET_WEATHER":        (0x7F12, "v"),     # -> current weather out var
    "PLAY_SOUND":         (0x7F13, "iifff"), # bank sound x y z -> one-shot SFX
    "MISSION_START":      (0x7F14, "s"),     # name(str8) -> raise ON_MISSION
    "SET_OBJECTIVE":      (0x7F15, "s"),     # GXT-key(str8) -> objective line
    "MISSION_PASSED":     (0x7F16, ""),      # green banner + clear mission
    "MISSION_FAILED":     (0x7F17, "s"),     # GXT-key(str8) reason -> red banner + clear
    # scripted camera (SA ids)
    "SET_FIXED_CAMERA_POSITION": (0x015F, "ffffff"),
    "POINT_CAMERA_AT_POINT":     (0x0160, "fffi"),
    "RESTORE_CAMERA":            (0x015A, ""),
    # pickups (SA ids)
    "CREATE_PICKUP":             (0x0213, "iifffv"),   # model type x y z -> handle
    "HAS_PICKUP_BEEN_COLLECTED": (0x0214, "i"),        # handle -> condition
    # --- SA-canonical mission ABI: original main.scm missions port 1:1 ---
    "ADD_BLIP_FOR_COORD":        (0x018A, "fffv"),     # X Y Z -> blip handle var
    "LOCATE_PLAYER_ANY_MEANS_2D":(0x00E3, "iffffi"),   # player X Y rX rY drawSphere -> cond (box)
    "LOCATE_PLAYER_ANY_MEANS_3D":(0x00F5, "iffffffi"), # player X Y Z rX rY rZ drawSphere -> cond
    "PRINT_BIG":                 (0x00BA, "sii"),       # GXT-key time style -> big text
    "DECLARE_MISSION_FLAG":      (0x0180, "v"),         # bind $ONMISSION global cell
    "REGISTER_MISSION_PASSED":   (0x0318, "s"),         # GXT mission-name -> pass banner
    "PLAY_MISSION_PASSED_TUNE":  (0x0394, "i"),         # jingle id
    "PRINT_HELP":                (0x03E5, "s"),         # GXT key -> top-left help box (mission hint)
    "CLEAR_HELP":                (0x03E6, ""),          # dismiss the help box
}

# Arg type tags (mirror ScriptParam.h)
TAG_INT32, TAG_GVAR, TAG_LVAR, TAG_INT8, TAG_INT16, TAG_FLOAT = 1, 2, 3, 4, 5, 6
TAG_STR8 = 9

GLOBAL_BASE = 8   # global var space starts at file offset 8 (after the 's' chunk header)
