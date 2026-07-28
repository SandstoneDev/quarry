"""Generate the Discord-invite QR for the qrsplash mod.
Outputs: qr.bin (u8 size + row-major packed bits, MSB first) + qr_check.png (verify)."""
import qrcode, struct, sys, os

URL = "https://discord.gg/wV8x46ZTDu"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."

qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, border=0)
qr.add_data(URL)
qr.make(fit=True)
m = qr.get_matrix()
n = len(m)
print(f"QR version={qr.version} modules={n}x{n}")

# qr.bin: u8 n, then per row ceil(n/8) bytes, bit7-first, 1 = black module
rows = []
for y in range(n):
    b = bytearray((n + 7) // 8)
    for x in range(n):
        if m[y][x]:
            b[x >> 3] |= 0x80 >> (x & 7)
    rows.append(bytes(b))
blob = struct.pack("<B", n) + b"".join(rows)
with open(os.path.join(OUT, "qr.bin"), "wb") as f:
    f.write(blob)
print(f"qr.bin = {len(blob)} bytes")

# verification PNG (10x scale + 4-module quiet zone)
img = qrcode.make(URL, error_correction=qrcode.constants.ERROR_CORRECT_M, border=4)
img.save(os.path.join(OUT, "qr_check.png"))
print("qr_check.png saved")
