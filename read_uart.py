"""
read_uart.py -- Read AxMAC on-chip logit results from the 野火征途 board via UART.

Protocol: 0xAB | logit[0..9] as 10 × int32 little-endian = 41 bytes per packet.
Usage:
    py -3.14 read_uart.py              # auto-detect CH340, single capture
    py -3.14 read_uart.py --port COM3  # explicit port
    py -3.14 read_uart.py --loop       # keep reading (useful across multiple burns)
"""
import argparse
import struct
import sys

HEADER   = 0xAB
N_LOGITS = 10
PKT_LEN  = 1 + N_LOGITS * 4   # 41 bytes


def find_ch340_port():
    import serial.tools.list_ports
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").upper()
        if "CH340" in desc or "CH341" in desc or "USB-SERIAL" in desc:
            return p.device
    return None


def read_packet(ser):
    """Sync on 0xAB header, then read 40 bytes of logit data."""
    while True:
        b = ser.read(1)
        if not b:
            raise TimeoutError("No data received from board.")
        if b[0] == HEADER:
            break
    payload = ser.read(N_LOGITS * 4)
    if len(payload) < N_LOGITS * 4:
        raise IOError(f"Short read: got {len(payload)} bytes, expected {N_LOGITS * 4}")
    return struct.unpack('<' + 'i' * N_LOGITS, payload)


def display(logits, label=""):
    argmax = max(range(len(logits)), key=lambda i: logits[i])
    tag = f"  [{label}]" if label else ""
    print(f"\nOn-chip result{tag}:")
    for i, v in enumerate(logits):
        marker = " <-- argmax" if i == argmax else ""
        print(f"  logit[{i}] = {v:10d}{marker}")
    print(f"  Predicted class: {argmax}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",  default=None, help="COM port (e.g. COM3)")
    parser.add_argument("--baud",  type=int, default=115200)
    parser.add_argument("--loop",  action="store_true", help="keep reading packets")
    parser.add_argument("--label", default="", help="config label shown in output")
    args = parser.parse_args()

    try:
        import serial
    except ImportError:
        sys.exit("pyserial not installed. Run: py -3.14 -m pip install pyserial")

    port = args.port or find_ch340_port()
    if port is None:
        sys.exit("CH340 port not found. Connect the USB-UART cable and specify --port COMx.")

    print(f"Opening {port} at {args.baud} baud ...")
    with serial.Serial(port, args.baud, timeout=5) as ser:
        print("Waiting for board packet (reset board if nothing arrives) ...")
        if args.loop:
            count = 0
            while True:
                try:
                    logits = read_packet(ser)
                    count += 1
                    display(logits, label=f"{args.label} #{count}")
                except (TimeoutError, IOError) as e:
                    print(f"[warn] {e}")
        else:
            logits = read_packet(ser)
            display(logits, label=args.label)


if __name__ == "__main__":
    main()
