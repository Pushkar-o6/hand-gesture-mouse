import ctypes
from ctypes import wintypes
import time

user32 = ctypes.windll.user32

# Structs for SendInput
class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)))

class KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)))

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg", wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD))

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = (("ki", KEYBDINPUT),
                    ("mi", MOUSEINPUT),
                    ("hi", HARDWAREINPUT))
    _anonymous_ = ("_input",)
    _fields_ = (("type", wintypes.DWORD),
                ("_input", _INPUT))

# Constants
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

# Virtual screen metrics
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

# Virtual-Key codes
VK_TAB = 0x09
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12 # ALT
VK_OEM_PLUS = 0xBB 
VK_OEM_MINUS = 0xBD

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002

def _send_input(inputs):
    nInputs = len(inputs)
    LPINPUT = INPUT * nInputs
    pInputs = LPINPUT(*inputs)
    cbSize = ctypes.c_int(ctypes.sizeof(INPUT))
    return user32.SendInput(nInputs, pInputs, cbSize)

def send_mouse_input(flags, dx=0, dy=0, data=0):
    x = INPUT(type=INPUT_MOUSE,
              mi=MOUSEINPUT(dx, dy, data, flags, 0, None))
    return _send_input([x])

def move_cursor(x, y):
    v_width = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    v_height = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    v_left = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    v_top = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)

    if v_width == 0 or v_height == 0:
        return

    x_calc = int(((x - v_left) * 65536) / v_width)
    y_calc = int(((y - v_top) * 65536) / v_height)

    send_mouse_input(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, dx=x_calc, dy=y_calc)

def mouse_down(button='left'):
    if button == 'left':
        send_mouse_input(MOUSEEVENTF_LEFTDOWN)
    elif button == 'right':
        send_mouse_input(MOUSEEVENTF_RIGHTDOWN)

def mouse_up(button='left'):
    if button == 'left':
        send_mouse_input(MOUSEEVENTF_LEFTUP)
    elif button == 'right':
        send_mouse_input(MOUSEEVENTF_RIGHTUP)

def click(button='left'):
    mouse_down(button)
    mouse_up(button)

def right_click():
    click('right')

def scroll(amount):
    # Windows expects WHEEL_DELTA (120) multiples. PyAutoGUI might use different scales, but we'll scale here.
    # amount is typically clicks * 120. If amount is in "lines", multiply by 120.
    send_mouse_input(MOUSEEVENTF_WHEEL, data=int(amount * 120))

VK_MAP = {
    'alt': VK_MENU,
    'tab': VK_TAB,
    'shift': VK_SHIFT,
    'ctrl': VK_CONTROL,
    '+': VK_OEM_PLUS,
    '-': VK_OEM_MINUS,
    'esc': 0x1B,
    'enter': 0x0D,
    'win': 0x5B
}

def _resolve_vk(vk):
    if isinstance(vk, str):
        return VK_MAP.get(vk.lower(), ord(vk.upper()) if len(vk) == 1 else 0)
    return vk

def _key_input(vk, down=True):
    vk = _resolve_vk(vk)
    flags = 0
    if not down:
        flags |= KEYEVENTF_KEYUP
    x = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=None))
    return _send_input([x])

def key_down(vk):
    _key_input(vk, True)

def key_up(vk):
    _key_input(vk, False)

def press(vk):
    key_down(vk)
    key_up(vk)

def hotkey(*keys):
    vks = [_resolve_vk(k) for k in keys]
    for vk in vks:
        if vk: key_down(vk)
    for vk in reversed(vks):
        if vk: key_up(vk)
