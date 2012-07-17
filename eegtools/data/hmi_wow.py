#!/usr/bin/env python
import argparse, logging
from collections import namedtuple
import numpy as np

# TODO: 
# - diff scan_code and key_id?

LOG_MOUSE_EVENTS = {
  512 : 'mouse movement',
  513 : 'mouse left pressed',
  514 : 'mouse left released',
  516 : 'mouse right pressed',
  517 : 'mouse right released',
  519 : 'mouse scroll button pressed',
  520 : 'mouse scroll button released',
  522 : 'mouse scroll'}

EVENTS = {
  8: ('key pressed [Back]', 8),
  9: ('key pressed [Tab]', 9),
  13: ('key pressed [Return]', 13),
  20: ('key pressed [Capital]', 20),
  27: ('key pressed [Escape]', 27),
  32: ('key pressed [Space]', 32),
  35: ('key pressed [End]', 35),
  36: ('key pressed [Home]', 36),
  37: ('key pressed [Left]', 37),
  38: ('key pressed [Up]', 38),
  39: ('key pressed [Right]', 39),
  40: ('key pressed [Down]', 40),
  45: ('key pressed [Insert]', 45),
  46: ('key pressed [Delete]', 46),
  48: ('key pressed [0]', 48),
  49: ('key pressed [1]', 49),
  50: ('key pressed [2]', 50),
  51: ('key pressed [3]', 51),
  52: ('key pressed [4]', 52),
  53: ('key pressed [5]', 53),
  54: ('key pressed [6]', 54),
  55: ('key pressed [7]', 55),
  56: ('key pressed [8]', 56),
  57: ('key pressed [9]', 57),
  65: ('key pressed [A]', 65),
  66: ('key pressed [B]', 66),
  67: ('key pressed [C]', 67),
  68: ('key pressed [D]', 68),
  69: ('key pressed [E]', 69),
  70: ('key pressed [F]', 70),
  71: ('key pressed [G]', 71),
  72: ('key pressed [H]', 72),
  73: ('key pressed [I]', 73),
  74: ('key pressed [J]', 74),
  75: ('key pressed [K]', 75),
  76: ('key pressed [L]', 76),
  77: ('key pressed [M]', 77),
  78: ('key pressed [N]', 78),
  79: ('key pressed [O]', 79),
  80: ('key pressed [P]', 80),
  81: ('key pressed [Q]', 81),
  82: ('key pressed [R]', 82),
  83: ('key pressed [S]', 83),
  84: ('key pressed [T]', 84),
  85: ('key pressed [U]', 85),
  86: ('key pressed [V]', 86),
  87: ('key pressed [W]', 87),
  88: ('key pressed [X]', 88),
  89: ('key pressed [Y]', 89),
  90: ('key pressed [Z]', 90),
  91: ('key pressed [Lwin]', 91),
  112: ('key pressed [F1]', 112),
  113: ('key pressed [F2]', 113),
  114: ('key pressed [F3]', 114),
  115: ('key pressed [F4]', 115),
  116: ('key pressed [F5]', 116),
  117: ('key pressed [F6]', 117),
  118: ('key pressed [F7]', 118),
  119: ('key pressed [F8]', 119),
  120: ('key pressed [F9]', 120),
  121: ('key pressed [F10]', 121),
  122: ('key pressed [F11]', 122),
  123: ('key pressed [F12]', 123),
  144: ('key pressed [Numlock]', 144),
  160: ('key pressed [Lshift]', 160),
  161: ('key pressed [Rshift]', 161),
  162: ('key pressed [Lcontrol]', 162),
  163: ('key pressed [Rcontrol]', 163),
  164: ('key pressed [Lmenu]', 164),
  165: ('key pressed [Rmenu]', 165),
  166: ('key pressed [Browser_Back]', 166),
  173: ('key pressed [Volume_Mute]', 173),
  174: ('key pressed [Volume_Down]', 174),
  175: ('key pressed [Volume_Up]', 175),
  187: ('key pressed [Oem_Plus]', 187),
  188: ('key pressed [Oem_Comma]', 188),
  189: ('key pressed [Oem_Minus]', 189),
  190: ('key pressed [Oem_Period]', 190),
  191: ('key pressed [Oem_2]', 191),
  192: ('key pressed [Oem_3]', 192),
  220: ('key pressed [Oem_5]', 220),
  222: ('key pressed [Oem_7]', 222),
  255: ('key pressed [None]', 255),
  513: ('mouse left pressed', 1),
  514: ('mouse left released', 2),
  516: ('mouse right pressed', 4),
  517: ('mouse right released', 5),
  519: ('mouse scroll button pressed', 7),
  520: ('mouse scroll button released', 8),
  522: ('mouse scroll', 10)}
   

log = logging.getLogger(__name__)

Event = namedtuple('Event', 'window desc time marker code x y wheel'.split())

def line_to_event(line):
  field = line.strip().split('\t')

  if field[-1] == 'key':
    time, marker, key_id, scan_code, key_name, _, window, _ = field
    return Event(window=window, desc='key pressed [%s]' % key_name, 
      time=int(time), marker=int(marker), code=int(key_id), x=None, y=None, 
      wheel=None)
    
  elif field[-1] == 'mouse':
    time, marker, code, pos, wheel, _, window, _ = field
    x, y = tuple(map(int, pos.strip('()').split(', ')))
    return Event(window=window, desc=LOG_MOUSE_EVENTS[int(code)],
      time=int(time), marker=int(marker), code=int(code), x=x, y=y, 
      wheel=int(wheel))
  else:
    logging.error('Event not recognized: "%s"!' % line)


def sanitize_logged_events(events):
  events = sorted([e for e in events if e], key=lambda e: e.time)
  events = [e for e in events if e.window == 'World of Warcraft']
  return events


def events_to_array(events):
  return np.asarray([(e.code, e.time, e.time) for e in events]).T


if __name__ == '__main__':
  from pprint import pprint
  logging.basicConfig()

  # CLI:
  p = argparse.ArgumentParser()
  p.add_argument('logs', nargs='+')
  args = p.parse_args()

  # Read all events:
  events = []
  for log in args.logs:
    for line in open(log):
      events.append(line_to_event(line))

  # Display event definition for copy-pasting:
  rosetta = set([(e.code, (e.desc, e.marker)) for e in events if e])
  assert len(set(zip(*rosetta)[0])) == len(rosetta), 'Event codes not unique!'
  pprint(dict(rosetta))

  print events_to_array(sanitize_logged_events(events))
