"""Public Railbound tile vocabulary (upstream-compatible integer encoding)."""

TRACK_NAMES = {
    0: 'empty', 1: 'horizontal', 2: 'vertical', 3: 'passenger_end_right',
    4: 'obstacle', 5: 'bottom_right', 6: 'bottom_left', 7: 'top_right',
    8: 'top_left', 9: 'bottom_right_left', 10: 'bottom_right_top',
    11: 'bottom_left_right', 12: 'bottom_left_top', 13: 'top_right_left',
    14: 'top_right_bottom', 15: 'top_left_right', 16: 'top_left_bottom',
    17: 'tunnel_left', 18: 'tunnel_right', 19: 'tunnel_down', 20: 'tunnel_up',
    21: 'freight_end_right', 22: 'freight_end_left',
    29: 'passenger_end_left', 30: 'passenger_end_down', 31: 'passenger_end_up',
    32: 'freight_end_down', 33: 'freight_end_up',
    34: 'station_building_left', 35: 'station_building_right',
    36: 'station_building_down', 37: 'station_building_up',
}
MOD_NAMES = {0:'none',1:'button',2:'tunnel_pair',3:'closed_gate',4:'open_gate',
             5:'button_switched_junction',6:'passenger_station',7:'self_switching_junction',
             8:'closed_semaphore',9:'deactivated',10:'car_start',11:'post_office'}
NUMBERED_MODIFIERS = {1,2,3,4,5,6,11}
DIRECTIONS = {'LEFT':(0,-1),'RIGHT':(0,1),'DOWN':(1,0),'UP':(-1,0)}
SIMPLE_TRACKS = {1,2,5,6,7,8}
PLACEABLE_TRACKS = SIMPLE_TRACKS | set(range(9,17))
TUNNELS = {17,18,19,20}
ENDINGS = {3,21,22,29,30,31,32,33}
# Endpoint directions. For junctions the first two endpoints are the preferred
# curved route, and the third is the straight branch.
PORTS = {1:('LEFT','RIGHT'),2:('UP','DOWN'),5:('DOWN','RIGHT'),6:('DOWN','LEFT'),
         7:('UP','RIGHT'),8:('UP','LEFT'),9:('DOWN','RIGHT','LEFT'),
         10:('DOWN','RIGHT','UP'),11:('DOWN','LEFT','RIGHT'),
         12:('DOWN','LEFT','UP'),13:('UP','RIGHT','LEFT'),
         14:('UP','RIGHT','DOWN'),15:('UP','LEFT','RIGHT'),16:('UP','LEFT','DOWN')}


def mechanisms(level):
    """Derive rules from actual contents, never trust decorative metadata."""
    tiles={v for row in level['board'] for v in row}
    mods={v for row in level['mods'] for v in row}
    found=[]
    if tiles & set(range(9,17)) or level['tracks']: found.append('junctions')
    if tiles & TUNNELS: found.append('tunnels')
    if mods & {1,3,4,5}: found.append('switches_and_gates')
    if 7 in mods: found.append('self_switching_junctions')
    if 6 in mods: found.append('passenger_stations')
    if 11 in mods: found.append('post_offices')
    if 8 in mods or level['semaphores']: found.append('semaphores')
    if sum(tile in ENDINGS for row in level['board'] for tile in row)>1: found.append('multiple_trains')
    if any(c['type']=='DECOY' for c in level['cars']): found.append('decoy_cars')
    return found
