"""Top-down PNG boards, solutions and validator snapshots.

CLI: python -m make_images levels/extended-hard --recursive
     python -m make_images solutions/extended-hard --recursive --levels-root levels/extended-hard
"""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from validator.tiles import PORTS, DIRECTIONS, TUNNELS, ENDINGS


# Layout dimensions use logical pixels; drawing dimensions use 1/100 of a cell.
# Every field can be overridden with a style dict or JSON file.
DEFAULT_STYLE = {
    'cell_size': 112, 'margin': 42, 'scale': 2,
    'background': '#f8f6ee', 'cell': '#e9eedc', 'cell_alternate': '#e5ebd7',
    'grid': '#d1dac1', 'board_border': '#b8c5a8', 'shadow': '#ccd3be',
    'ink': '#243d43', 'muted_ink': '#63746d', 'white': '#fffdf5',
    'original_track': '#60787b', 'rail_highlight': '#dce6df',
    'rail_shadow': '#364f53', 'ballast': '#d1d5c4', 'sleeper': '#8d9581',
    'added_track': '#b47a3e', 'added_rail_highlight': '#f4cd81',
    'added_sleeper': '#b99666', 'added_cell': '#f1e7ca',
    'normal_car': '#258bab', 'normal_roof': '#51b5c7',
    'numeral_car': '#8761aa', 'numeral_roof': '#b399d0',
    'decoy_car': '#e7b64b', 'decoy_roof': '#f6d477',
    'window': '#d8f1ed', 'wheel': '#30464a', 'metal': '#8ea0a0',
    'passenger': '#257e9e', 'freight': '#805aa3',
    'skin': '#f1bf91', 'hair': '#514335', 'luggage': '#b97945',
    'signal_unlit': '#4a5b57', 'switch_linkage': '#a19877',
    'platform': '#faf5de', 'platform_edge': '#bfab77', 'parcel': '#dfb880',
    'gate_closed': '#da655b', 'gate_open': '#38946b', 'error': '#da384e',
    'tunnel_stone': '#899b92', 'tunnel_highlight': '#aebbb0',
    'tunnel_mouth': '#263f42', 'obstacle': '#9cad93',
    'obstacle_light': '#b5c5a5', 'obstacle_dark': '#879a7b',
    'roof': '#b87c66', 'building': '#faf1d7',
    'panel_background': '#fffdf6',
    'groups': ['#a85879', '#337e74', '#556fa7', '#a26f2b', '#815fa5', '#367d94'],
    'rail_gauge': 17, 'rail_width': 4.5, 'rail_glint_width': 1.4,
    'ballast_width': 29, 'sleeper_length': 29, 'sleeper_width': 5,
    'sleeper_spacing': 12, 'curve_segments': 32,
    'inactive_dash_length': 6, 'inactive_dash_gap': 5,
    'car_scale': 1.0, 'engine_scale': 1.0, 'mechanism_scale': 1.0,
    'occupied_button_scale': .62,
    'vehicle_outline_width': 1.7, 'vehicle_label_size': 18,
    'direction_arrow_length': 19, 'direction_arrow_width': 18, 'direction_arrow_front': 35,
    'coordinate_size': 12, 'board_shadow_offset': 4,
    'motion_trails': True, 'ghost_opacity': .30, 'ghost_ink': '#687b86',
    'motion_color': '#df514a', 'motion_width': 3.3, 'motion_tilt': 10,
    'collision_car_scale': .86, 'impact_size': 20, 'impact_core': '#ffbf57',
}
TUNNEL_DIRECTIONS = {17: 'LEFT', 18: 'RIGHT', 19: 'DOWN', 20: 'UP'}
END_DIRECTIONS = {3:'RIGHT', 21:'RIGHT', 22:'LEFT', 29:'LEFT', 30:'DOWN', 31:'UP', 32:'DOWN', 33:'UP'}
OPPOSITE = {'LEFT':'RIGHT', 'RIGHT':'LEFT', 'UP':'DOWN', 'DOWN':'UP'}


@lru_cache(maxsize=128)
def _font(size, bold=False):
    names = ('DejaVuSans-Bold.ttf','C:/Windows/Fonts/arialbd.ttf') if bold else ('DejaVuSans.ttf','C:/Windows/Fonts/arial.ttf')
    for name in names:
        try:
            return ImageFont.truetype(name, max(1,int(size)))
        except OSError:
            pass
    return ImageFont.load_default()


def _roman(number):
    number += 1
    parts = []
    for value, symbol in ((1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')):
        count, number = divmod(number, value)
        parts.append(symbol*count)
    return ''.join(parts)


def _car_label(car):
    if car['type'] in ('DECOY','CRASHED'):
        return 'D'+str(car.get('num',0)+1)
    return _roman(car['num']) if car['type']=='NUMERAL' else str(car['num']+1)


class _Pen:
    """Cell-local direction-aware drawing primitives."""
    def __init__(self, draw, style, cx, cy, unit, direction='RIGHT'):
        self.d, self.s, self.cx, self.cy, self.u = draw, style, cx, cy, unit
        self.dy, self.dx = DIRECTIONS[direction]

    def point(self,x,y):
        return self.cx+(x*self.dx-y*self.dy)*self.u, self.cy+(x*self.dy+y*self.dx)*self.u

    def color(self,value):
        return self.s.get(value,value) if value is not None else None

    def line(self,points,color,width=1):
        self.d.line([self.point(*p) for p in points],fill=self.color(color),width=max(1,round(width*self.u)),joint='curve')

    def polygon(self,points,fill,outline=None,width=1):
        pts=[self.point(*p) for p in points]
        self.d.polygon(pts,fill=self.color(fill))
        if outline:
            self.d.line(pts+pts[:1],fill=self.color(outline),width=max(1,round(width*self.u)),joint='curve')

    def rect(self,box,fill,outline=None,radius=0,width=1):
        if self.dx and self.dy:
            self.polygon([(box[0],box[1]),(box[2],box[1]),(box[2],box[3]),(box[0],box[3])],fill,outline,width)
            return
        a,b=self.point(box[0],box[1]),self.point(box[2],box[3])
        box=tuple(round(v) for v in (min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1])))
        rr=max(0,min(round(radius*self.u),(box[2]-box[0]-2)//2,(box[3]-box[1]-2)//2))
        self.d.rounded_rectangle(box,radius=rr,fill=self.color(fill),outline=self.color(outline),width=max(1,round(width*self.u)))

    def circle(self,x,y,radius,fill,outline=None,width=1):
        px,py=self.point(x,y)
        rr=radius*self.u
        self.d.ellipse((px-rr,py-rr,px+rr,py+rr),fill=self.color(fill),outline=self.color(outline),width=max(1,round(width*self.u)))

    def text(self,x,y,value,size=12,fill='ink',bold=True):
        px,py=self.point(x,y)
        font=_font(round(size*self.u),bold)
        box=self.d.textbbox((0,0),str(value),font=font)
        self.d.text((px-(box[2]-box[0])/2,py-(box[3]-box[1])/2-box[1]),str(value),fill=self.color(fill),font=font)


def _vehicle(p,kind='NORMAL',label='1',engine=False):
    """Blunt rear, pointed front and a broad arrow painted inside the nose."""
    base,roof=('numeral_car','numeral_roof') if kind=='NUMERAL' else (('decoy_car','decoy_roof') if kind in ('DECOY','CRASHED') else ('normal_car','normal_roof'))
    outline=p.s['vehicle_outline_width']
    p.rect((-35,-24,36,28),'shadow',radius=9)
    for x in (-23,20):
        p.rect((x-7,-27,x+7,-18),'wheel',radius=3)
        p.rect((x-7,18,x+7,27),'wheel',radius=3)
        p.line([(x-3,-25),(x+3,-25)],'metal',1.4)
        p.line([(x-3,25),(x+3,25)],'metal',1.4)
    p.rect((-40,-5,-29,5),'wheel',radius=2)
    if engine:
        # Glazed cab, cylindrical boiler, chimney and slatted cowcatcher.
        p.polygon([(25,-19),(41,-13),(46,0),(41,13),(25,19)],base,'ink',outline)
        for yy in (-10,0,10):
            p.line([(30,yy),(42,yy*.65)],'rail_highlight',1.5)
        p.rect((-35,-23,-6,23),base,'ink',4,outline)
        p.rect((-30,-17,-13,17),'window','ink',2,1)
        p.line([(-22,-17),(-22,17)],base,2)
        p.rect((-12,-16,29,16),base,'ink',10,outline)
        p.line([(-6,-10),(19,-10)],roof,4)
        p.line([(0,-15),(0,15)],roof,2)
        p.circle(15,0,8,'wheel','metal',2)
        p.circle(15,0,3,'ink')
        p.circle(-8,0,5,roof,'ink',1)
    elif kind in ('DECOY','CRASHED'):
        p.polygon([(-32,-21),(22,-21),(39,0),(22,21),(-32,21)],base,'ink',outline)
        p.rect((-26,-16,12,16),roof,'ink',2,1)
        p.line([(-25,-15),(10,15)],base,3)
        p.line([(-25,15),(10,-15)],base,3)
        p.line([(-23,-20),(-23,20)],'ink',2)
        p.line([(-30,-14),(-16,-14)],'ink',3)
    else:
        p.polygon([(-33,-22),(22,-22),(39,0),(22,22),(-33,22)],base,'ink',outline)
        p.rect((-28,-16,18,16),roof,radius=5)
        if kind=='NUMERAL':
            for xx in (-22,-11,0,11):
                p.line([(xx,-15),(xx,15)],base,1.4)
        else:
            for xx in (-22,-7,8):
                p.rect((xx-4,-20,xx+5,-14),'window',radius=1)
                p.rect((xx-4,14,xx+5,20),'window',radius=1)
        p.line([(-27,-11),(11,-11)],'rail_highlight',1.2)
    if not engine:
        p.circle(-7,0,12.5 if len(label)<3 else 14,base)
        p.text(-7,0,label,p.s['vehicle_label_size'] if len(label)<3 else 14,'ink' if kind in ('DECOY','CRASHED') else 'white')
    front=p.s['direction_arrow_front']+(7 if engine else 0)
    length,width=p.s['direction_arrow_length'],p.s['direction_arrow_width']
    p.polygon([(front,0),(front-length*.65,-width/2),(front-length*.65,-width/4),(front-length,-width/4),(front-length,width/4),(front-length*.65,width/4),(front-length*.65,width/2)],'white',base,.6)


def _person(p,x=0,y=0,color='passenger'):
    # A waving traveller with a face, coat, shoes and a separate suitcase.
    p.line([(x+7,y+3),(x+7,y),(x+10,y),(x+10,y+3)],'ink',1)
    p.rect((x+6,y+3,x+11,y+9),'luggage','ink',1.5,.8)
    p.line([(x+8.5,y+4),(x+8.5,y+8)],'parcel',1)
    p.line([(x-2,y+4),(x-3,y+9)],'ink',2.5)
    p.line([(x+2,y+4),(x+3,y+9)],'ink',2.5)
    p.line([(x-5,y+9),(x-2,y+9)],'ink',1.8)
    p.line([(x+2,y+9),(x+5,y+9)],'ink',1.8)
    p.line([(x-3,y),(x-7,y-2),(x-8,y-6)],color,2.7)
    p.circle(x-8,y-7,1.7,'skin')
    p.line([(x+3,y),(x+6,y+5)],color,2.7)
    p.circle(x+6,y+5,1.5,'skin')
    p.rect((x-3.5,y-1,x+3.5,y+5),color,radius=1.8)
    p.line([(x-1,y),(x+1,y)],'white',1)
    p.circle(x,y-6,4,'skin','hair',.7)
    p.line([(x-3,y-8),(x,y-10),(x+3,y-8)],'hair',1.8)
    p.circle(x+1.4,y-5.7,.55,'ink')


def _parcel(p,x=0,y=0):
    p.rect((x-11,y-10,x+11,y+10),'parcel','freight',2,1.5)
    p.rect((x-2.5,y-9,x+2.5,y+9),'platform')
    p.line([(x-10,y-4),(x+10,y-4)],'freight',1)


def _button(p,group,color):
    p.rect((-18,-14,18,18),'metal','ink',5,1.5)
    p.circle(0,0,14,'ink')
    p.circle(0,-3,12,color,'white',1.5)
    p.text(0,-1,group,13,'white')
    for xx in (-14,14):
        p.circle(xx,13,1.5,'ink')


def _visible_button_pen(p,tile,car,factor):
    """Keep a covered press plate visible on one rail endpoint in its own cell."""
    dy,dx=DIRECTIONS.get(car.get('direction'),(0,1))
    def clearance(side):
        sy,sx=DIRECTIONS[side]
        along=38*(sx*dx+sy*dy);across=38*(sx*(-dy)+sy*dx)
        margin=max(abs(across)-28,-along-35,along-40)
        return margin,-along
    side=max(PORTS.get(tile,('LEFT',)),key=clearance)
    sy,sx=DIRECTIONS[side];cx,cy=p.point(38*sx,38*sy)
    return _Pen(p.d,p.s,cx,cy,p.u*factor)


def _junction_frame(p,tile):
    """Use a stable frame with the common stem pointing down."""
    ports=PORTS.get(tile,())
    if len(ports)!=3:
        # Illegal answers can retain a switch on a non-junction tile.
        return p,1
    stem=next(side for side in ports if OPPOSITE[side] not in ports)
    q=_Pen(p.d,p.s,p.cx,p.cy,p.u,{'DOWN':'RIGHT','UP':'LEFT','RIGHT':'UP','LEFT':'DOWN'}[stem])
    branch=next(side for side in ports[:2] if side!=stem)
    dy,dx=DIRECTIONS[branch]
    return q,dx*q.dx+dy*q.dy


def _junction_control(p,tile,group=None,color='gate_open'):
    """A side-mounted points lever; the rail geometry shows its selection."""
    q,sign=_junction_frame(p,tile)
    cx,cy=q.point(-33,-34)
    lever=_Pen(p.d,p.s,cx,cy,p.u*p.s['mechanism_scale'])
    lever.dx,lever.dy=q.dx,q.dy
    lever.rect((-8,-5,8,5),'metal','ink',3,1)
    lever.circle(0,0,4,'ink','white',.8)
    end=(sign*7,-7)
    lever.line([(0,0),end],'ink',5)
    lever.line([(0,0),end],'metal',2.5)
    lever.circle(*end,8,color,'white',1.2)
    if group is not None:
        lever.text(*end,group,10,'white')
    else:
        # Two curved arrows on the handle identify the automatic return action.
        ex,ey=end
        lever.line([(ex-4,ey-1),(ex-2,ey-4),(ex+2,ey-4),(ex+4,ey-1)],'white',1.1)
        lever.polygon([(ex+4,ey-1),(ex+1,ey-2),(ex+4,ey-4)],'white')
        lever.line([(ex+4,ey+1),(ex+2,ey+4),(ex-2,ey+4),(ex-4,ey+1)],'white',1.1)
        lever.polygon([(ex-4,ey+1),(ex-1,ey+2),(ex-4,ey+4)],'white')


def _gate(p,group,closed,color):
    # Local x is along the track; the lowered barrier is perpendicular to it.
    p.rect((-7,-34,8,-20),'metal','ink',3,1.4)
    p.circle(0,-27,3.5,'gate_closed' if closed else 'gate_open')
    a,b=(0,-24),((0,29) if closed else (30,-24))
    p.line([a,b],'ink',8)
    p.line([a,b],'white',6)
    length=math.dist(a,b)
    for start in range(1,round(length),13):
        t0,t1=start/length,min((start+6)/length,1)
        p.line([(a[0]+(b[0]-a[0])*t0,a[1]+(b[1]-a[1])*t0),(a[0]+(b[0]-a[0])*t1,a[1]+(b[1]-a[1])*t1)],'gate_closed' if closed else 'gate_open',6)
    if closed:
        p.rect((-4,28,4,34),'metal','ink',2)
    p.circle(0,-40,9,color,'white',1)
    p.text(0,-40,group,10,'white')


def _semaphore(p,opened=False):
    # Two-aspect signal head, plus an arm that clears the rails when raised.
    p.rect((-34,-19,-20,-12),'metal','ink',2,1)
    p.line([(-27,-16),(-27,-38)],'metal',4)
    p.rect((-37,-46,-17,-15),'ink','metal',7,1)
    p.circle(-27,-36,6.5,'signal_unlit' if opened else 'gate_closed','white',.9)
    p.circle(-27,-23,6.5,'gate_open' if opened else 'signal_unlit','white',.9)
    p.circle(-29,-38 if not opened else -25,1.5,'white')
    p.line([(-19,-23),(-10,-23)],'metal',4)
    a=(-10,-23)
    b=(18,-40) if opened else (-10,25)
    p.line([a,b],'ink',8)
    p.line([a,b],'gate_open' if opened else 'gate_closed',5.5)
    p.line([(b[0]+(a[0]-b[0])*.19,b[1]+(a[1]-b[1])*.19),b],'white',5.5)
    p.circle(*a,4,'metal','ink',1)


def _tunnel(p,group,color):
    # The local right edge is an open throat; rails disappear under a stone cap.
    p.rect((-32,-34,23,36),'shadow',radius=15)
    p.rect((-34,-34,14,32),'tunnel_stone','ink',14,1.3)
    p.rect((-23,-26,19,26),'tunnel_mouth',radius=8)
    p.rect((-34,-34,-12,32),'tunnel_stone',radius=8)
    p.rect((-20,-35,20,-24),'tunnel_highlight','tunnel_stone',3)
    p.rect((-20,24,20,35),'tunnel_highlight','tunnel_stone',3)
    for xx in (-18,-3,12):
        p.line([(xx,-34),(xx,-25)],'tunnel_stone',1.5)
        p.line([(xx,25),(xx,34)],'tunnel_stone',1.5)
    p.rect((-34,-15,-13,15),color,'white',5,1)
    p.text(-23.5,0,str(group),14,'white')
    p.line([(20,-24),(20,24)],'tunnel_highlight',3)


def _station(p,number,freight=False,served=False):
    color='freight' if freight else 'passenger'
    # The inner platform edge faces the rail; all waiting objects stay off it.
    p.rect((-38,-9,38,10),'shadow',radius=3)
    p.rect((-39,-10,38,9),'platform','platform_edge',3,1)
    p.line([(-37,-9),(36,-9)],color,2)
    for xx in range(-34,36,9):
        p.line([(xx,-9),(xx+4,-9)],'white',1.2)
    p.rect((-3,-18,10,-11),'platform','platform_edge',1,1)
    p.line([(-1,-15),(8,-15)],color,1)
    icon_x,icon_y=p.point(-17,0)
    if not freight:
        icon_x+=p.dy*p.u
    icon=_Pen(p.d,p.s,icon_x,icon_y,p.u)
    if served:
        # An empty bench and a check replace the traveller/parcel after pickup.
        p.rect((-31,-3,-5,3),'platform_edge',color,1,.8)
        p.line([(-27,3),(-27,6)],color,1.5)
        p.line([(-9,3),(-9,6)],color,1.5)
        p.circle(-18,0,7,'gate_open','white',.8)
        p.line([(-22,0),(-19,3),(-14,-3)],'white',2)
    elif freight:
        _parcel(_Pen(p.d,p.s,icon_x,icon_y,p.u*.8))
    else:
        _person(icon,0,0,color)
    p.circle(24,0,8.5,color)
    p.text(24,0,_roman(number) if freight else number+1,12,'white')


def _station_pen(p,tile):
    """Keep platforms beside the rail, with boarding steps facing inward."""
    if tile in PORTS and len(PORTS[tile])==3:
        # A junction uses three cell edges; its fourth edge is the only side
        # where a platform cannot cover an arriving or departing rail.
        side=next(side for side in DIRECTIONS if side not in PORTS[tile])
        y,x=DIRECTIONS[side]
        x,y=x*39,y*39
        direction={'UP':'LEFT','DOWN':'RIGHT','LEFT':'DOWN','RIGHT':'UP'}[side]
    elif tile==2:
        x,y,direction=39,0,'UP'
    elif tile in (5,6):
        x,y,direction=0,-39,'LEFT'
    else:
        x,y,direction=0,39,'RIGHT'
    cx,cy=p.point(x,y)
    return _Pen(p.d,p.s,cx,cy,p.u*p.s['mechanism_scale'],direction)


def _route(a,b,segments=32):
    """True quarter-circle curves meet adjacent rail endpoints tangentially."""
    ay,ax=DIRECTIONS[a]
    by,bx=DIRECTIONS[b]
    if ax+bx==0 and ay+by==0:
        return [(ax*50+(bx-ax)*50*i/segments,ay*50+(by-ay)*50*i/segments) for i in range(segments+1)]
    ox,oy=(ax+bx)*50,(ay+by)*50
    start,stop=math.atan2(ay*50-oy,ax*50-ox),math.atan2(by*50-oy,bx*50-ox)
    delta=(stop-start+math.pi)%(2*math.pi)-math.pi
    return [(ox+50*math.cos(start+delta*i/segments),oy+50*math.sin(start+delta*i/segments)) for i in range(segments+1)]


def _rail_mount(p,tile,factor=1):
    """Place a fixture at the actual rail midpoint and follow its tangent."""
    q=_Pen(p.d,p.s,p.cx,p.cy,p.u*factor,_track_direction(tile))
    if tile in (5,6,7,8):
        route=_route(*PORTS[tile])
        middle=len(route)//2
        a,b=route[middle-1],route[middle+1]
        length=math.dist(a,b)
        q.dx,q.dy=(b[0]-a[0])/length,(b[1]-a[1])/length
        q.cx,q.cy=p.point(*route[middle])
    return q


def _dashes(points,length,gap,distances):
    """Use centerline distances so both sides of a rail have matching dashes."""
    if length<=0 or gap<=0:
        raise ValueError('Rail dash length and gap must be positive')
    visible,remaining,stroke=True,length,[]
    for i,(a,b) in enumerate(zip(points,points[1:])):
        distance=distances[i+1]-distances[i]
        if distance<=1e-9:
            continue
        offset=0
        while offset<distance-1e-9:
            step=min(remaining,distance-offset)
            start=tuple(a[j]+(b[j]-a[j])*offset/distance for j in (0,1))
            end=tuple(a[j]+(b[j]-a[j])*(offset+step)/distance for j in (0,1))
            if visible:
                if not stroke:
                    stroke.append(start)
                stroke.append(end)
            offset+=step
            remaining-=step
            if remaining<=1e-9:
                if visible and stroke:
                    yield stroke
                    stroke=[]
                visible=not visible
                remaining=length if visible else gap
    if stroke:
        yield stroke


def _rail(p,points,added=False,draw_bed=True,draw_rails=True,dashed=False):
    s=p.s
    if draw_bed:
        p.line(points,'ballast',s['ballast_width'])
    normals,distances=[],[0]
    for i,point in enumerate(points):
        a,b=points[max(i-1,0)],points[min(i+1,len(points)-1)]
        length=math.dist(a,b) or 1
        normals.append((-(b[1]-a[1])/length,(b[0]-a[0])/length))
        if i:
            distances.append(distances[-1]+math.dist(points[i-1],point))
    if draw_bed:
        next_tie=s['sleeper_spacing']/2
        for i,distance in enumerate(distances):
            if distance<next_tie:
                continue
            xx,yy=points[i]
            nx,ny=normals[i]
            half=s['sleeper_length']/2
            p.line([(xx-nx*half,yy-ny*half),(xx+nx*half,yy+ny*half)],'added_sleeper' if added else 'sleeper',s['sleeper_width'])
            next_tie+=s['sleeper_spacing']
    if draw_rails:
        for side in (-1,1):
            rail=[(xx+nx*s['rail_gauge']/2*side,yy+ny*s['rail_gauge']/2*side) for (xx,yy),(nx,ny) in zip(points,normals)]
            strokes=_dashes(rail,s['inactive_dash_length'],s['inactive_dash_gap'],distances) if dashed else [rail]
            for stroke in strokes:
                p.line(stroke,'rail_shadow',s['rail_width']+1.4)
                p.line(stroke,'added_track' if added else 'original_track',s['rail_width'])
                p.line(stroke,'added_rail_highlight' if added else 'rail_highlight',s['rail_glint_width'])


def _junction_tracks(p,tile,added=False,switching=False):
    """Solid current routes; switching junctions add one dashed inactive bend."""
    ports=PORTS[tile]
    routes=[_route(ports[0],ports[1],p.s['curve_segments']),
            _route(ports[2],OPPOSITE[ports[2]],p.s['curve_segments'])]
    if switching:
        q,_=_junction_frame(p,tile)
        rod=[(-33,-34),(-33,-17),(-12,7),(0,30)]
        q.line(rod,'ink',3)
        q.line(rod,'switch_linkage',1.5)
    # Separate layers keep either rail bed from painting over the other route.
    for route in routes:
        _rail(p,route,added,draw_rails=False)
    if switching:
        stem=next(side for side in ports if OPPOSITE[side] not in ports)
        inactive=_route(stem,ports[2],p.s['curve_segments'])
        _rail(p,inactive,added,draw_bed=False,dashed=True)
    for route in routes:
        _rail(p,route,added,draw_bed=False)


def _track_direction(tile):
    if tile==2:
        return 'DOWN'
    ports=PORTS.get(tile)
    if ports and len(ports)==3:
        return ports[2]
    if ports and len(ports)==2 and tile!=1:
        return ports[0]
    return 'RIGHT'


def render(level,output_path,state=None,solution=None,style=None,*,annotations=None):
    """Write a PNG, returning Path. Snapshot values override solution values.

    Original-level rails stay steel; added rails are amber. Coordinates are
    zero-based [row,column]. Input objects are never mutated. Style accepts a
    dict or JSON path; see DEFAULT_STYLE for adjustable colors and geometry.
    Optional English annotations are for local diagnostic exports only.
    """
    if isinstance(level,(str,Path)):
        level=json.loads(Path(level).read_text(encoding='utf-8'))
    if isinstance(solution,(str,Path)):
        solution=json.loads(Path(solution).read_text(encoding='utf-8'))
    if solution is not None and 'add_tracks' in solution:
        from validator.answers import load_solution
        solution=load_solution(level,solution)
    if isinstance(style,(str,Path)):
        style=json.loads(Path(style).read_text(encoding='utf-8'))
    s=dict(DEFAULT_STYLE)
    s.update(style or {})
    cell,margin,scale=int(s['cell_size']),int(s['margin']),int(s['scale'])
    if not 40<=cell<=512 or not 1<=scale<=4 or margin<16:
        raise ValueError('Invalid render geometry')
    data=dict(level)
    data.update(solution or {})
    data.update(state or {})
    board=data['board']
    h,w=len(board),len(board[0])
    width=w*cell+margin*2
    board_height=h*cell+margin*2
    annotations=list(annotations or [])
    height=board_height+(len(annotations)*22+24 if annotations else 0)
    if h*w>65536 or width*height*scale*scale>100_000_000:
        raise ValueError('Image exceeds render resource limit')
    im=Image.new('RGB',(round(width*scale),round(height*scale)),s['background'])
    d=ImageDraw.Draw(im)
    c,m=cell*scale,margin*scale
    board_left=(width-w*cell)*scale/2

    def pen(y,x,direction='RIGHT',factor=1):
        return _Pen(d,s,board_left+(x+.5)*c,m+(y+.5)*c,c/100*factor,direction)

    def label(x,y,value,size,color='ink',bold=False):
        _Pen(d,s,x,y,scale).text(0,0,value,size,color,bold)

    shadow=s['board_shadow_offset']*scale
    d.rounded_rectangle((board_left-2*scale+shadow,m+shadow,board_left+w*c+2*scale+shadow,m+h*c+2*scale+shadow),radius=8*scale,fill=s['shadow'])
    for y,row in enumerate(board):
        for x,tile in enumerate(row):
            added=level['board'][y][x]==0 and tile!=0
            fill=s['added_cell'] if added else s['cell_alternate'] if (x+y)%2 else s['cell']
            d.rectangle((board_left+x*c,m+y*c,board_left+(x+1)*c,m+(y+1)*c),fill=fill)
    # Draw grid first so every rail remains continuous across cell boundaries.
    for x in range(w+1):
        d.line((board_left+x*c,m,board_left+x*c,m+h*c),fill=s['grid'],width=scale)
    for y in range(h+1):
        d.line((board_left,m+y*c,board_left+w*c,m+y*c),fill=s['grid'],width=scale)
    d.rectangle((board_left,m,board_left+w*c,m+h*c),outline=s['board_border'],width=scale)
    for x in range(w):
        label(board_left+(x+.5)*c,m*.48,x,s['coordinate_size'],'muted_ink',True)
    for y in range(h):
        label(board_left-m*.57,m+(y+.5)*c,y,s['coordinate_size'],'muted_ink',True)
    for y,row in enumerate(board):
        for x,tile in enumerate(row):
            p=pen(y,x)
            added=level['board'][y][x]==0 and tile!=0
            if tile in PORTS:
                ports=PORTS[tile]
                if len(ports)==3:
                    _junction_tracks(p,tile,added,switching=data['mods'][y][x] in (5,7))
                else:
                    _rail(p,_route(ports[0],ports[1],s['curve_segments']),added)
            elif tile in TUNNELS:
                p=pen(y,x,TUNNEL_DIRECTIONS[tile])
                _rail(p,[(t,0) for t in range(-6,51,2)])
                group=data['mod_nums'][y][x]
                _tunnel(p,group,s['groups'][group%len(s['groups'])])
            elif tile in ENDINGS:
                p=pen(y,x,END_DIRECTIONS[tile])
                _rail(p,[(t,0) for t in range(-50,21,2)])
                _vehicle(pen(y,x,END_DIRECTIONS[tile],s['engine_scale']),'NUMERAL' if tile in (21,22,32,33) else 'NORMAL',engine=True)
            elif tile==4:
                p.polygon([(-31,24),(-36,-3),(-18,-27),(9,-31),(32,-7),(30,24)],'shadow')
                p.polygon([(-33,17),(-32,-7),(-14,-29),(9,-29),(30,-10),(26,19),(4,29)],'obstacle')
                p.polygon([(-32,-7),(-14,-29),(9,-29),(3,-4),(-12,5)],'obstacle_light')
                p.polygon([(3,-4),(30,-10),(26,19),(4,29)],'obstacle_dark')
                p.line([(-15,6),(3,-4),(11,12)],'obstacle_dark',2)
            elif tile in (34,35,36,37):
                p=pen(y,x,{34:'LEFT',35:'RIGHT',36:'DOWN',37:'UP'}[tile])
                p.rect((-30,-30,33,34),'shadow',radius=5)
                p.rect((-30,-29,29,29),'building','platform_edge',3,1)
                p.polygon([(-31,-31),(21,-31),(32,0),(21,31),(-31,31),(-20,0)],'roof','ink',1)
                p.line([(-20,0),(32,0)],'platform',2)
                p.rect((28,-9,35,9),'window','ink',1)
    base_mods=(solution or level).get('mods',level['mods'])
    cars_at={tuple(car['pos']):car for car in data.get('cars',[])
             if car.get('status') not in ('delivered','finished','removed')}
    covered_buttons=[]
    for y,row in enumerate(data['mods']):
        for x,mod in enumerate(row):
            tile,num=board[y][x],data['mod_nums'][y][x]
            color=s['groups'][num%len(s['groups'])]
            factor=s['mechanism_scale']
            if mod==1:
                if (y,x) in cars_at:
                    covered_buttons.append((_visible_button_pen(pen(y,x),tile,cars_at[y,x],
                                            factor*s['occupied_button_scale']),num,color))
                else:
                    _button(_rail_mount(pen(y,x),tile,factor),num,color)
            elif mod in (3,4):
                _gate(_rail_mount(pen(y,x),tile,factor),num,mod==3,color)
            elif mod in (5,7):
                # Retain the submitted rail drawing; only the lever may use its original frame.
                control_tile=tile if len(PORTS.get(tile,()))==3 else level['board'][y][x]
                _junction_control(pen(y,x),control_tile,num if mod==5 else None,color if mod==5 else 'gate_open')
            elif mod in (6,11) or mod==9 and base_mods[y][x] in (6,11):
                freight=(base_mods[y][x] if mod==9 else mod)==11
                _station(_station_pen(pen(y,x),tile),num,freight,mod==9)
            elif mod==8 or mod==9 and base_mods[y][x] in (0,8):
                _semaphore(_rail_mount(pen(y,x),tile,.9*factor),mod==9)
    has_conflicts=bool(data.get('conflicts'))
    visible=[] if has_conflicts else [car for car in data.get('cars',[]) if car.get('status') not in ('delivered','finished','removed')]
    occupancy=defaultdict(list)
    for car in visible:
        occupancy[tuple(car['pos'])].append(car)
    errors=set()
    for pos in [] if has_conflicts else data.get('error_positions',[]):
        if isinstance(pos,dict):
            pos=pos.get('pos',pos.get('position'))
        if isinstance(pos,(tuple,list)) and len(pos)==2:
            errors.add((max(0,min(h-1,pos[0])),max(0,min(w-1,pos[1]))))
    for (y,x),cars in occupancy.items():
        count=len(cars)
        if count>1:
            errors.add((y,x))
        for i,car in enumerate(cars):
            if count>1:
                nr=math.ceil(math.sqrt(count))
                nc=math.ceil(count/nr)
                xx=(i%nc+.5)/nc-.5
                yy=(i//nc+.5)/nr-.5
                q=pen(y+yy*.9,x+xx*.9,car.get('direction','RIGHT'),min(.72,.9/max(nr,nc)))
            else:
                q=pen(y,x,car.get('direction','RIGHT'),s['car_scale'])
            _vehicle(q,car['type'],_car_label(car))
    for q,num,color in covered_buttons:
        _button(q,num,color)
    for y,x in errors:
        p=pen(y,x)
        p.rect((-47,-47,47,47),None,'error',7,3)
        p.circle(38,-38,8,'error','white',1)
        p.text(38,-38,'!',11,'white')
    if has_conflicts:
        from .conflicts import draw_failure
        draw_failure(im,data,s,board_left,m,c,scale)
    if annotations:
        left,right=m,(width-margin)*scale
        top=(board_height+7)*scale
        d.rounded_rectangle((left,top,right,(height-7)*scale),radius=7*scale,
                            fill=s['panel_background'],outline=s['error'],width=2*scale)
        for index,line in enumerate(annotations):
            size=14
            while size>6:
                font=_font(size*scale,index==0)
                bounds=d.textbbox((0,0),line,font=font)
                if bounds[2]-bounds[0]<=right-left-24*scale:break
                size-=1
            d.text((left+12*scale,top+(7+index*22)*scale),line,
                   font=_font(size*scale,index==0),fill=s['error'] if index==0 else s['ink'])
    if scale>1:
        im=im.resize((round(width),round(height)),getattr(Image,'Resampling',Image).LANCZOS)
    output_path=Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    im.save(output_path,format='PNG')
    return output_path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--recursive',action='store_true')
    p.add_argument('--levels-root',type=Path)
    p.add_argument('--level',type=Path,help='Original level when rendering a state/solution')
    p.add_argument('--style',type=Path)
    p.add_argument('--state-index',type=int,default=-1)
    args=p.parse_args(argv)
    files=sorted(args.input.rglob('*.json') if args.recursive else args.input.glob('*.json')) if args.input.is_dir() else [args.input]
    count=0
    for path in files:
        data=json.loads(path.read_text(encoding='utf-8'))
        state=None
        if 'board' not in data and 'trace' not in data and 'add_tracks' not in data:
            continue
        if 'trace' in data:
            if not data['trace']:
                continue
            state=dict(data['trace'][args.state_index])
            if args.state_index in (-1,len(data['trace'])-1):
                state.setdefault('error_positions',data.get('error_positions',[]))
        original=data
        if args.level:
            original=json.loads(args.level.read_text(encoding='utf-8'))
        elif args.levels_root:
            original=json.loads((args.levels_root/path.parent.name/path.name).read_text(encoding='utf-8'))
        elif state is not None:
            p.error('--level is required to render a validation trace')
        elif 'add_tracks' in data:
            p.error('--level or --levels-root is required to render added tracks')
        out=args.output if args.output and not args.input.is_dir() else (args.output/path.parent.name/path.with_suffix('.png').name if args.output else path.parent/'images'/path.with_suffix('.png').name)
        render(original,out,state=state,solution=data if state is None else None,style=args.style)
        count+=1
    print('Rendered %d image(s)'%count)


if __name__=='__main__':
    main()
