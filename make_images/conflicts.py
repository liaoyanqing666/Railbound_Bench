"""Motion trails, contact poses and cause-specific failure markers."""
import math
from collections import defaultdict

from PIL import Image, ImageDraw, ImageColor

from validator.tiles import DIRECTIONS


def draw_failure(image, data, style, left, top, cell, scale):
    from .render import _Pen, _vehicle, _car_label, _station, _station_pen

    layer = image.convert('RGBA')
    draw = ImageDraw.Draw(layer)
    height, width = len(data['board']), len(data['board'][0])
    events = data['conflicts']
    previous = {car['id']: car for car in data.get('previous_cars', [])}
    motion = {item['id']: item for item in data.get('motion', [])}
    current = {car.get('id', index): car for index, car in enumerate(data['cars'])}

    def point(pos):
        return left + (pos[1] + .5) * cell, top + (pos[0] + .5) * cell

    def bounded(pos):
        return [max(-.48, min(height - .52, pos[0])), max(-.48, min(width - .52, pos[1]))]

    def rgba(color, alpha=255):
        if color in style:
            color = style[color]
        rgb = ImageColor.getrgb(color) if isinstance(color, str) else tuple(color)
        return rgb[:3] + (alpha,)

    def curve(start, end, color='motion_color', dashed=False, bend=.35, arrow=True, lane=False):
        a, b = point(start), point(end)
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length < cell * .08:
            return
        if lane:
            shift = .35 * cell * (1 if bend >= 0 else -1)
            offset = (0, shift) if abs(dx) >= abs(dy) else (shift, 0)
            original_a, original_b = a, b
            a = (a[0] + offset[0], a[1] + offset[1])
            b = (b[0] + offset[0], b[1] + offset[1])
            draw.line((original_a, a), fill=rgba('ghost_ink', 135), width=scale)
            draw.line((b, original_b), fill=rgba(color, 170), width=scale)
            bend = 0
        nx, ny = -dy / length, dx / length
        control = ((a[0] + b[0]) / 2 + nx * bend * cell,
                   (a[1] + b[1]) / 2 + ny * bend * cell)
        points = [((1 - t) ** 2 * a[0] + 2 * (1 - t) * t * control[0] + t * t * b[0],
                   (1 - t) ** 2 * a[1] + 2 * (1 - t) * t * control[1] + t * t * b[1])
                  for t in (i / 40 for i in range(41))]
        strokes = [points[i:i + 3] for i in range(0, 38, 5)] if dashed else [points]
        for stroke in strokes:
            draw.line(stroke, fill=rgba('white'), width=max(1, round((style['motion_width'] + 3) * scale)), joint='curve')
            draw.line(stroke, fill=rgba(color), width=max(1, round(style['motion_width'] * scale)), joint='curve')
        if arrow:
            vx, vy = points[-1][0] - points[-4][0], points[-1][1] - points[-4][1]
            norm = math.hypot(vx, vy) or 1
            ux, uy = vx / norm, vy / norm
            px, py = points[-1]
            size = 11 * scale
            draw.polygon([(px, py), (px - ux * size - uy * size * .5, py - uy * size + ux * size * .5),
                          (px - ux * size + uy * size * .5, py - uy * size - ux * size * .5)], fill=rgba(color))

    def sprite(car, pos, factor=1, tilt=0, ghost=False):
        size = math.ceil(cell * 1.5)
        sprite = Image.new('RGBA', (size, size))
        p = _Pen(ImageDraw.Draw(sprite), style, size / 2, size / 2, cell / 100 * factor, car['direction'])
        _vehicle(p, car['type'], _car_label(car))
        if tilt:
            sprite = sprite.rotate(tilt, resample=Image.Resampling.BICUBIC)
        if ghost:
            wash = Image.new('RGBA', sprite.size, rgba('background'))
            faded = Image.blend(sprite, wash, .38)
            faded.putalpha(sprite.getchannel('A').point(lambda value: round(value * style['ghost_opacity'])))
            sprite = faded
        x, y = point(pos)
        layer.alpha_composite(sprite, (round(x - size / 2), round(y - size / 2)))

    def focus(pos, semantic=False):
        x, y = point(pos)
        r = cell * .46
        halo = Image.new('RGBA', layer.size)
        hd = ImageDraw.Draw(halo)
        hd.rounded_rectangle((x - r, y - r, x + r, y + r), radius=12 * scale,
                             fill=rgba('error', 24 if semantic else 15),
                             outline=rgba('error', 140), width=2 * scale)
        layer.alpha_composite(halo)
        span = cell * .16
        for xx, yy, sx, sy in ((x-r,y-r,1,1),(x+r,y-r,-1,1),(x-r,y+r,1,-1),(x+r,y+r,-1,-1)):
            draw.line([(xx,yy+sy*span),(xx,yy),(xx+sx*span,yy)], fill=rgba('error'), width=3 * scale)

    def burst(pos, small=False):
        x, y = point(pos)
        radius = style['impact_size'] * scale * (.8 if small else 1)
        for inner, outer in ((1.15, 1.55), (1.75, 1.95)):
            for angle in (0, .8, 1.8, 2.8, 3.8, 4.8, 5.7):
                draw.line([(x+math.cos(angle)*radius*inner,y+math.sin(angle)*radius*inner),
                           (x+math.cos(angle)*radius*outer,y+math.sin(angle)*radius*outer)],
                          fill=rgba('error', 210), width=2 * scale)
        points = [(x + math.cos(i*math.pi/8) * radius * (1 if i%2==0 else .62),
                   y + math.sin(i*math.pi/8) * radius * (1 if i%2==0 else .62)) for i in range(16)]
        draw.polygon(points, fill=rgba('white'))
        draw.line(points + points[:1], fill=rgba('ink'), width=2 * scale, joint='curve')
        inner = [(x+math.cos(i*math.pi/5)*radius*(.55 if i%2==0 else .22),
                  y+math.sin(i*math.pi/5)*radius*(.55 if i%2==0 else .22)) for i in range(10)]
        draw.polygon(inner, fill=rgba('impact_core'))

    physical = [event for event in events if event['kind'] in ('collision', 'tunnel_collision')]

    def contact(event):
        pos = list(event['impact'])
        if event['geometry'] == 'rear_end':
            dy, dx = DIRECTIONS[motion[event['leader_id']]['direction']]
            pos = [pos[0] - dy*.34, pos[1] - dx*.34]
        return pos

    poses, pose_distances = {}, {}
    for event in physical:
        if event['geometry'] == 'tunnel_opposite':
            for index in event['car_ids']:
                item = motion[index]
                pos = item['entrance'] or item['from']
                dy, dx = DIRECTIONS[item['direction']]
                poses[index] = ([pos[0] - dy*.34, pos[1] - dx*.34], item['direction'], (-1)**index * 8)
            continue
        for index in event['car_ids']:
            item = motion[index]
            dy, dx = DIRECTIONS[item['direction']]
            offset = .34 if event.get('leader_id') == index else -.34
            impact = contact(event)
            pos = [impact[0] + dy*offset, impact[1] + dx*offset]
            distance = math.dist(event['impact'], item['from'])
            if index not in pose_distances or distance < pose_distances[index]:
                poses[index] = (pos, item['direction'], (-1)**index * style['motion_tilt'])
                pose_distances[index] = distance
    semantic = {index for event in events if event['kind'] not in ('collision', 'tunnel_collision', 'derailment')
                for index in event['car_ids']}
    stationary_failures = set()

    def front_impact(index):
        item = motion[index]
        dy, dx = DIRECTIONS[item['direction']]
        return [item['from'][0] + dy*.47, item['from'][1] + dx*.47]

    for event in events:
        if event['kind'] == 'derailment':
            index = event['car_ids'][0]
            item = motion[index]
            stationary_failures.add(index)
            poses.setdefault(index, (item['from'], item['direction'], 0))

    focus_positions = set()
    for event in events:
        if event['kind'] == 'derailment':
            focus_positions.add(tuple(motion[event['car_ids'][0]]['from']))
        else:
            focus_positions.update(map(tuple,event['positions']))
    for pos in sorted(focus_positions):
        focus(pos, any(list(pos) in event.get('stations', []) for event in events))

    if style['motion_trails']:
        for index, item in motion.items():
            car = previous[index]
            destination = poses.get(index, (current[index]['pos'],))[0]
            if math.dist(destination, item['from']) > .08 and car['status'] != 'crashed':
                sprite(car, item['from'], .93, ghost=True)
        for index, item in motion.items():
            if item['waiting_reason'] == 'wreck':
                continue
            if current[index]['type']=='DECOY' and current[index]['status']=='crashed' and item['to']==item['from']:
                continue
            event = next((event for event in physical if index in event['car_ids']), None)
            if index in stationary_failures:
                target = front_impact(index)
            elif event and event['geometry'] == 'tunnel_opposite':
                target = item['entrance']
            elif event and event['geometry'] == 'edge_swap':
                target = event['impact']
            elif event and event['geometry'] == 'rear_end' and index != event['leader_id']:
                target = contact(event)
            elif event and item['entrance']:
                curve(item['from'], item['entrance'], bend=.2)
                curve(item['entrance'], item['to'], color='freight', dashed=True, bend=.1)
                continue
            else:
                target = item['to'] if item['to'] != item['from'] else item['attempted']
            if item['waiting_reason'] in ('gate', 'semaphore', 'passenger_stop', 'post_office') and item['to'] == item['from']:
                continue
            curve(item['from'], bounded(target), dashed=index in semantic,
                  bend=(-1 if index%2==0 else 1), lane=True)

    for index, car in current.items():
        if car['status'] == 'delivered':
            continue
        if index in poses:
            pos, direction, tilt = poses[index]
            display = dict(car, direction=direction)
            sprite(display, pos, style['car_scale'] if index in stationary_failures else style['collision_car_scale'], tilt)
        else:
            sprite(car, car['pos'], .94 if index in semantic else 1)

    grouped = defaultdict(list)
    for event in physical:
        key = (event['geometry'], tuple(round(value, 3) for value in contact(event)))
        grouped[key].append(event)
    for (geometry, impact), group in grouped.items():
        if geometry == 'tunnel_opposite':
            positions = group[0]['positions']
            curve(positions[0], positions[1], color='freight', dashed=True, bend=.12, arrow=False)
            for pos in positions:
                burst(pos, small=True)
        else:
            burst(impact)

    for event in events:
        if event['kind'] in ('collision', 'tunnel_collision'):
            continue
        index = event['car_ids'][0]
        car = current[index]
        if event['kind'] == 'unvisited_station':
            for pos in event['stations']:
                row, col = pos
                x,y = point(pos)
                highlighted = dict(style, passenger=style['error'], freight=style['error'])
                p = _station_pen(_Pen(draw,highlighted,x,y,cell/100),data['board'][row][col])
                _station(p,car['num'],car['type']=='NUMERAL')
                p.circle(1,0,6,'error','white',1)
                p.text(1,0,'!',9,'white')
            continue
        pos = bounded(event['impact'])
        if event['kind'] == 'derailment':
            burst(front_impact(index), small=True)
        if event['kind'] != 'derailment':
            x, y = point([pos[0], pos[1] - DIRECTIONS[motion[index]['direction']][1]*.36])
            r = 10*scale
            draw.ellipse((x-r,y-r,x+r,y+r), fill=rgba('white'), outline=rgba('error'), width=2*scale)
            draw.line((x-r*.65,y-r*.65,x+r*.65,y+r*.65), fill=rgba('error'), width=3*scale)
    image.paste(layer.convert('RGB'))
