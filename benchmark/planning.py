from __future__ import annotations
import heapq, math
import numpy as np

XMIN, XMAX, YMIN, YMAX = -.39, .39, -.29, .29
STEP = .02
PAYLOAD_EXTRA = .041


def dist(p, q):
    return math.hypot(p[0]-q[0], p[1]-q[1])


def clearance(p, obstacles):
    return min((dist(p, (x, y)) - (r + PAYLOAD_EXTRA) for x, y, r in obstacles), default=float('inf'))


def within_collision(p, obstacles):
    return [i for i, (x, y, r) in enumerate(obstacles) if dist(p, (x, y)) < r + PAYLOAD_EXTRA]


def line_is_free(a, b, obstacles, interval=.008):
    n=max(1, math.ceil(dist(a,b)/interval))
    for i in range(n+1):
        t=i/n
        p=(a[0]*(1-t)+b[0]*t, a[1]*(1-t)+b[1]*t)
        if clearance(p, obstacles) < 0:
            return False
    return True


def direct_route(start, goal, hint='direct'):
    start,goal=tuple(start),tuple(goal)
    if hint=='upper': return [start,(0.0,.255),goal]
    if hint=='lower': return [start,(0.0,-.255),goal]
    return [start,goal]


def safe_route(start, goal, obstacles, hint='direct'):
    start,goal=tuple(start),tuple(goal)
    if line_is_free(start,goal,obstacles): return [start,goal]
    xs=np.arange(XMIN,XMAX+STEP*.5,STEP); ys=np.arange(YMIN,YMAX+STEP*.5,STEP)
    def idx(p):
        return (max(0,min(len(xs)-1,round((p[0]-XMIN)/STEP))),
                max(0,min(len(ys)-1,round((p[1]-YMIN)/STEP))))
    def coord(q): return (float(xs[q[0]]),float(ys[q[1]]))
    src,dst=idx(start),idx(goal)
    def valid(q):
        return 0<=q[0]<len(xs) and 0<=q[1]<len(ys) and (q in (src,dst) or clearance(coord(q),obstacles)>=0)
    parent={src:None}; best={src:0.}; frontier=[(dist(start,goal),0.,src)]; settled=set()
    neigh=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while frontier:
        _,g,p=heapq.heappop(frontier)
        if p in settled: continue
        settled.add(p)
        if p==dst: break
        for dx,dy in neigh:
            q=(p[0]+dx,p[1]+dy)
            if not valid(q): continue
            step=STEP*(math.sqrt(2) if dx and dy else 1)
            # tiny preference only breaks ties; safety remains geometry-based
            y=coord(q)[1]
            bias=0.0002*(max(0,-y) if hint=='upper' else max(0,y) if hint=='lower' else 0)
            ng=g+step+bias
            if ng < best.get(q,float('inf')):
                best[q]=ng; parent[q]=p
                heapq.heappush(frontier,(ng+dist(coord(q),goal),ng,q))
    if dst not in parent: return []
    cells=[]; p=dst
    while p is not None: cells.append(p); p=parent[p]
    cells.reverse(); pts=[start]+[coord(q) for q in cells[1:-1]]+[goal]
    # greedy line-of-sight simplification
    simp=[pts[0]]; i=0
    while i < len(pts)-1:
        j=len(pts)-1
        while j>i+1 and not line_is_free(pts[i],pts[j],obstacles): j-=1
        simp.append(pts[j]); i=j
    return simp


def length(path):
    return sum(dist(path[i],path[i+1]) for i in range(len(path)-1)) if path else 0.0


def point_at_fraction(path, frac):
    if not path: return (0.,0.)
    total=length(path)
    if total<=0:return tuple(path[-1])
    target=max(0,min(1,float(frac)))*total; acc=0.
    for a,b in zip(path[:-1],path[1:]):
        d=dist(a,b)
        if acc+d>=target:
            t=(target-acc)/max(d,1e-12)
            return (a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t)
        acc+=d
    return tuple(path[-1])


def split_path(path, frac):
    """Split a polyline at arclength fraction, preserving both pieces."""
    if len(path)<2:return list(path),list(path)
    total=length(path); target=max(0,min(1,float(frac)))*total; acc=0.
    pre=[path[0]]
    for i,(a,b) in enumerate(zip(path[:-1],path[1:])):
        d=dist(a,b)
        if acc+d >= target:
            t=(target-acc)/max(d,1e-12)
            cut=(a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t)
            if dist(pre[-1],cut)>1e-10: pre.append(cut)
            rest=[cut]
            if dist(cut,b)>1e-10: rest.append(b)
            rest.extend(path[i+2:])
            return pre,rest
        pre.append(b); acc+=d
    return list(path),[path[-1]]


def path_is_free(path, obstacles):
    return all(line_is_free(a,b,obstacles) for a,b in zip(path[:-1],path[1:]))
