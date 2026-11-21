from __future__ import annotations
import math, pathlib
import numpy as np
from .planning import dist, within_collision

COLOR_Y={'red':-.18,'blue':0.,'yellow':.18}
GOAL_Y={'cyan':-.18,'magenta':0.,'green':.18}
SCENARIOS=('clear','blocked','dynamic','clutter')


def _clip(v,lo,hi): return float(max(lo,min(hi,v)))


def generate_task(scene_id:int, seed_base:int):
    if scene_id<1: raise ValueError('scene_id must be >= 1')
    scenario=SCENARIOS[(scene_id-1)%len(SCENARIOS)]
    seed=int(seed_base)+int(scene_id)
    rng=np.random.default_rng(seed)
    object_color=str(rng.choice(list(COLOR_Y)))
    goal_color=str(rng.choice(list(GOAL_Y)))
    object_poses={c:(float(-.285+rng.uniform(-.014,.014)),float(y+rng.uniform(-.015,.015))) for c,y in COLOR_Y.items()}
    jx=lambda s: float(rng.uniform(-s,s))
    if scenario=='clear':
        a=(-.06,.255,.047); b=(.09,-.255,.047)
    elif scenario=='blocked':
        a=(-.055,.055,.047); b=(.075,-.075,.047)
    elif scenario=='dynamic':
        a=(-.07,.16,.047); b=(.08,-.17,.047)
    else:
        a=(-.055,.025,.047); b=(.085,.115,.047)
    static_a=(_clip(a[0]+jx(.018),-.17,.16),_clip(a[1]+jx(.018),-.26,.26),a[2])
    static_b=(_clip(b[0]+jx(.018),-.17,.17),_clip(b[1]+jx(.018),-.26,.26),b[2])
    moving=(.14,.265,.051)
    start=object_poses[object_color]; goal=(.29,GOAL_Y[goal_color])
    # identical future obstacle position for every compared method in this scene
    t=.62
    dyn=(_clip(start[0]*(1-t)+goal[0]*t+jx(.008),-.16,.22),
         _clip(start[1]*(1-t)+goal[1]*t+jx(.008),-.24,.24),.051)
    return dict(scene_id=scene_id,scenario=scenario,seed=seed,object_color=object_color,
                goal_color=goal_color,object_poses=object_poses,static_a=static_a,
                static_b=static_b,moving=moving,moving_dynamic=dyn,
                instruction=f'Place the {object_color} cube in the {goal_color} target zone.')


class Tabletop:
    """MuJoCo Cartesian gantry with idealized suction; not a dexterous grasp benchmark."""
    def __init__(self, task, xml_path, render_required=False):
        import mujoco
        self.mj=mujoco; self.task=dict(task)
        self.model=mujoco.MjModel.from_xml_path(str(pathlib.Path(xml_path).resolve()))
        self.data=mujoco.MjData(self.model)
        self.addr_xyz=[int(self.model.joint(f'axis_{a}').qposadr[0]) for a in 'xyz']
        self.dof_xyz=[int(self.model.joint(f'axis_{a}').dofadr[0]) for a in 'xyz']
        self.cube_addr={c:int(self.model.joint(f'free_{c}').qposadr[0]) for c in COLOR_Y}
        self.moving_id=int(self.model.body('moving_obstacle').mocapid[0])
        self.renderer=None; self.render_error=''; self.robot_xy=(0.,-.26); self.held=None
        self.visited_obstacles=set(); self.violations=0; self.min_clearance=float('inf')
        self.sim_motion_distance=0.; self.steps=0; self.tool_tip_z=.15
        self._reset()
        try:
            self.renderer=mujoco.Renderer(self.model,height=600,width=800)
        except Exception as exc:
            self.render_error=f'{type(exc).__name__}: {exc}'
            if render_required:
                raise RuntimeError('MuJoCo RGB rendering is required for live Gemini input. '+self.render_error) from exc

    def _reset(self):
        self.mj.mj_resetData(self.model,self.data)
        for c,(x,y) in self.task['object_poses'].items():
            adr=self.cube_addr[c]; self.data.qpos[adr:adr+7]=[x,y,.033,1,0,0,0]
        for name,p in (('static_a',self.task['static_a']),('static_b',self.task['static_b'])):
            self.model.body_pos[int(self.model.body(name).id),:]=[p[0],p[1],.072]
        self.set_robot(self.robot_xy,.15,advance=False); self.move_obstacle(self.task['moving'])
        self.mj.mj_forward(self.model,self.data)
        for _ in range(10): self.mj.mj_step(self.model,self.data)

    def move_obstacle(self, obs):
        self.task['moving']=tuple(obs)
        self.data.mocap_pos[self.moving_id,:]=[obs[0],obs[1],.072]
        self.data.mocap_quat[self.moving_id,:]=[1.,0.,0.,0.]
        self.mj.mj_forward(self.model,self.data)

    @property
    def obstacles(self): return [self.task['static_a'],self.task['static_b'],self.task['moving']]

    @property
    def cube_position(self):
        return {c:tuple(float(v) for v in self.data.qpos[adr:adr+3]) for c,adr in self.cube_addr.items()}

    def set_robot(self,xy,z,advance=True):
        self.robot_xy=tuple(float(v) for v in xy); self.tool_tip_z=float(z)
        joints=(self.robot_xy[0],self.robot_xy[1],self.tool_tip_z-.305)
        for ctrl_i,(adr,dof,value) in enumerate(zip(self.addr_xyz,self.dof_xyz,joints)):
            self.data.qpos[adr]=value; self.data.qvel[dof]=0.; self.data.ctrl[ctrl_i]=value
        if self.held is not None:
            adr=self.cube_addr[self.held]
            self.data.qpos[adr:adr+7]=[self.robot_xy[0],self.robot_xy[1],max(.033,self.tool_tip_z-.035),1,0,0,0]
            dof=int(self.model.joint(f'free_{self.held}').dofadr[0]); self.data.qvel[dof:dof+6]=0.
        self.mj.mj_forward(self.model,self.data)
        if advance: self.mj.mj_step(self.model,self.data); self.steps+=1
        if self.held is not None:
            hits=within_collision(self.robot_xy,self.obstacles)
            for h in hits:
                if h not in self.visited_obstacles: self.visited_obstacles.add(h); self.violations+=1
            for x,y,r in self.obstacles:
                self.min_clearance=min(self.min_clearance,dist(self.robot_xy,(x,y))-(r+.041))

    def travel(self,path,z=.15):
        if not path:return
        for goal in path[1:]:
            cur=self.robot_xy; d=dist(cur,goal); n=max(1,math.ceil(d/.012))
            for j in range(1,n+1):
                pt=(cur[0]+(goal[0]-cur[0])*j/n,cur[1]+(goal[1]-cur[1])*j/n)
                self.set_robot(pt,z)
            self.sim_motion_distance+=d

    def pick(self,color):
        pos=self.cube_position[color]; self.travel([self.robot_xy,(pos[0],pos[1])],z=.15)
        for z in (.12,.09,.066): self.set_robot(self.robot_xy,z)
        self.held=color; self.set_robot(self.robot_xy,.11); self.set_robot(self.robot_xy,.15)

    def place(self,goal):
        self.set_robot(self.robot_xy,.08); self.set_robot(self.robot_xy,.065); self.held=None
        for z in (.09,.15): self.set_robot(self.robot_xy,z)
        for _ in range(30): self.mj.mj_step(self.model,self.data); self.steps+=1

    def target_point(self,goal_color): return (.29,GOAL_Y[goal_color])

    def success(self):
        p=self.cube_position[self.task['object_color']]
        return dist(p,self.target_point(self.task['goal_color']))<=.064

    def snapshot(self,filename=None):
        if self.renderer is None: raise RuntimeError('Renderer unavailable: '+self.render_error)
        from PIL import Image
        self.mj.mj_forward(self.model,self.data); self.renderer.update_scene(self.data,camera='overhead')
        im=Image.fromarray(self.renderer.render().copy())
        if filename is not None:
            p=pathlib.Path(filename); p.parent.mkdir(parents=True,exist_ok=True); im.save(p)
        return im

    def close(self):
        if self.renderer is not None:self.renderer.close()
