import taichi as ti
import matplotlib.cm as cm

# A simple app showcasing taichi sparse programming applied
# to solving the Eikanol equation
# ref: "A Fast Iterative Method for Eikonal Equations"
# ref: "Fast Methods for Eikonal Equations: an Experimental Survey"

ti.init(arch=ti.x64, default_fp=ti.f64, enable_profiler=True)

# inputs
source_pts = [[.1,.05],[.9,.9]] # list of tuples of source points , 1x1 domain
obstacles = 1 # 0: no obstacles, 1:walls
N = 1024 # number of grid points
N_gui = 700 # gui resolution
eps = 1.0e-10 # convergence criteria
img_scaler = .75 # scale image brightness
dont_end = True


active = ti.var(dt=ti.i32)# the active cell mask
active_new = ti.var(dt=ti.i32)
phi = ti.var(dt=ti.f64) # level set solution
speed = ti.var(dt=ti.f64) # propagation speed map
img = ti.var(dt=ti.f32, shape=(N_gui,N_gui))
empty = ti.var(dt=ti.i32, shape=())

b_size = 8

blocks = ti.root.dense(ti.ij, N//b_size**3).dense(ti.ij, b_size).dense(ti.ij, b_size)
blocks.dense(ti.ij, 8).place(phi)
blocks.dense(ti.ij, 8).place(speed)

blocks = ti.root.pointer(ti.ij, N//b_size**3).pointer(ti.ij, b_size).pointer(ti.ij, b_size)
blocks.dense(ti.ij, b_size).place(active)

blocks_new = ti.root.pointer(ti.ij, N//b_size**3).pointer(ti.ij, b_size).pointer(ti.ij, b_size)
blocks_new.dense(ti.ij, b_size).place(active_new)

BIG = 1.0e8
h = 1.0/N

def clear_active():
    blocks.deactivate_all()

def clear_active_new():
    blocks_new.deactivate_all()

@ti.func
def is_interior(i,j):
    return i > 0 and i < N-1 and j > 0  and j < N-1

@ti.kernel
def init_phi():
    # intialize domain to a large value
    for i, j in phi:
        phi[i,j] = BIG

@ti.kernel
def init_speed():
    for i,j in speed:
        if obstacles == 1:
            speed[i,j] = get_speed_walls(i,j)
        else:
            speed[i,j] = 1.0

@ti.func
def get_speed_walls(i,j):
    x = i*h
    y = j*h
    s = 1.0
    if x > 0 and x < .5 and y > .1 and y < .125:
        s = 0.0
    if x > .3 and x < 1.0 and y > .5 and y < .525:
        s = 0.0
    if x > .2 and x < .225 and y > .55 and y < 1.0:
        s = 0.0
    if x > .3 and x < 1.0 and y > .2 and y < .225:
        s = 0.0
    if x > .7 and x < .725 and y > .8 and y < 1.0:
        s = 0.0
    if x > .25 and x < .5 and y > .3 and y < .4:
        s = 0.0
    return s

@ti.kernel
def init_active():
    # intialize active phi band around source points
    for i, j in phi:
        if is_interior(i, j) and phi[i, j] == 0.0:
            empty[None] = 0
            for di,dj in ti.static([[-1,0],[1,0],[0,-1],[0,1]]):
                active[i+di, j+dj] = 1

@ti.kernel
def update():
    # update the level set at active points by solving eikanol equation
    # also check for convergence and build a new set of active points
    for i,j in active:
        if is_interior(i,j) and active[i,j] == 1:
            t = solve_eikanol(i,j)
            # check the cell for convergence
            if abs(t-phi[i,j]) < eps:
                phi[i,j] = t

                # update neighboring inactive points and see if they improve
                # if they do, add them to the new active set
                for di,dj in ti.static([[-1,0],[1,0],[0,-1],[0,1]]):
                    if is_interior(i+di,j+dj) and active[i+di,j+dj] == 0:
                        t = solve_eikanol(i+di,j+dj)
                        if t < phi[i+di,j+dj]:
                            phi[i+di,j+dj] = t
                            active_new[i+di, j+dj] = 1
                            empty[None] = 0
            else:
                phi[i,j] = t
                active_new[i,j] = 1
                empty[None] = 0

@ti.func
def solve_eikanol(i,j):
    t = BIG

    if speed[i,j] > 0.0:
        tx = min(phi[i+1,j], phi[i-1,j])
        ty = min(phi[i,j+1], phi[i,j-1])
        tmin = min(tx,ty)
        tmax = max(tx,ty)

        # 1-D update
        t = tmin + h/speed[i,j]

        if tmax != BIG:
            # 2-d update by solving quadratic equation
            a = 2.0
            b = -2.0 * (tx + ty)
            c = (tx**2 + ty**2) - h**2 / speed[i,j]**2
            q = b**2 - 4.0 * a * c

            t2 = 0.0
            if q > 0.0:
                t2 = (- b + ti.sqrt(q)) / (2.0 * a)

            if t2 > tmax:
                t = t2

    return t

@ti.kernel
def copy_active_new_to_active():
    for i,j in active_new:
        active[i,j] = active_new[i,j]

@ti.kernel
def zero_img():
    for i,j in img:
        img[i,j] = 0.0

@ti.kernel
def paint_phi():
    for i,j in img:
        img[i,j] = phi[i*N//N_gui,j*N//N_gui]

@ti.kernel
def paint_active():
    for i,j in active:
        if active[i,j] == 1:
            img[i*N_gui//N,j*N_gui//N] = float(active[i,j])

@ti.kernel
def set_boundary():
    for i,j in img:
        if not is_interior(i,j):
            img[i,j] = 0.0



gui_phi = ti.GUI('Level Set', (N_gui, N_gui))
gui_active = ti.GUI('Active Band', (N_gui, N_gui))

init_phi()
for x,y in source_pts:
    i = int(x*N)
    j = int(y*N)
    phi[i,j] = 0.0
init_speed()
init_active()

# iterate until active is empty
n = 0
while not empty[None] or dont_end:
    empty[None] = 1
    update()
    clear_active()
    copy_active_new_to_active()
    clear_active_new()

    if n%50 == 0:
        zero_img()
        paint_active()
        gui_active.set_image(img.to_numpy())
        gui_active.show()

        paint_phi()
        gui_phi.set_image(cm.prism(1.0-img_scaler*img.to_numpy()))
        gui_phi.show()

    n += 1
ti.profiler_print()
