import taichi as ti
import numpy as np

# gauss seidel method for the possion equation
real = ti.f32
ti.init(arch=ti.x64, default_fp=real, enable_profiler=True)

res = [128, 128]
bc = 0 # 0:dirichlet, 1:neumann, 2:periodic

res2 = [2*n for n in res]

x = ti.var(dt=real) # solution
b = ti.var(dt=real) # rhs
r = ti.var(dt=real) # residual
img = ti.var(dt=real, shape=(res2[0], res2[1]))
rtr = ti.var(dt=real, shape=()) # sum residual
x_avg = ti.var(dt=real, shape=()) # sum residual

tol = 1.0e-5 if real==ti.f32 else 1.0e-12
dim = len(res)
indices = ti.ijk if dim == 3 else ti.ij

N_grid = [2 * res[d] // 8 for d in range(dim)]
offset = [- res[d] // 2 for d in range(dim)]
blocks = ti.root.pointer(indices, N_grid)
for f in [x, b, r]:
    blocks.dense(indices, 8).place(f, offset=offset)

@ti.func
def is_interior(I):
    return all(0 <= I < ti.Vector(res))

@ti.func
def is_ghost(I):
    return all(-1 <= I < ti.Vector(res) + 1) and not is_interior(I)

@ti.func
def neighbor_sum(x, I):
    ret = 0.0
    for i in ti.static(range(dim)):
        offset = ti.Vector.unit(dim, i)
        ret += x[I + offset] + x[I - offset]
    return ret

@ti.kernel
def init():
    for I in ti.grouped(ti.ndrange(*res2)):
        I = I - ti.Vector(res)//2
        if is_interior(I):
            x[I] = 0.0
            b[I] = 1.0
            for d in ti.static(range(dim)):
                b[I] *= ti.sin(2.0 * np.pi *  I[d]/res[d])

        if is_ghost(I):
            x[I] = 0.0

@ti.kernel
def compute_rtr():
    rtr[None] = 0.0
    for I in ti.grouped(x):
        if is_interior(I):
            r[I] = b[I] - ((2.0 * dim) * x[I] - neighbor_sum(x, I))

    for I in ti.grouped(x):
        rtr[None] += r[I] * r[I]

@ti.kernel
def subract_x_avg():
    x_avg[None] = 0.0
    for I in ti.grouped(x):
        if is_interior(I):
            x_avg[None] += x[I]

    for I in ti.grouped(x):
        if is_interior(I):
            x_avg[None] = x[I] - (x_avg[None]*res[0]**dim)


@ti.kernel
def smooth(phase: ti.template()):
    # phase = red/black Gauss-Seidel phase
    for I in ti.grouped(x):
        if I.sum() & 1 == phase:
            if is_interior(I):
                x[I] = (b[I] + neighbor_sum(x, I)) / (2.0 * dim)

@ti.kernel
def periodic_bc():
    # apply periodic boundaries one direction at the time
    for d in ti.static(range(dim)):
        for I in ti.grouped(x):
            if I[d] == -1:
                di = ti.Vector.unit(dim, d) * res[d]
                x[I] = x[I + di]
        for I in ti.grouped(x):
            if I[d] == res[d]:
                di = ti.Vector.unit(dim, d) * res[d]
                x[I] = x[I - di]

@ti.kernel
def neumann_bc():
    # apply boundary conditiones to x
    for I in ti.grouped(x):
        if is_ghost(I):
            for d in ti.static(range(dim)):
                di = ti.Vector.unit(dim, d)
                if is_interior(I + di):
                    x[I] = x[I + di]
                if is_interior(I - di):
                    x[I] = x[I - di]

@ti.kernel
def paint():
    for I in ti.grouped(x):
        i = I[0] + res[0]//2
        j = I[1] + res[1]//2
        img[i, j] = x[I]/(res[0] * 2 * np.pi) + .5

gui = ti.GUI("gs", res=(res2[0], res2[1]))
init()
compute_rtr()
print(rtr[None])

n = 0
while rtr[None] > tol:
    smooth(0)
    smooth(1)

    subract_x_avg()

    if bc == 1:
        neumann_bc()
    if bc == 2:
        periodic_bc()

    if n % 100 == 0:
        compute_rtr()
        paint()
        gui.set_image(img)
        gui.show()
        print(rtr[None])
        print(n)
    n += 1
