import mujoco as mj
import mujoco.viewer as viewer
import numpy as np

model = mj.MjModel.from_xml_path("mjmodel.xml")
data = mj.MjData(model)

def save_model_high_precision(model, filename):
    with open(filename, "w") as f:
        # 1. 模型名称
        model_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_MODEL, 0)
        f.write(f"Model Name: {model_name if model_name else 'Unnamed'}\n")
        f.write("=" * 50 + "\n\n")
        
        # 2. 遍历 Body（包含质量、质心位置、惯量）
        f.write("BODY INFORMATION (Mass & Inertia)\n")
        f.write("-" * 50 + "\n")
        for i in range(model.nbody):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
            name = name if name else f"body_{i}"
            
            mass = model.body_mass[i]
            inertia = model.body_inertia[i]  # 这是一个长度为3的向量 [Ixx, Iyy, Izz]
            ipos = model.body_ipos[i]        # 质心相对于该 body 原点的偏移
            
            f.write(f"Body {i} [{name}]:\n")
            f.write(f"  Mass:    {mass:.12f}\n")
            f.write(f"  Inertia: {np.array2string(inertia, precision=12, separator=', ')}\n")
            f.write(f"  CoM Pos: {np.array2string(ipos, precision=12, separator=', ')}\n")
            f.write("\n")

        f.write("\n")

        # 3. 遍历 Geom（几何体参数）
        f.write("GEOM INFORMATION (Shape & Position)\n")
        f.write("-" * 50 + "\n")
        for i in range(model.ngeom):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
            name = name if name else f"geom_{i}"
            
            f.write(f"Geom {i} [{name}]:\n")
            f.write(f"  Size: {np.array2string(model.geom_size[i], precision=12, separator=', ')}\n")
            f.write(f"  Pos:  {np.array2string(model.geom_pos[i], precision=12, separator=', ')}\n")
            f.write("\n")

save_model_high_precision(model, "high_res_model.txt")


# def read_mass(body_name):
#     for i in range(model.nbody):
#         if body_name == mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i):
#             mass = model.body_mass[i]
#             print(f"体 {i} ({body_name}): {mass:.6f} kg")
#     return mass

# def read_inertia(body_name):
#     for i in range(model.nbody):
#         if body_name == mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i):
#             inertia = model.body_inertia[i]
#             print(f"体 {i} ({body_name}): {inertia[0]:.6f} {inertia[1]:.6f} {inertia[2]:.6f} ")
#     return inertia

# def read_size(geom_name):
#     for i in range(model.ngeom):
#         if geom_name == mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i):
#             size = model.geom_size[i]
#             print(f"体 {i} ({geom_name}): {size[0]:.6f} {size[1]:.6f} {size[2]:.6f} ")
#     return size

# class User_Body:
#     def __init__(self, body_name, mass, inertia, longth=0):
#         self.body_name = body_name
#         self.mass = mass
#         self.inertia = inertia
#         self.longth = longth

#     def read (self):
#         self.mass = read_mass(self.body_name)
#         self.inertia = read_inertia(self.body_name)



# M = User_Body("gimbal", 0, [0,0,0])
# ob1 = User_Body("object1", 0, [0,0,0])
# ob2 = User_Body("object2", 0, [0,0,0])
# ob3 = User_Body("object3", 0, [0,0,0])
# ob4 = User_Body("object4", 0, [0,0,0])
# m_bl1 = User_Body("bigleg1", 0, [0,0,0],)
# m_bl2 = User_Body("bigleg2", 0, [0,0,0])
# m_bl3 = User_Body("bigleg3", 0, [0,0,0])
# m_bl4 = User_Body("bigleg4", 0, [0,0,0])
# m_sl1 = User_Body("smallleg1", 0, [0,0,0])
# m_sl2 = User_Body("smallleg2", 0, [0,0,0])
# m_sl3 = User_Body("smallleg3", 0, [0,0,0])
# m_sl4 = User_Body("smallleg4", 0, [0,0,0])
# m_w1 = User_Body("wheel1", 0, [0,0,0])
# m_w2 = User_Body("wheel2", 0, [0,0,0])
# m_w3 = User_Body("wheel3", 0, [0,0,0])
# m_w4 = User_Body("wheel4", 0, [0,0,0])
# M.read()
# ob1.read()
# ob2.read()
# ob3.read()
# ob4.read()
# m_bl1.read()
# m_bl2.read()
# m_bl3.read()
# m_bl4.read()
# m_sl1.read()
# m_sl2.read()
# m_sl3.read()
# m_sl4.read()
# m_w1.read()
# m_w2.read()
# m_w3.read()
# m_w4.read()
# m_wl = User_Body("wheel_weld_l", m_w1.mass+m_w3.mass, [m_w1.inertia[0]+m_w3.inertia[0], m_w1.inertia[1]+m_w3.inertia[1], m_w1.inertia[2]+m_w3.inertia[2]])
# m_wr = User_Body("wheel_weld_r", m_w2.mass+m_w4.mass, [m_w2.inertia[0]+m_w4.inertia[0], m_w2.inertia[1]+m_w4.inertia[1], m_w2.inertia[2]+m_w4.inertia[2]])

viewer.launch(model, data)