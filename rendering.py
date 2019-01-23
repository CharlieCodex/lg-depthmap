import numpy as np
import moderngl
from PyQt5 import QtOpenGL, QtWidgets, QtCore
import time

from PIL import Image

class QGLControllerWidget(QtOpenGL.QGLWidget):
    def __init__(self):
        fmt = QtOpenGL.QGLFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QtOpenGL.QGLFormat.CoreProfile)
        super().__init__(fmt, None)
        self.mouse = (0,0)
        self.mouseDrag = False
        self.eye = (20,0,20)
        self.ctx = None
        # nx, ny = (1000,100)
        # self.x, self.y = np.mgrid[-5:5:(nx*1j), -5:5:(ny*1j)].astype(np.float32)

    def update_data(self, im, depth):
        print(im.shape[:2], im.dtype)
        self.img_tex = self.ctx.texture(
            im.shape[:2], 3, im.astype('uint8').tobytes())
        self.mesh_samp = self.ctx.depth_texture(
            depth.shape[:2], depth.astype('float32').tobytes())
        self.img_tex.build_mipmaps()
        self.mesh_samp.build_mipmaps()
        self.img_tex.use(0)
        self.mesh_samp.use(1)

    def initializeGL(self):
        self.ctx = moderngl.create_context()
        with open('vert.glsl', 'r') as f:
            vert_shader = f.read()
        with open('frag.glsl', 'r') as f:
            frag_shader = f.read()
        prog = self.ctx.program(
            vertex_shader=vert_shader,
            fragment_shader=frag_shader
        )
        pts = np.array(((0.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
        print(pts.shape)
        print(pts.min(), pts.mean(), pts.max())
        buf_size = pts.nbytes
        self.vbo = self.ctx.buffer(reserve=buf_size)
        self.vbo.write(pts.tobytes())
        self.vao = self.ctx.simple_vertex_array(prog,
                                                self.vbo,
                                                'vert')
        self.update_data(np.zeros((400,400,3)), np.zeros((400,400)))
        self.update_window()
        self.clear_gray = 0.001
        self.clear = False

    def rotate(self, dx, dy):
        dt = dx / 100
        dp = -dy / 100
        r_ = np.sqrt(self.eye[0]**2+self.eye[1]**2)
        r = np.sqrt(r_**2+self.eye[2]**2)
        theta = np.arctan2(self.eye[1],self.eye[0]) + dt
        psi = np.arctan2(self.eye[2],r_) + dp
        self.eye = r*np.cos(theta)*np.cos(psi),r*np.sin(theta)*np.cos(psi),r*np.sin(psi)
        print('Eye: ', self.eye)
        self.update_window()

    def update_window(self):
        self.vao.program['z_near'].value = 0
        self.vao.program['z_far'].value = 1000.0
        self.vao.program['ratio'].value = 2560 / 1600
        self.vao.program['fovy'].value = 60

        self.vao.program['eye'].value = self.eye
        self.vao.program['center'].value = (0, 0, 0)
        self.vao.program['up'].value = (0, 0, 1)
        self.clear = True

    def paintGL(self):
        self.ctx.viewport = (0, 0, self.width() * 2, self.height() * 2)
        self.ctx.clear(40,40,40,self.clear_gray,1.0)
        self.vao.render()

    def mouseMoveEvent(self, event):
        if self.mouseDrag:
            x,y = event.globalX(), event.globalY()
            dx = self.mouse[0] - x
            dy = self.mouse[1] - y
            print('dx,dy:\t',dx,dy)
            self.mouse = x,y
            self.rotate(dx, dy)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            x,y = event.globalX(), event.globalY()
            self.mouse = x,y
            self.mouseDrag = True

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.mouseDrag = False

    def save(self):
        t = time.time()
        print('saved at t={}'.format(t))
        self.grabFrameBuffer().save('{}.png'.format(t))
        # fbo = self.ctx.simple_framebuffer((2560, 1600))
        # fbo.use()
        # fbo.clear(0.0, 0.0, 0.0, 1.0)
        # self.vao.render(moderngl.POINTS)
        # Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1).save('{}.png'.format(time.time()))

    def keyPressEvent(self, e):
        key = e.key()
        if key == QtCore.Qt.Key_1:
            self.update_data(np.random.uniform(low=0, high=0, size=(400,400,3)), np.random.uniform(low=0, high=0, size=(400,400)))
        if key == QtCore.Qt.Key_2:
            self.update_data(np.random.uniform(low=0, high=255, size=(400,400,3)), np.random.uniform(low=0, high=300, size=(400,400)))
        elif key == QtCore.Qt.Key_S:
            self.save()


app = QtWidgets.QApplication([])
window = QGLControllerWidget()
window.resize(2560, 1280)
window.showFullScreen()
timer = QtCore.QTimer()
timer.timeout.connect(window.update)
timer.start(0)
app.exec_()
