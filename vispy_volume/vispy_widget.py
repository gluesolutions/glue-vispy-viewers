from __future__ import absolute_import, division, print_function
import numpy as np
from itertools import cycle

from glue.qt.widgets.data_viewer import DataViewer
from glue.external.qt import QtGui, QtCore
from vispy import scene
from vispy.color import get_colormaps, BaseColormap

__all__ = ['QtVispyWidget']


# create colormaps that work well for translucent and additive volume rendering
class TransFire(BaseColormap):
    glsl_map = """
    vec4 translucent_fire(float t) {
        return vec4(pow(t, 0.5), t, t*t, max(0, t*1.05 - 0.05));
    }
    """


class TransGrays(BaseColormap):
    glsl_map = """
    vec4 translucent_grays(float t) {
        return vec4(t, t, t, t*0.05);
    }
    """


class QtVispyWidget(QtGui.QWidget):

    def __init__(self, parent=None):
        super(QtVispyWidget, self).__init__(parent=parent)

        self.canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
        self.canvas.measure_fps()

        self.data = None
        self.volume1 = view = None
        self.cam1 = cam2 = cam3 = None

        self.canvas.events.key_press.connect(self.on_key_press)

    def set_data(self, data):
        self.data = data

    def set_canvas(self):
        if self.data is None:
            return
        vol1 = np.nan_to_num(np.array(self.data))

        # Prepare canvas
        # Set up a viewbox to display the image with interactive pan/zoom
        self.view = self.canvas.central_widget.add_view()

        # Set whether we are emulating a 3D texture
        emulate_texture = False

        # Create the volume visuals, only one is visible
        self.volume1 = scene.visuals.Volume(vol1, parent=self.view.scene, threshold=0.1,
                                       emulate_texture=emulate_texture)
        # volume1.transform = scene.STTransform(translate=(64, 64, 0))

        # Create two cameras (1 for firstperson, 3 for 3d person)
        fov = 60.
        self.cam1 = scene.cameras.FlyCamera(parent=self.view.scene, fov=fov, name='Fly')
        self.cam2 = scene.cameras.TurntableCamera(parent=self.view.scene, fov=fov,
                                             name='Turntable')
        self.cam3 = scene.cameras.ArcballCamera(parent=self.view.scene, fov=fov, name='Arcball')
        self.view.camera = self.cam2  # Select turntable at firstate_texture=emulate_texture)

    def set_colormap(self):
        # Setup colormap iterators
        opaque_cmaps = cycle(get_colormaps())
        translucent_cmaps = cycle([TransFire(), TransGrays()])
        opaque_cmap = next(opaque_cmaps)
        translucent_cmap = next(translucent_cmaps)
        result = 1

    # Implement key presses
    # @canvas.events.key_press.connect
    def on_key_press(self, event):
        # result =1 # invoke every press...
        # global opaque_cmap, translucent_cmap, result
        # if self.view is None:
        #     return
        if event.text == '1':
        # if event.key() == QtCore.Qt.Key_Shift:
            cam_toggle = {self.cam1: self.cam2, self.cam2: self.cam3, self.cam3: self.cam1}
            self.view.camera = cam_toggle.get(self.view.camera, self.cam2)
            self.canvas.render()
    #        print(view.camera.name + ' camera')
        elif event.text == '2':
            methods = ['mip', 'translucent', 'iso', 'additive']
            method = methods[(methods.index(self.volume1.method) + 1) % 4]
            print("Volume render method: %s" % method)
            # self.cmap = opaque_cmap if method in ['mip', 'iso'] else translucent_cmap
            self.volume1.method = method
            # self.volume1.cmap = cmap
        elif event.text == '3':
            self.volume1.visible = not self.volume1.visible

        # Color scheme cannot work now
        '''elif event.text == '4':
            if self.volume1.method in ['mip', 'iso']:
                # cmap = opaque_cmap = next(opaque_cmaps)
            else:
                # cmap = translucent_cmap = next(translucent_cmaps)
            # volume1.cmap = cmap

        elif event.text == '0':
            self.cam1.set_range()
            self.cam3.set_range()
        elif event.text != '' and event.text in '[]':
            s = -0.025 if event.text == '[' else 0.025
            volume1.threshold += s
            th = volume1.threshold if volume1.visible else volume2.threshold
    #        print("Isosurface threshold: %0.3f" % th)
        # Add zoom out functionality for the third dimension
        elif event.text != '' and event.text in '=-':
            z = -1 if event.text == '-' else +1
            result += z
            if result > 0:
                volume1.transform = scene.STTransform(scale=(1, 1, result))
            else:
                result = 1
    #        print("Volume scale: %d" % result)
        # create colormaps that work well for translucent and additive volume rendering
class TransFire(BaseColormap):
    glsl_map = """
        vec4 translucent_fire(float t) {
        return vec4(pow(t, 0.5), t, t*t, max(0, t*1.05 - 0.05));
        }
        """

class TransGrays(BaseColormap):
    glsl_map = """
        vec4 translucent_grays(float t) {
        return vec4(t, t, t, t*0.05);
        }
        """
# Setup colormap iterators
opaque_cmaps = cycle(get_colormaps())
translucent_cmaps = cycle([TransFire(), TransGrays()])
opaque_cmap = next(opaque_cmaps)
translucent_cmap = next(translucent_cmaps)
result = 1'''
