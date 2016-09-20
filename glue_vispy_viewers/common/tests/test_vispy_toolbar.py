# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import warnings

from qtpy import QtWidgets

from glue.config import viewer_tool
from glue.viewers.common.qt.tool import Tool
from glue.viewers.common.qt.toolbar import BasicToolbar
from glue.viewers.common.qt.mouse_mode import MouseMode
from glue.icons.qt import get_icon
from glue.core.tests.util import simple_session

from ..vispy_widget import VispyWidget
from ..toolbar import VispyDataViewerToolbar
from ..new_toolbar import VispyViewerToolbar, SaveTool
from ..vispy_data_viewer import BaseVispyViewer

# we need to test both toolbar and tool here
# solve the viewer test bug first


class ExampleViewer(BaseVispyViewer):

    _toolbar_cls = VispyViewerToolbar

    def __init__(self, session, parent=None):
        super(ExampleViewer, self).__init__(session, parent=parent)
        v = VispyWidget(parent)
        self.central_widget = v
        self.setCentralWidget(self.central_widget)
        self.toolbar = self._toolbar_cls(vispy_widget=v, parent=self)

    def initialize_toolbar(self):
        super(ExampleViewer, self).initialize_toolbar()
        self.tool = SaveTool(self)
        self.toolbar.add_tool(self.tool)

    def _update_attributes(self):
        pass

    def callback(self, mode):
        self._called_back = True


def test_toolbar():
    session = simple_session()
    with warnings.catch_warnings(record=True) as w:
        viewer = ExampleViewer(session)
    assert len(w) == 1
