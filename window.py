#!/usr/bin/env python
import gi
gi.require_version('Gtk','3.0')
from gi.repository import Gtk
import random
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas
import cinematica_inversa as ci
from scipy.optimize import fsolve
print("HOLA")
def add_mark(button):
    value = scale.get_value()
    scale.add_mark(value, Gtk.PositionType.LEFT, "Mark")

def clear_marks(button):
    scale.clear_marks()

def scale_orientation(radiobutton):
	if radiobutton.get_label() == "Horizontal Scale":
		scale.set_orientation(Gtk.Orientation.HORIZONTAL)
	else:
		scale.set_orientation(Gtk.Orientation.VERTICAL)
estado = [0,0,0,0,0,0]
def replot(adjustment):
    estado[adjustment.tipo] = adjustment.get_value()
    RT = ci.calc_RT(*estado)
    V = ci.calc_V(RT)
    A = fsolve(ci.Finverso,ci.A0,args = V)
    A = A.reshape(3,6)
    ax.clear()
    ci.plot_plataforma(ax, A,V)
    #canvas.set_size_request(400,400)

    #canvas.figure = fig
    canvas.queue_draw()

    


window = Gtk.Window()
window.set_default_size(200, 200)
window.connect("destroy", lambda q: Gtk.main_quit())

grid = Gtk.Grid()
window.add(grid)

min_inicial = [-10,-10,-10,-10,-10,-10]
max_inicial = [10,10,10,10,10,10]
for i in range(6):
    adj = Gtk.Adjustment(0,min_inicial[i], max_inicial[i], 1, 10, 0)
    adj.tipo = i
    adj.connect("value-changed", replot)
    scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=adj)
    scale.set_size_request(100,20)

    scale.set_value_pos(Gtk.PositionType.BOTTOM)
    grid.attach(scale, 0, i, 2, 1)


sw = Gtk.ScrolledWindow()
grid.attach(sw,                    3, 0, 1, 8)
fig, ax = ci.create_figure()
ci.plot_plataforma(ax,ci.A0, ci.V0)

canvas = FigureCanvas(fig)
canvas.set_size_request(400,400)
sw.add_with_viewport(canvas)
sw.set_vexpand(True)
sw.set_hexpand(True)

window.show_all()

Gtk.main()
