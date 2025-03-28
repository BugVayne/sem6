import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QDialog, QLabel
from PyQt5.QtCore import Qt

from parts.circle_algorithm import CircleAlgorithmDialog, CircleDrawingArea
from parts.curve_algorithms import CurveDrawClass, CurveAlgorithmDialog
from parts.line_algorithm import LineDrawingArea, LineAlgorithmDialog
from parts.object_algorithms import ViewerWithMenu, Object3DViewerDialog
from parts.polygon_algorithms import ViewerWithPolygonMenu
from parts.triangulation_algorithms import DelaunayViewer
from parts.voronoi import VoronoiViewer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Graphic Editor")
        self.setGeometry(100, 100, 1300, 900)

        self.drawing_area = LineDrawingArea()
        self.drawing_area.draw_grid()
        self.setCentralWidget(self.drawing_area)

        self.create_menu()

        self.overlay_label = QLabel(self)
        self.overlay_label.setGeometry(1000, 20, 280, 100)
        self.overlay_label.setAlignment(Qt.AlignTop | Qt.AlignRight)
        self.overlay_label.setWordWrap(True)
        self.overlay_label.setStyleSheet("""
            background-color: rgba(0, 0, 0, 128);  
            color: white;  /* White text */
            font-size: 14px;
            border-radius: 4px;
            padding: 10px;
        """)
        self.overlay_label.setVisible(False)

    def clear_canvas(self):
        if self.drawing_area:
            self.drawing_area.scene.clear()
            self.drawing_area.draw_grid()

    def create_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("Menu")
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        clear_action = QAction("Clear canvas", self)
        clear_action.triggered.connect(self.clear_canvas)
        file_menu.addAction(clear_action)

        line_action = QAction("Draw a Line", self)
        line_action.triggered.connect(self.show_algorithm_dialog)
        menu_bar.addAction(line_action)


        circle_action = QAction("Draw a Circle", self)
        circle_action.triggered.connect(self.show_circle_algorithm_dialog)
        menu_bar.addAction(circle_action)


        curve_action = QAction("Draw a Curve", self)
        curve_action.triggered.connect(self.show_curve_algorithm_dialog)
        menu_bar.addAction(curve_action)

        object3d_action = QAction("3D object", self)
        object3d_action.triggered.connect(self.show_3d_object_dialog)
        menu_bar.addAction(object3d_action)


        polygon_action = QAction("Polygon", self)
        polygon_action.triggered.connect(self.show_polygon_dialog)
        menu_bar.addAction(polygon_action)

        delaunay_action = QAction("Delaunay", self)
        delaunay_action.triggered.connect(self.delaunay_dialog)
        menu_bar.addAction(delaunay_action)

        voronoi_action = QAction("Voronoi", self)
        voronoi_action.triggered.connect(self.voronoi_dialog)
        menu_bar.addAction(voronoi_action)

    def voronoi_dialog(self):
        self.drawing_area = VoronoiViewer()
        self.setCentralWidget(self.drawing_area)

    def delaunay_dialog(self):
        self.drawing_area = DelaunayViewer()
        self.setCentralWidget(self.drawing_area)

    def show_polygon_dialog(self):
        self.drawing_area = ViewerWithPolygonMenu(5)
        self.setCentralWidget(self.drawing_area)


    def show_3d_object_dialog(self):
        dialog = Object3DViewerDialog()
        if dialog.exec_() == QDialog.Accepted:
            object_file = dialog.get_selection()
            if object_file:
                prefix = "C:/ME/BSUIR/sem6/GIIS/lab/objects/"
                self.drawing_area = ViewerWithMenu()
                self.drawing_area.viewer.load_object(prefix + object_file)
                self.drawing_area.viewer.show()

                self.setCentralWidget(self.drawing_area)



    def show_circle_algorithm_dialog(self):
        dialog = CircleAlgorithmDialog()
        if dialog.exec_() == QDialog.Accepted:
            curve, debug_mode, clear_canvas = dialog.get_selection()
            if curve:
                self.drawing_area = CircleDrawingArea()
                self.drawing_area.draw_grid()
                self.setCentralWidget(self.drawing_area)
                self.drawing_area.set_algorithm(curve, debug_mode, clear_canvas)
                self.update_overlay_label(curve, debug_mode, clear_canvas, "Second-Order Curves")
                self.overlay_label.raise_()

    def show_curve_algorithm_dialog(self):
        dialog = CurveAlgorithmDialog()
        if dialog.exec_() == QDialog.Accepted:
            curve, debug_mode, clear_canvas = dialog.get_selection()
            if curve:
                self.drawing_area = CurveDrawClass()
                self.drawing_area.draw_grid()
                self.setCentralWidget(self.drawing_area)
                self.drawing_area.set_algorithm(curve, debug_mode, clear_canvas)
                self.update_overlay_label(curve, debug_mode, clear_canvas, "Curves")
                self.overlay_label.raise_()


    def show_algorithm_dialog(self):
        dialog = LineAlgorithmDialog()
        if dialog.exec_() == QDialog.Accepted:
            algorithm, debug_mode, clear_canvas = dialog.get_selection()
            if algorithm:
                self.drawing_area = LineDrawingArea()
                self.setCentralWidget(self.drawing_area)
                self.drawing_area.draw_grid()
                self.drawing_area.set_algorithm(algorithm, debug_mode, clear_canvas)
                self.update_overlay_label(algorithm, debug_mode, clear_canvas, "Lines")
                self.overlay_label.raise_()

    def update_overlay_label(self, algorithm, debug_mode, clear_canvas, mode):
        debug_status = "ON" if debug_mode else "OFF"
        clear_status = "ON" if clear_canvas else "OFF"

        self.overlay_label.setText(
            f"Algorithm/figure: {algorithm}\n"
            f"Debug Mode: {debug_status}\n"
            f"Clear Canvas: {clear_status}\n"
            f"Mode: {mode}"
        )
        self.overlay_label.setVisible(True)
        self.overlay_label.raise_()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())