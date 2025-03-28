import sys
from PyQt5.QtWidgets import (QApplication, QGraphicsView,
                             QGraphicsScene, QVBoxLayout, QFrame, QPushButton, QHBoxLayout, QDialog)
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QPen, QBrush, QPainter


class DelaunayTriangulation:
    def __init__(self, points):
        self.points = points
        self.triangles = []

        
        self._create_super_triangle()

        
        for point in points:
            self._add_point(point)

        
        self._remove_super_triangle()

    def _create_super_triangle(self):
        min_x = min(p.x() for p in self.points)
        min_y = min(p.y() for p in self.points)
        max_x = max(p.x() for p in self.points)
        max_y = max(p.y() for p in self.points)

        
        width = max_x - min_x
        height = max_y - min_y

        
        p1 = QPointF(min_x - width, min_y - height * 2)
        p2 = QPointF(max_x + width, min_y - height * 2)
        p3 = QPointF(min_x + width / 2, max_y + height * 2)

        self.super_triangle = (p1, p2, p3)
        self.triangles.append((p1, p2, p3))

    def _add_point(self, point):
        bad_triangles = []

        
        for triangle in self.triangles:
            if self._is_point_in_circumcircle(point, triangle):
                bad_triangles.append(triangle)

        polygon = []
        
        for triangle in bad_triangles:
            for i in range(3):
                edge = (triangle[i], triangle[(i + 1) % 3])
                is_shared = False

                for other_triangle in bad_triangles:
                    if triangle == other_triangle:
                        continue

                    for j in range(3):
                        other_edge = (other_triangle[j], other_triangle[(j + 1) % 3])
                        if (edge[0] == other_edge[1] and edge[1] == other_edge[0]):
                            is_shared = True
                            break

                if not is_shared:
                    polygon.append(edge)

        
        for triangle in bad_triangles:
            self.triangles.remove(triangle)

        
        for edge in polygon:
            new_triangle = (edge[0], edge[1], point)
            self.triangles.append(new_triangle)

    def _remove_super_triangle(self):
        p1, p2, p3 = self.super_triangle
        triangles_to_remove = []

        for triangle in self.triangles:
            if (p1 in triangle) or (p2 in triangle) or (p3 in triangle):
                triangles_to_remove.append(triangle)

        for triangle in triangles_to_remove:
            self.triangles.remove(triangle)

    def _is_point_in_circumcircle(self, point, triangle):
        a, b, c = triangle
        ax, ay = a.x(), a.y()
        bx, by = b.x(), b.y()
        cx, cy = c.x(), c.y()
        px, py = point.x(), point.y()

        
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if d == 0:
            return False  

        ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay) + (cx ** 2 + cy ** 2) * (ay - by)) / d
        uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx) + (cx ** 2 + cy ** 2) * (bx - ax)) / d

        radius_sq = (ax - ux) ** 2 + (ay - uy) ** 2
        dist_sq = (px - ux) ** 2 + (py - uy) ** 2

        return dist_sq <= radius_sq

    def compute_voronoi(self):
        voronoi_edges = []
        circumcenters = {}

        for triangle in self.triangles:
            
            triangle_key = tuple((point.x(), point.y()) for point in triangle)

            
            cx, cy = self._compute_circumcenter(triangle)
            circumcenters[triangle_key] = (cx, cy)  

        for triangle in self.triangles:
            
            triangle_key = tuple((point.x(), point.y()) for point in triangle)

            for i in range(3):
                neighbor = self._find_neighbor(triangle, i)
                if neighbor:
                    
                    neighbor_key = tuple((point.x(), point.y()) for point in neighbor)

                    
                    edge_start = circumcenters[triangle_key]
                    edge_end = circumcenters[neighbor_key]
                    voronoi_edges.append((QPointF(*edge_start), QPointF(*edge_end)))

        return voronoi_edges

    def _compute_circumcenter(self, triangle):
        a, b, c = triangle
        ax, ay = a.x(), a.y()
        bx, by = b.x(), b.y()
        cx, cy = c.x(), c.y()

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if d == 0:
            return float('inf'), float('inf')  

        ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay) + (cx ** 2 + cy ** 2) * (ay - by)) / d
        uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx) + (cx ** 2 + cy ** 2) * (bx - ax)) / d

        return ux, uy

    def _find_neighbor(self, triangle, edge_index):
        edge = (triangle[edge_index], triangle[(edge_index + 1) % 3])
        for other_triangle in self.triangles:
            if other_triangle == triangle:
                continue
            for i in range(3):
                other_edge = (other_triangle[i], other_triangle[(i + 1) % 3])
                if edge[0] == other_edge[1] and edge[1] == other_edge[0]:
                    return other_triangle
        return None


class VoronoiDiagramView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setFrameShape(QFrame.NoFrame)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.points = []
        self.voronoi_edges = []
        self.point_pen = QPen(Qt.red, 3)
        self.point_brush = QBrush(Qt.red)
        self.edge_pen = QPen(Qt.blue, 1)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            point = QPointF(scene_pos.x(), scene_pos.y())
            self.points.append(point)
            self.update_voronoi()
            self.draw_scene()

    def update_voronoi(self):
        if len(self.points) < 3:
            self.voronoi_edges = []
            return

        triangulation = DelaunayTriangulation(self.points)
        self.voronoi_edges = triangulation.compute_voronoi()

    def draw_scene(self):
        self.scene.clear()
        for edge in self.voronoi_edges:
            self.scene.addLine(edge[0].x(), edge[0].y(), edge[1].x(), edge[1].y(), self.edge_pen)
        for point in self.points:
            self.scene.addEllipse(point.x() - 3, point.y() - 3, 6, 6, self.point_pen, self.point_brush)

    def clear_points(self):
        self.points = []
        self.voronoi_edges = []
        self.scene.clear()
        self.update()


class VoronoiViewer(QDialog):
    def __init__(self):
        super().__init__()

        self.voronoi_view = VoronoiDiagramView()

        clear_button = QPushButton("Clear Points")
        clear_button.clicked.connect(self.voronoi_view.clear_points)

        button_layout = QVBoxLayout()
        button_layout.addWidget(clear_button)
        button_layout.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.voronoi_view)
        self.setLayout(main_layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoronoiViewer()
    window.show()
    sys.exit(app.exec_())