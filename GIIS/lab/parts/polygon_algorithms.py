import sys
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsView,
    QGraphicsScene,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox, QMessageBox, QInputDialog, QCheckBox,
)
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtCore import Qt, QPointF
import time


class PolygonDrawer(QGraphicsView):
    def __init__(self, pixel_size=3):
        super().__init__()

        self.scene = QGraphicsScene()
        self.scene.setSceneRect(0, 0, 1000, 800)
        self.setScene(self.scene)

        self.pixel_size = pixel_size  
        self.polygon_points = []  
        self.is_drawing = True  
        self.line_algorithm = "Bresenham"  
        self.debug_mode = False  

        self.line_start_point = None
        self.line_end_point = None
        self.is_marking_line = False
        self.is_selecting_seed = False
        self.seed_point = None  


        self.fill_algorithm = "Simple Seed Fill"  
        self.debug_mode = False  
        self.fill_delay = 10  

        self.draw_grid()

    def start_seed_selection(self):
        """Enables seed point selection mode."""
        self.is_selecting_seed = True
        self.seed_point = None


    def simple_seed_fill(self, x, y):
        """Simple seed fill algorithm."""
        target_color = self.get_pixel_color(x, y)
        fill_color = QColor(0, 255, 0)  

        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if self.get_pixel_color(cx, cy) == target_color:
                self.fill_pixel_color(cx, cy, fill_color)
                if self.debug_mode:
                    QApplication.processEvents()
                    time.sleep(self.fill_delay / 1000)
                stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])

    def scanline_seed_fill(self, x, y):
        """Scanline seed fill algorithm with a visited set."""
        target_color = self.get_pixel_color(x, y)
        fill_color = QColor(0, 255, 0)  

        if target_color == fill_color:  
            return

        print(f"Starting scanline seed fill at ({x}, {y}) with target color: {target_color.getRgb()}")

        stack = [(x, y)]
        visited = set()  
        fill_pixels = []  

        while stack:
            cx, cy = stack.pop()

            if (cx, cy) in visited:  
                continue

            visited.add((cx, cy))

            
            left = cx
            while left >= 0 and self.get_pixel_color(left, cy) == target_color:
                left -= 1
            left += 1  

            
            right = cx
            while right < self.scene.width() // self.pixel_size and self.get_pixel_color(right, cy) == target_color:
                right += 1
            right -= 1  

            
            for px in range(left-1, right + 1):
                fill_pixels.append((px, cy))
                visited.add((px, cy))  

            
            for dy in [-1, 1]:  
                for px in range(left, right + 1):
                    if (px, cy + dy) not in visited and self.get_pixel_color(px, cy + dy) == target_color:
                        stack.append((px, cy + dy))

        
        for px, py in fill_pixels:
            self.fill_pixel_color(px, py, fill_color)
            if self.debug_mode:
                QApplication.processEvents()
                time.sleep(self.fill_delay / 1000)

    def active_edge_list_fill(self):
        """Active Edge List fill algorithm that properly fills the polygon interior with adjustments for borders."""
        if len(self.polygon_points) < 3:
            return

        edges = []
        fill_color = QColor(0, 255, 0)  

        
        n = len(self.polygon_points)
        for i in range(n):
            p1 = self.polygon_points[i]
            p2 = self.polygon_points[(i + 1) % n]

            
            if p1.y() == p2.y():
                continue

            
            if p1.y() > p2.y():
                p1, p2 = p2, p1

            
            dy = p2.y() - p1.y()
            dx = p2.x() - p1.x()
            inv_slope = dx / dy if dy != 0 else 0

            edges.append({
                "y_min": p1.y(),
                "y_max": p2.y(),
                "x": p1.x(),
                "inv_slope": inv_slope
            })

        
        edges.sort(key=lambda e: e["y_min"])

        
        y = min(edge["y_min"] for edge in edges)   
        y_max = max(edge["y_max"] for edge in edges)
        active_edges = []

        while y <= y_max:
            
            active_edges.extend(edge for edge in edges if edge["y_min"] == y)

            
            active_edges = [edge for edge in active_edges if edge["y_max"] > y]

            
            active_edges.sort(key=lambda e: e["x"])

            
            for i in range(0, len(active_edges), 2):
                if i + 1 >= len(active_edges):
                    break

                x_start = int(active_edges[i]["x"] / self.pixel_size)
                x_end = int(active_edges[i + 1]["x"] / self.pixel_size)

                
                if x_start > x_end:
                    x_start, x_end = x_end, x_start

                
                x_start += 1
                x_end -= 1

                
                if x_start <= x_end:
                    for x in range(x_start, x_end + 1):
                        self.fill_pixel_color(x, int(y / self.pixel_size), fill_color)
                        if self.debug_mode:
                            QApplication.processEvents()
                            time.sleep(self.fill_delay / 1000)

            
            for edge in active_edges:
                edge["x"] += edge["inv_slope"] * self.pixel_size

            y += self.pixel_size  

    def ordered_edge_list_fill(self):
        """Ordered Edge List fill algorithm."""
        if len(self.polygon_points) < 3:
            return

        edges = []
        fill_color = QColor(0, 255, 0)  

        
        n = len(self.polygon_points)
        for i in range(n):
            p1 = self.polygon_points[i]
            p2 = self.polygon_points[(i + 1) % n]

            
            if p1.y() == p2.y():
                continue

            
            if p1.y() > p2.y():
                p1, p2 = p2, p1

            
            y_min = p1.y() / self.pixel_size
            y_max = p2.y() / self.pixel_size
            x = p1.x() / self.pixel_size
            dx = (p2.x() - p1.x()) / (p2.y() - p1.y()) / self.pixel_size * self.pixel_size

            edges.append({
                "y_min": y_min,
                "y_max": y_max,
                "x": x,
                "dx": dx
            })

        
        edges.sort(key=lambda e: e["y_min"])

        
        y = int(min(edge["y_min"] for edge in edges))
        y_max = int(max(edge["y_max"] for edge in edges)) + 1  
        active_edges = []

        while y <= y_max:
            
            for edge in edges:
                if int(edge["y_min"]) == y:
                    active_edges.append(edge)

            
            active_edges = [e for e in active_edges if int(e["y_max"]) > y]

            
            active_edges.sort(key=lambda e: e["x"])

            
            for i in range(0, len(active_edges), 2):
                if i + 1 >= len(active_edges):
                    break

                x_start = int(active_edges[i]["x"])
                x_end = int(active_edges[i + 1]["x"])

                
                if x_start > x_end:
                    x_start, x_end = x_end, x_start

                for x in range(x_start, x_end + 1):
                    self.fill_pixel_color(x, y, fill_color)
                    if self.debug_mode:
                        QApplication.processEvents()
                        time.sleep(self.fill_delay / 1000)

            
            for edge in active_edges:
                edge["x"] += edge["dx"]

            y += 1

    def get_pixel_color(self, x, y):
        """Returns the color of the pixel at (x, y)."""
        item = self.scene.itemAt(x * self.pixel_size, y * self.pixel_size, self.transform())
        if item:
            return item.brush().color()
        return QColor(255, 255, 255)  

    def fill_pixel_color(self, x, y, color=QColor(0, 0, 0)):
        """Fills a pixel at the given grid coordinates with the specified color."""
        self.scene.addRect(
            x * self.pixel_size, y * self.pixel_size,
            self.pixel_size, self.pixel_size,
            QPen(QColor(0, 0, 0), 0), color
        )

    def set_fill_algorithm(self, algorithm):
        """Sets the fill algorithm."""
        self.fill_algorithm = algorithm

    def toggle_debug_mode(self, state):
        """Toggles the debug mode on or off."""
        self.debug_mode = state == Qt.Checked

    def seed_fill(self, x, y):
        """Starts the seed fill process."""
        if self.fill_algorithm == "Simple Seed Fill":
            self.simple_seed_fill(x, y)
        elif self.fill_algorithm == "Scanline Seed Fill":
            self.scanline_seed_fill(x, y)
        elif self.fill_algorithm == "Ordered Edge List":
            self.ordered_edge_list_fill()
        elif self.fill_algorithm == "Active Edge List":
            self.active_edge_list_fill()

    def draw_grid(self):
        
        
        
        
        
        
        
        pass

    def fill_pixel(self, x, y):
        """Fills a pixel at the given grid coordinates."""
        self.scene.addRect(
            x * self.pixel_size, y * self.pixel_size,
            self.pixel_size, self.pixel_size,
            QPen(Qt.black), QColor(0, 0, 0)
        )

    def fill_pixel_red(self, x, y):
        """Fills a pixel at the given grid coordinates with a larger red dot."""
        dot_size = self.pixel_size * 2  
        self.scene.addRect(
            (x * self.pixel_size - dot_size // 2)+5,  
            (y * self.pixel_size - dot_size // 2)+5,  
            dot_size, dot_size,  
            QPen(Qt.black), QColor(255, 0, 0)
        )

    def mousePressEvent(self, event):
        """Handles mouse clicks for polygon drawing, line marking, and seed point selection."""
        scene_position = self.mapToScene(event.pos())
        grid_x = int(scene_position.x() // self.pixel_size)
        grid_y = int(scene_position.y() // self.pixel_size)

        if self.is_selecting_seed:
            
            self.seed_point = QPointF(grid_x, grid_y)
            self.is_selecting_seed = False
            self.seed_fill(grid_x, grid_y)  
        elif self.is_drawing:
            
            self.polygon_points.append(QPointF(grid_x * self.pixel_size, grid_y * self.pixel_size))
            self.scene.addRect(
                grid_x * self.pixel_size, grid_y * self.pixel_size,
                self.pixel_size, self.pixel_size,
                QPen(Qt.black), QColor(0, 0, 0)
            )
            if len(self.polygon_points) > 1:
                prev_point = self.polygon_points[-2]
                current_point = self.polygon_points[-1]
                self.draw_line(prev_point, current_point)
        elif self.is_marking_line:
            
            if not self.line_start_point:
                self.line_start_point = QPointF(grid_x * self.pixel_size, grid_y * self.pixel_size)
                self.scene.addRect(
                    grid_x * self.pixel_size, grid_y * self.pixel_size,
                    self.pixel_size, self.pixel_size,
                    QPen(Qt.blue), QColor(0, 0, 255)
                )
            else:
                self.line_end_point = QPointF(grid_x * self.pixel_size, grid_y * self.pixel_size)
                self.scene.addRect(
                    grid_x * self.pixel_size, grid_y * self.pixel_size,
                    self.pixel_size, self.pixel_size,
                    QPen(Qt.blue), QColor(0, 0, 255)
                )
                self.draw_line_and_find_intersections()
                self.line_start_point = None
                self.line_end_point = None
                self.is_marking_line = False

    def draw_line_and_find_intersections(self):
        """Draws a line between the two marked points and finds intersections with the polygon."""
        if self.line_start_point and self.line_end_point:
            
            self.draw_line(self.line_start_point, self.line_end_point)

            
            intersections = self.find_intersections(self.line_start_point, self.line_end_point)
            for intersection in intersections:
                self.fill_pixel_red(
                    intersection.x() // self.pixel_size,
                    intersection.y() // self.pixel_size,
                )
            QMessageBox.information(self, "Intersections", f"Found {len(intersections)} intersections.")


    def mouseDoubleClickEvent(self, event):
        """Handles double-click to close the polygon."""
        if self.is_drawing and len(self.polygon_points) > 2:
            
            first_point = self.polygon_points[0]
            last_point = self.polygon_points[-1]
            self.draw_line(last_point, first_point)
            self.is_drawing = False  

    def find_intersections(self, start, end):
        """Finds intersections of a line with the polygon."""
        intersections = []
        for i in range(len(self.polygon_points)):
            p1 = self.polygon_points[i]
            p2 = self.polygon_points[(i + 1) % len(self.polygon_points)]
            intersection = self.line_segment_intersection(start, end, p1, p2)
            if intersection:
                intersections.append(intersection)
        return intersections

    def line_segment_intersection(self, p1, p2, q1, q2):
        """Finds the intersection point of two line segments."""

        def det(a, b, c, d):
            return a * d - b * c

        x1, y1, x2, y2 = p1.x(), p1.y(), p2.x(), p2.y()
        x3, y3, x4, y4 = q1.x(), q1.y(), q2.x(), q2.y()

        denominator = det(x1 - x2, y1 - y2, x3 - x4, y3 - y4)
        if denominator == 0:
            return None  

        px = det(det(x1, y1, x2, y2), x1 - x2, det(x3, y3, x4, y4), x3 - x4) / denominator
        py = det(det(x1, y1, x2, y2), y1 - y2, det(x3, y3, x4, y4), y3 - y4) / denominator

        if (min(x1, x2) <= px <= max(x1, x2) and
                min(y1, y2) <= py <= max(y1, y2) and
                min(x3, x4) <= px <= max(x3, x4) and
                min(y3, y4) <= py <= max(y3, y4)):
            return QPointF(px, py)
        return None

    def start_drawing(self):
        """Starts the drawing process."""
        self.is_drawing = True
        self.polygon_points = []  

    def reset(self):
        """Resets the scene and clears the drawn polygon."""
        self.scene.clear()
        self.polygon_points = []
        self.is_drawing = True
        self.draw_grid()  

    def is_convex_polygon(self):
        """Checks if the polygon is convex."""
        if len(self.polygon_points) < 3:
            return False  

        def cross_product_sign(p1, p2, p3):
            """Calculates the cross product of vectors (p1p2) and (p2p3)."""
            dx1 = p2.x() - p1.x()
            dy1 = p2.y() - p1.y()
            dx2 = p3.x() - p2.x()
            dy2 = p3.y() - p2.y()
            return dx1 * dy2 - dy1 * dx2

        signs = []
        n = len(self.polygon_points)
        for i in range(n):
            p1 = self.polygon_points[i]
            p2 = self.polygon_points[(i + 1) % n]
            p3 = self.polygon_points[(i + 2) % n]
            cross_product = cross_product_sign(p1, p2, p3)
            signs.append(cross_product > 0)

        
        return all(signs) or not any(signs)

    def set_line_algorithm(self, algorithm):
        """Sets the line-drawing algorithm."""
        self.line_algorithm = algorithm

    def draw_line(self, start, end):
        """Draws a line between two points using the selected algorithm."""
        if self.line_algorithm == "Bresenham":
            self.bresenham(start, end, self.debug_mode)
        elif self.line_algorithm == "CDA":
            self.cda(start, end, self.debug_mode)
        elif self.line_algorithm == "Wu":
            self.wu(start, end, self.debug_mode)

    def bresenham(self, start, end, debug_mode):
        x1, y1 = start.x() // self.pixel_size, start.y() // self.pixel_size
        x2, y2 = end.x() // self.pixel_size, end.y() // self.pixel_size
        dx = x2 - x1
        dy = y2 - y1
        sx = 1 if dx > 0 else -1
        sy = 1 if dy > 0 else -1
        dx = abs(dx)
        dy = abs(dy)
        err = dx - dy

        while True:
            self.fill_pixel(x1, y1)
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

    def cda(self, start, end, debug_mode):
        x1, y1 = start.x() // self.pixel_size, start.y() // self.pixel_size
        x2, y2 = end.x() // self.pixel_size, end.y() // self.pixel_size
        dx = x2 - x1
        dy = y2 - y1
        steps = int(max(abs(dx), abs(dy)))

        x_inc = dx / float(steps)
        y_inc = dy / float(steps)

        x, y = x1, y1
        for _ in range(steps + 1):
            self.fill_pixel(int(x), int(y))
            x += x_inc
            y += y_inc

    def wu(self, start, end, debug_mode):
        x1, y1 = start.x() / self.pixel_size, start.y() / self.pixel_size
        x2, y2 = end.x() / self.pixel_size, end.y() / self.pixel_size

        def ipart(x):
            return int(x)

        def round_(x):
            return ipart(x + 0.5)

        def fpart(x):
            return x - ipart(x)

        def rfpart(x):
            return 1 - fpart(x)

        def plot(x, y, c):
            c = 1 - c
            color = QColor(int(255 * c), int(255 * c), int(255 * c))
            self.scene.addRect(x * self.pixel_size, y * self.pixel_size,
                               self.pixel_size, self.pixel_size, QPen(color), color)

        steep = abs(y2 - y1) > abs(x2 - x1)
        if steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2

        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        dx = x2 - x1
        dy = y2 - y1
        gradient = dy / dx if dx != 0 else 1

        xend = round_(x1)
        yend = y1 + gradient * (xend - x1)
        xgap = rfpart(x1 + 0.5)
        xpxl1 = ipart(xend)
        ypxl1 = ipart(yend)
        if steep:
            plot(ypxl1, xpxl1, rfpart(yend) * xgap)
            plot(ypxl1 + 1, xpxl1, fpart(yend) * xgap)
        else:
            plot(xpxl1, ypxl1, rfpart(yend) * xgap)
            plot(xpxl1, ypxl1 + 1, fpart(yend) * xgap)
        intery = yend + gradient

        xend = round_(x2)
        yend = y2 + gradient * (xend - x2)
        xgap = fpart(x2 + 0.5)
        xpxl2 = ipart(xend)
        ypxl2 = ipart(yend)
        if steep:
            plot(ypxl2, xpxl2, rfpart(yend) * xgap)
            plot(ypxl2 + 1, xpxl2, fpart(yend) * xgap)
        else:
            plot(xpxl2, ypxl2, rfpart(yend) * xgap)
            plot(xpxl2, ypxl2 + 1, fpart(yend) * xgap)

        for x in range(xpxl1 + 1, xpxl2):
            if steep:
                plot(ipart(intery), x, rfpart(intery))
                plot(ipart(intery) + 1, x, fpart(intery))
            else:
                plot(x, ipart(intery), rfpart(intery))
                plot(x, ipart(intery) + 1, fpart(intery))
            intery += gradient

    def convex_hull_graham(self):
        """Constructs the convex hull using Graham's scan algorithm."""
        if len(self.polygon_points) < 3:
            QMessageBox.warning(self, "Error", "At least 3 points are required for a convex hull!")
            return []

        
        points = sorted(self.polygon_points, key=lambda p: (p.x(), p.y()))

        
        def cross(o, a, b):
            return (a.x() - o.x()) * (b.y() - o.y()) - (a.y() - o.y()) * (b.x() - o.x())

        
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        
        return lower[:-1] + upper[:-1]

    def convex_hull_jarvis(self):
        """Constructs the convex hull using Jarvis March (Gift Wrapping) algorithm."""
        if len(self.polygon_points) < 3:
            QMessageBox.warning(self, "Error", "At least 3 points are required for a convex hull!")
            return []

        
        hull = []
        leftmost = min(self.polygon_points, key=lambda p: p.x())
        point_on_hull = leftmost

        while True:
            hull.append(point_on_hull)
            next_point = self.polygon_points[0]
            for p in self.polygon_points:
                if next_point == point_on_hull or \
                        (p != point_on_hull and self.orientation(point_on_hull, next_point, p) < 0):
                    next_point = p
            point_on_hull = next_point
            if point_on_hull == leftmost:
                break

        return hull
    def orientation(self, p, q, r):
        """Determines the orientation of the triplet (p, q, r).
        Returns:
            0 -> p, q and r are collinear
            >0 -> Clockwise
            <0 -> Counterclockwise
        """
        return (q.y() - p.y()) * (r.x() - q.x()) - (q.x() - p.x()) * (r.y() - q.y())

    def draw_convex_hull(self, hull):
        """Draws the convex hull."""
        if not hull:
            return
        for i in range(len(hull)):
            self.draw_line(hull[i], hull[(i + 1) % len(hull)])


class ViewerWithPolygonMenu(QDialog):
    def __init__(self, pixel_size=10):
        super().__init__()
        self.setWindowTitle("Polygon Drawer")

        layout = QHBoxLayout()

        
        self.polygon_drawer = PolygonDrawer(pixel_size)

        
        button_layout = QVBoxLayout()

        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.polygon_drawer.reset)
        button_layout.addWidget(self.reset_button)

        self.algorithm_selector = QComboBox()
        self.algorithm_selector.addItems(["Bresenham", "CDA", "Wu"])
        self.algorithm_selector.currentTextChanged.connect(
            self.polygon_drawer.set_line_algorithm
        )
        button_layout.addWidget(self.algorithm_selector)

        self.convexity_button = QPushButton("Check Convexity")
        self.convexity_button.clicked.connect(self.check_convexity)
        button_layout.addWidget(self.convexity_button)

        self.hull_button = QPushButton("Build Convex Hull")
        self.hull_button.clicked.connect(self.build_convex_hull)
        button_layout.addWidget(self.hull_button)

        self.intersection_button = QPushButton("Find Intersections")
        self.intersection_button.clicked.connect(self.start_marking_line)
        button_layout.addWidget(self.intersection_button)

        

        self.seed_fill_button = QPushButton("Seed Fill")
        self.seed_fill_button.clicked.connect(self.start_seed_fill)
        button_layout.addWidget(self.seed_fill_button)

        
        self.debug_checkbox = QCheckBox("Debug Mode")
        self.debug_checkbox.stateChanged.connect(self.polygon_drawer.toggle_debug_mode)
        button_layout.addWidget(self.debug_checkbox)

        button_layout.addStretch()
        layout.addLayout(button_layout)
        layout.addWidget(self.polygon_drawer)

        self.setLayout(layout)


    def toggle_debug_mode(self, state):
        """Toggles the debug mode on or off."""
        self.polygon_drawer.debug_mode = state == Qt.Checked

    def start_marking_line(self):
        """Starts the process of marking points for the line."""
        self.polygon_drawer.is_marking_line = True
        self.polygon_drawer.is_drawing = False
        QMessageBox.information(self, "Mark Points", "Click on the scene to mark two points for the line.")


    def build_convex_hull(self):
        """Builds the convex hull using the selected method."""
        method, ok = QInputDialog.getItem(
            self,
            "Select Method",
            "Choose a method to construct the convex hull:",
            ["Graham's Scan", "Jarvis March"],
            0,
            False,
        )

        if not ok:
            return

        if method == "Graham's Scan":
            hull = self.polygon_drawer.convex_hull_graham()
        elif method == "Jarvis March":
            hull = self.polygon_drawer.convex_hull_jarvis()
        else:
            return

        
        self.polygon_drawer.scene.clear()
        self.polygon_drawer.draw_grid()  

        
        for point in self.polygon_drawer.polygon_points:
            self.polygon_drawer.scene.addRect(
                point.x(), point.y(),
                self.polygon_drawer.pixel_size, self.polygon_drawer.pixel_size,
                QPen(Qt.black), QColor(100, 0, 0)
            )

        
        self.polygon_drawer.draw_convex_hull(hull)

    def check_convexity(self):
        """Checks if the polygon is convex and displays the result."""
        is_convex = self.polygon_drawer.is_convex_polygon()
        message = "The polygon is convex!" if is_convex else "The polygon is not convex!"
        QMessageBox.information(self, "Convexity Check", message)

    def start_seed_fill(self):
        """Starts the seed fill process by allowing the user to select a seed point."""
        algorithms = ["Ordered Edge List", "Active Edge List", "Simple Seed Fill", "Scanline Seed Fill"]
        algorithm, ok = QInputDialog.getItem(
            self,
            "Choose Fill Algorithm",
            "Select an algorithm:",
            algorithms,
            0,
            False,
        )
        if ok:
            self.polygon_drawer.set_fill_algorithm(algorithm)
            self.polygon_drawer.start_seed_selection()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ViewerWithPolygonMenu(pixel_size=10)
    viewer.show()
    sys.exit(app.exec_())