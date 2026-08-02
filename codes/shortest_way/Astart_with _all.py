import numpy as np
import heapq
from typing import List, Tuple, Dict, Set
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.widgets import Button
import random
import time
from collections import deque
import json
import os
from matplotlib import patheffects

class UltimateGridPathfinder:
    def __init__(self, rows: int = 10, cols: int = 10):
        """Ultimate interactive grid pathfinder with polished interface"""
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)
        self.obstacles = set()
        self.point_a = None
        self.point_b = None
        self.path = []
        self.waypoints = []
        self.explored_nodes = set()
        self.frontier_nodes = set()
        
        # Algorithm selection
        self.current_algorithm = 'A*'
        self.available_algorithms = {
            'A*': self.a_star_search,
            'Dijkstra': self.dijkstra_search,
            'BFS': self.bfs_search,
            'Greedy BFS': self.greedy_bfs_search,
            'Bidirectional A*': self.bidirectional_a_star
        }
        
        # Terrain system
        self.terrain_types = {
            'normal': {'cost': 1, 'color': '#FFFFFF', 'name': 'Normal'},
            'grass': {'cost': 2, 'color': '#90EE90', 'name': 'Grass'},
            'mud': {'cost': 4, 'color': '#D2691E', 'name': 'Mud'},
            'road': {'cost': 0.7, 'color': '#C0C0C0', 'name': 'Road'},
            'water': {'cost': 8, 'color': '#6495ED', 'name': 'Water'}
        }
        self.terrain_grid = np.full((rows, cols), 'normal', dtype=object)
        self.current_terrain = 'normal'
        
        # Statistics tracking
        self.stats = {
            'nodes_explored': 0,
            'path_length': 0,
            'computation_time': 0,
            'path_cost': 0,
            'turns': 0,
            'max_frontier_size': 0
        }
        
        # Store all button references
        self.all_buttons = []
        self.algo_buttons = {}
        
        # Setup the figure with better proportions
        self.fig = plt.figure(figsize=(18, 11))
        self.fig.patch.set_facecolor('#2C3E50')
        self.setup_layout()
        self.setup_ui()
        self.connect_events()
        self.draw_grid()
        
    def setup_layout(self):
        """Setup clean, non-overlapping layout"""
        # Main grid area (left side) - centered and larger
        self.ax_grid = plt.axes([0.06, 0.18, 0.56, 0.78])
        self.ax_grid.set_facecolor('#ECF0F1')
        
        # Right panel - split into sections with proper spacing
        
        # Algorithm panel (top right) - now contains actual buttons
        self.ax_algo_bg = plt.axes([0.66, 0.78, 0.30, 0.18])
        self.ax_algo_bg.set_facecolor('#34495E')
        self.ax_algo_bg.axis('off')
        
        # Statistics panel (middle right)
        self.ax_stats = plt.axes([0.66, 0.50, 0.30, 0.26])
        self.ax_stats.set_facecolor('#34495E')
        self.ax_stats.axis('off')
        
        # Terrain legend (bottom right)
        self.ax_terrain = plt.axes([0.66, 0.18, 0.30, 0.30])
        self.ax_terrain.set_facecolor('#34495E')
        self.ax_terrain.axis('off')
        
    def setup_ui(self):
        """Setup organized UI with clickable algorithm buttons"""
        
        # Bottom control panel background
        control_bg = plt.axes([0.06, 0.02, 0.90, 0.14])
        control_bg.set_facecolor('#2C3E50')
        control_bg.axis('off')
        
        # Button dimensions
        btn_width = 0.09
        btn_height = 0.035
        start_x = 0.07
        y_row1 = 0.12
        y_row2 = 0.07
        y_row3 = 0.02
        spacing = 0.005
        
        # ===== ROW 1: Mode buttons =====
        modes = [
            ('📍 Point A', '#2ECC71', '#27AE60'),
            ('📍 Point B', '#E74C3C', '#C0392B'),
            ('📍 Waypoint', '#F39C12', '#E67E22'),
            ('🚧 Obstacle', '#95A5A6', '#7F8C8D'),
            ('🗑️ Remove', '#9B59B6', '#8E44AD'),
        ]
        
        self.mode_buttons = []
        for i, (text, color, hover) in enumerate(modes):
            x = start_x + i * (btn_width + spacing)
            ax_btn = plt.axes([x, y_row1, btn_width, btn_height])
            btn = Button(ax_btn, text, color=color, hovercolor=hover)
            btn.label.set_fontsize(8)
            btn.label.set_fontweight('bold')
            self.mode_buttons.append(btn)
            self.all_buttons.append(btn)
        
        # ===== ROW 2: Terrain painting buttons =====
        terrains = [
            ('Normal', '#FFFFFF', '#BDC3C7', 'normal'),
            ('🌿 Grass', '#90EE90', '#27AE60', 'grass'),
            ('💧 Mud', '#D2691E', '#A0522D', 'mud'),
            ('🛣️ Road', '#C0C0C0', '#95A5A6', 'road'),
            ('🌊 Water', '#6495ED', '#2980B9', 'water'),
        ]
        
        self.terrain_buttons = []
        for i, (text, color, hover, terrain) in enumerate(terrains):
            x = start_x + i * (btn_width + spacing)
            ax_btn = plt.axes([x, y_row2, btn_width, btn_height])
            btn = Button(ax_btn, text, color=color, hovercolor=hover)
            btn.label.set_fontsize(7)
            self.terrain_buttons.append(btn)
            self.all_buttons.append(btn)
            btn.on_clicked(lambda x, t=terrain: self.set_terrain_mode(t))
        
        # ===== ROW 3: Action buttons =====
        actions = [
            ('🔍 Find Path', '#3498DB', '#2980B9'),
            ('🎲 Random Obs', '#E67E22', '#D35400'),
            ('🗑️ Clear All', '#E74C3C', '#C0392B'),
            ('💾 Save', '#2ECC71', '#27AE60'),
            ('📂 Load', '#F1C40F', '#F39C12'),
        ]
        
        self.action_buttons = []
        for i, (text, color, hover) in enumerate(actions):
            x = start_x + i * (btn_width + spacing)
            ax_btn = plt.axes([x, y_row3, btn_width, btn_height])
            btn = Button(ax_btn, text, color=color, hovercolor=hover)
            btn.label.set_fontsize(8)
            btn.label.set_fontweight('bold')
            self.action_buttons.append(btn)
            self.all_buttons.append(btn)
        
        # ===== Connect mode button callbacks =====
        self.mode_buttons[0].on_clicked(lambda x: self.set_mode('Point A'))
        self.mode_buttons[1].on_clicked(lambda x: self.set_mode('Point B'))
        self.mode_buttons[2].on_clicked(lambda x: self.set_mode('Waypoint'))
        self.mode_buttons[3].on_clicked(lambda x: self.set_mode('Obstacle'))
        self.mode_buttons[4].on_clicked(lambda x: self.set_mode('Remove'))
        
        # ===== Connect action button callbacks =====
        self.action_buttons[0].on_clicked(lambda x: self.manual_find_path())
        self.action_buttons[1].on_clicked(lambda x: self.add_random_obstacles())
        self.action_buttons[2].on_clicked(lambda x: self.clear_all())
        self.action_buttons[3].on_clicked(lambda x: self.save_scenario())
        self.action_buttons[4].on_clicked(lambda x: self.load_scenario())
        
        # ===== RIGHT PANEL: Algorithm buttons (ACTUAL CLICKABLE BUTTONS) =====
        algo_start_y = 0.92
        algo_height = 0.025
        algo_spacing = 0.008
        algo_width = 0.26
        algo_x = 0.68
        
        for i, algo_name in enumerate(self.available_algorithms.keys()):
            y_pos = algo_start_y - i * (algo_height + algo_spacing)
            ax_algo_btn = plt.axes([algo_x, y_pos, algo_width, algo_height])
            
            # Set color based on whether it's the current algorithm
            btn_color = '#3498DB' if algo_name == self.current_algorithm else '#7F8C8D'
            btn_hover = '#2980B9' if algo_name == self.current_algorithm else '#95A5A6'
            
            btn = Button(ax_algo_btn, algo_name, color=btn_color, hovercolor=btn_hover)
            btn.label.set_fontsize(9)
            btn.label.set_color('white')
            btn.label.set_fontweight('bold' if algo_name == self.current_algorithm else 'normal')
            
            # Store reference and connect callback
            self.algo_buttons[algo_name] = btn
            self.all_buttons.append(btn)
            btn.on_clicked(lambda x, a=algo_name: self.switch_algorithm(a))
        
        # Current mode and algorithm indicator
        self.current_mode = 'Point A'
        self.mode_indicator = self.fig.text(0.5, 0.165, 
            f'Current Mode: {self.current_mode} | Algorithm: {self.current_algorithm}',
            ha='center', fontsize=10, fontweight='bold', color='#ECF0F1',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#34495E', edgecolor='#2C3E50', alpha=0.9))
        
        # Add title labels for right panels
        self.algo_title = self.fig.text(0.81, 0.96, '⚡ ALGORITHMS (Click to Select)', 
                                        ha='center', fontsize=11, fontweight='bold', 
                                        color='#ECF0F1')
        self.stats_title = self.fig.text(0.81, 0.76, '📊 STATISTICS', 
                                         ha='center', fontsize=11, fontweight='bold', 
                                         color='#ECF0F1')
        self.terrain_title = self.fig.text(0.81, 0.48, '🌍 TERRAIN TYPES', 
                                           ha='center', fontsize=11, fontweight='bold', 
                                           color='#ECF0F1')
        
    def switch_algorithm(self, algorithm):
        """Switch between pathfinding algorithms"""
        self.current_algorithm = algorithm
        
        # Update all algorithm button colors
        for algo_name, btn in self.algo_buttons.items():
            if algo_name == algorithm:
                btn.color = '#3498DB'
                btn.hovercolor = '#2980B9'
                btn.label.set_fontweight('bold')
            else:
                btn.color = '#7F8C8D'
                btn.hovercolor = '#95A5A6'
                btn.label.set_fontweight('normal')
            btn.ax.set_facecolor(btn.color)
        
        self.mode_indicator.set_text(
            f'Current Mode: {self.current_mode} | Algorithm: {algorithm}'
        )
        print(f"\n🔄 Switched to {algorithm} algorithm")
        
        if self.point_a and self.point_b:
            self.find_path()
        self.draw_grid()
        
    def set_terrain_mode(self, terrain_type):
        """Switch terrain painting mode"""
        self.current_mode = 'Terrain'
        self.current_terrain = terrain_type
        terrain_name = self.terrain_types[terrain_type]['name']
        self.mode_indicator.set_text(
            f'Painting Terrain: {terrain_name} | Algorithm: {self.current_algorithm}'
        )
        print(f"\n🎨 Painting terrain: {terrain_name} (Cost: {self.terrain_types[terrain_type]['cost']})")
        self.fig.canvas.draw_idle()
        
    def set_mode(self, mode):
        """Change the current editing mode"""
        self.current_mode = mode
        self.mode_indicator.set_text(
            f'Current Mode: {mode} | Algorithm: {self.current_algorithm}'
        )
        print(f"\n🖱️  Mode: {mode}")
        
        # Reset all mode buttons to default colors
        default_colors = ['#2ECC71', '#E74C3C', '#F39C12', '#95A5A6', '#9B59B6']
        mode_names = ['Point A', 'Point B', 'Waypoint', 'Obstacle', 'Remove']
        
        for i, btn in enumerate(self.mode_buttons):
            if mode_names[i] in mode:
                btn.ax.set_facecolor('#E74C3C')
                btn.label.set_color('white')
            else:
                btn.ax.set_facecolor(default_colors[i])
                btn.label.set_color('black')
        
        self.fig.canvas.draw_idle()
        
    def connect_events(self):
        """Connect mouse and keyboard events"""
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.key is None:
            return
            
        key = event.key.upper()
        
        shortcuts = {
            'A': 'Point A', 'B': 'Point B', 'W': 'Waypoint',
            'O': 'Obstacle', 'R': 'Remove', 'F': 'Find Path',
            'C': 'Clear All', '1': 'normal', '2': 'grass',
            '3': 'mud', '4': 'road', '5': 'water',
            'G': 'Random Obstacles', 'S': 'Save', 'L': 'Load'
        }
        
        if key in shortcuts:
            if key in ['1', '2', '3', '4', '5']:
                self.set_terrain_mode(shortcuts[key])
            elif key == 'F':
                self.manual_find_path()
            elif key == 'G':
                self.add_random_obstacles()
            elif key == 'C':
                self.clear_all()
            elif key == 'S':
                self.save_scenario()
            elif key == 'L':
                self.load_scenario()
            else:
                self.set_mode(shortcuts[key])
        
    def on_click(self, event):
        """Handle mouse clicks on the grid"""
        if event.inaxes != self.ax_grid:
            return
        
        col = int(event.xdata)
        row = (self.rows - 1) - int(event.ydata)
        
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return
        
        print(f"\n📍 Clicked: ({row}, {col})")
        
        if self.current_mode == 'Point A':
            self.point_a = (row, col)
            print(f"✅ Point A: {self.point_a}")
        elif self.current_mode == 'Point B':
            self.point_b = (row, col)
            print(f"✅ Point B: {self.point_b}")
        elif self.current_mode == 'Waypoint':
            if (row, col) not in self.waypoints and (row, col) != self.point_a and (row, col) != self.point_b:
                self.waypoints.append((row, col))
                print(f"📍 Waypoint added: ({row}, {col}) | Total: {len(self.waypoints)}")
        elif self.current_mode == 'Obstacle':
            if (row, col) != self.point_a and (row, col) != self.point_b and (row, col) not in self.waypoints:
                self.obstacles.add((row, col))
                self.grid[row, col] = 1
                print(f"🚧 Obstacle: ({row}, {col}) | Total: {len(self.obstacles)}")
        elif self.current_mode == 'Terrain':
            if (row, col) not in self.obstacles:
                self.terrain_grid[row, col] = self.current_terrain
        elif self.current_mode == 'Remove':
            removed = False
            if (row, col) in self.obstacles:
                self.obstacles.remove((row, col))
                self.grid[row, col] = 0
                removed = True
                print(f"🗑️  Removed obstacle at ({row}, {col})")
            if (row, col) == self.point_a:
                self.point_a = None
                removed = True
                print("🗑️  Removed Point A")
            if (row, col) == self.point_b:
                self.point_b = None
                removed = True
                print("🗑️  Removed Point B")
            if (row, col) in self.waypoints:
                self.waypoints.remove((row, col))
                removed = True
                print(f"🗑️  Removed waypoint at ({row}, {col})")
            if not removed:
                self.terrain_grid[row, col] = 'normal'
                print(f"🗑️  Reset terrain at ({row}, {col})")
        
        if self.point_a and self.point_b:
            self.find_path()
        else:
            self.path = []
            
        self.draw_grid()
    
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_movement_cost(self, pos):
        terrain = self.terrain_grid[pos[0], pos[1]]
        return self.terrain_types[terrain]['cost']
    
    def get_neighbors(self, position):
        neighbors = []
        row, col = position
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                if (new_row, new_col) not in self.obstacles:
                    neighbors.append((new_row, new_col))
        return neighbors
    
    def a_star_search(self, start, end):
        """A* algorithm with terrain costs"""
        open_set = [(0, start)]
        heapq.heapify(open_set)
        came_from = {}
        g_score = {start: 0}
        open_set_hash = {start}
        
        self.explored_nodes = set()
        self.frontier_nodes = {start}
        self.stats['nodes_explored'] = 0
        self.stats['max_frontier_size'] = 1
        start_time = time.time()
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            self.stats['nodes_explored'] += 1
            self.explored_nodes.add(current)
            self.frontier_nodes = open_set_hash.copy()
            
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                self.stats['computation_time'] = (time.time() - start_time) * 1000
                self.stats['path_length'] = len(path) - 1
                self.stats['path_cost'] = g_score[end]
                return path
            
            for neighbor in self.get_neighbors(current):
                movement_cost = self.get_movement_cost(neighbor)
                tentative_g = g_score[current] + movement_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, end)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f, neighbor))
                        open_set_hash.add(neighbor)
                        self.stats['max_frontier_size'] = max(
                            self.stats['max_frontier_size'], len(open_set_hash))
        return []
    
    def dijkstra_search(self, start, end):
        """Dijkstra's algorithm"""
        open_set = [(0, start)]
        heapq.heapify(open_set)
        came_from = {}
        g_score = {start: 0}
        open_set_hash = {start}
        
        self.explored_nodes = set()
        self.frontier_nodes = {start}
        self.stats['nodes_explored'] = 0
        self.stats['max_frontier_size'] = 1
        start_time = time.time()
        
        while open_set:
            current_g, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            self.stats['nodes_explored'] += 1
            self.explored_nodes.add(current)
            self.frontier_nodes = open_set_hash.copy()
            
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                self.stats['computation_time'] = (time.time() - start_time) * 1000
                self.stats['path_length'] = len(path) - 1
                self.stats['path_cost'] = g_score[end]
                return path
            
            for neighbor in self.get_neighbors(current):
                movement_cost = self.get_movement_cost(neighbor)
                tentative_g = g_score[current] + movement_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (tentative_g, neighbor))
                        open_set_hash.add(neighbor)
                        self.stats['max_frontier_size'] = max(
                            self.stats['max_frontier_size'], len(open_set_hash))
        return []
    
    def bfs_search(self, start, end):
        """Breadth-First Search"""
        queue = deque([start])
        came_from = {}
        visited = {start}
        
        self.explored_nodes = set()
        self.frontier_nodes = set(queue)
        self.stats['nodes_explored'] = 0
        self.stats['max_frontier_size'] = 1
        start_time = time.time()
        
        while queue:
            current = queue.popleft()
            self.stats['nodes_explored'] += 1
            self.explored_nodes.add(current)
            
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                self.stats['computation_time'] = (time.time() - start_time) * 1000
                self.stats['path_length'] = len(path) - 1
                cost = sum(self.get_movement_cost(p) for p in path[1:])
                self.stats['path_cost'] = cost
                return path
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)
                    self.frontier_nodes = set(queue)
                    self.stats['max_frontier_size'] = max(
                        self.stats['max_frontier_size'], len(queue))
        return []
    
    def greedy_bfs_search(self, start, end):
        """Greedy Best-First Search"""
        open_set = [(self.heuristic(start, end), start)]
        heapq.heapify(open_set)
        came_from = {}
        visited = {start}
        
        self.explored_nodes = set()
        self.frontier_nodes = {start}
        self.stats['nodes_explored'] = 0
        self.stats['max_frontier_size'] = 1
        start_time = time.time()
        
        while open_set:
            current_h, current = heapq.heappop(open_set)
            self.stats['nodes_explored'] += 1
            self.explored_nodes.add(current)
            
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                self.stats['computation_time'] = (time.time() - start_time) * 1000
                self.stats['path_length'] = len(path) - 1
                cost = sum(self.get_movement_cost(p) for p in path[1:])
                self.stats['path_cost'] = cost
                return path
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (self.heuristic(neighbor, end), neighbor))
                    self.frontier_nodes = {n for _, n in open_set}
                    self.stats['max_frontier_size'] = max(
                        self.stats['max_frontier_size'], len(open_set))
        return []
    
    def bidirectional_a_star(self, start, end):
        """Bidirectional A* search"""
        forward_open = [(0, start)]
        forward_g = {start: 0}
        forward_came_from = {}
        forward_visited = {start}
        
        backward_open = [(0, end)]
        backward_g = {end: 0}
        backward_came_from = {}
        backward_visited = {end}
        
        self.explored_nodes = set()
        self.stats['nodes_explored'] = 0
        self.stats['max_frontier_size'] = 2
        start_time = time.time()
        
        while forward_open and backward_open:
            # Expand forward
            if forward_open:
                _, current = heapq.heappop(forward_open)
                self.stats['nodes_explored'] += 1
                self.explored_nodes.add(current)
                
                if current in backward_visited:
                    path_forward = []
                    temp = current
                    while temp in forward_came_from:
                        path_forward.append(temp)
                        temp = forward_came_from[temp]
                    path_forward.append(start)
                    path_forward.reverse()
                    
                    path_backward = []
                    temp = current
                    while temp in backward_came_from:
                        temp = backward_came_from[temp]
                        path_backward.append(temp)
                    
                    full_path = path_forward + path_backward
                    self.stats['computation_time'] = (time.time() - start_time) * 1000
                    self.stats['path_length'] = len(full_path) - 1
                    cost = sum(self.get_movement_cost(p) for p in full_path[1:])
                    self.stats['path_cost'] = cost
                    return full_path
                
                for neighbor in self.get_neighbors(current):
                    movement_cost = self.get_movement_cost(neighbor)
                    tentative_g = forward_g[current] + movement_cost
                    
                    if neighbor not in forward_g or tentative_g < forward_g[neighbor]:
                        forward_came_from[neighbor] = current
                        forward_g[neighbor] = tentative_g
                        f = tentative_g + self.heuristic(neighbor, end)
                        heapq.heappush(forward_open, (f, neighbor))
                        forward_visited.add(neighbor)
            
            # Expand backward
            if backward_open:
                _, current = heapq.heappop(backward_open)
                self.stats['nodes_explored'] += 1
                self.explored_nodes.add(current)
                
                if current in forward_visited:
                    path_forward = []
                    temp = current
                    while temp in forward_came_from:
                        path_forward.append(temp)
                        temp = forward_came_from[temp]
                    path_forward.append(start)
                    path_forward.reverse()
                    
                    path_backward = []
                    temp = current
                    while temp in backward_came_from:
                        temp = backward_came_from[temp]
                        path_backward.append(temp)
                    
                    full_path = path_forward + path_backward
                    self.stats['computation_time'] = (time.time() - start_time) * 1000
                    self.stats['path_length'] = len(full_path) - 1
                    cost = sum(self.get_movement_cost(p) for p in full_path[1:])
                    self.stats['path_cost'] = cost
                    return full_path
                
                for neighbor in self.get_neighbors(current):
                    movement_cost = self.get_movement_cost(neighbor)
                    tentative_g = backward_g[current] + movement_cost
                    
                    if neighbor not in backward_g or tentative_g < backward_g[neighbor]:
                        backward_came_from[neighbor] = current
                        backward_g[neighbor] = tentative_g
                        f = tentative_g + self.heuristic(neighbor, start)
                        heapq.heappush(backward_open, (f, neighbor))
                        backward_visited.add(neighbor)
            
            self.stats['max_frontier_size'] = max(
                self.stats['max_frontier_size'],
                len(forward_open) + len(backward_open))
        return []
    
    def find_path(self):
        """Find path using current algorithm with waypoint support"""
        if not self.point_a or not self.point_b:
            return
        
        print(f"\n🔍 Finding path using {self.current_algorithm}...")
        
        if self.waypoints:
            full_path = []
            points = [self.point_a] + self.waypoints + [self.point_b]
            
            for i in range(len(points) - 1):
                algorithm_func = self.available_algorithms[self.current_algorithm]
                segment_path = algorithm_func(points[i], points[i+1])
                
                if not segment_path:
                    print(f"❌ No path between waypoint {i} and {i+1}!")
                    self.path = []
                    return
                
                if full_path:
                    full_path.extend(segment_path[1:])
                else:
                    full_path.extend(segment_path)
            
            self.path = full_path
        else:
            algorithm_func = self.available_algorithms[self.current_algorithm]
            self.path = algorithm_func(self.point_a, self.point_b)
        
        self.calculate_path_metrics()
        self.print_path_stats()
        self.draw_grid()
    
    def calculate_path_metrics(self):
        """Calculate additional path metrics"""
        if not self.path or len(self.path) < 2:
            self.stats['turns'] = 0
            return
        
        turns = 0
        for i in range(1, len(self.path) - 1):
            prev_dir = (self.path[i][0] - self.path[i-1][0], 
                       self.path[i][1] - self.path[i-1][1])
            next_dir = (self.path[i+1][0] - self.path[i][0], 
                       self.path[i+1][1] - self.path[i][1])
            if prev_dir != next_dir:
                turns += 1
        
        self.stats['turns'] = turns
    
    def manual_find_path(self):
        """Manually trigger path finding"""
        if self.point_a and self.point_b:
            self.find_path()
        else:
            print("\n⚠️  Please set both Point A and Point B first!")
    
    def add_random_obstacles(self, density=0.2):
        """Add random obstacles"""
        available = [(i, j) for i in range(self.rows) 
                            for j in range(self.cols)
                            if (i, j) != self.point_a 
                            and (i, j) != self.point_b
                            and (i, j) not in self.obstacles
                            and (i, j) not in self.waypoints]
        
        count = int(len(available) * density)
        if count > 0:
            selected = random.sample(available, min(count, len(available)))
            for pos in selected:
                self.obstacles.add(pos)
                self.grid[pos[0], pos[1]] = 1
            
            print(f"🎲 Added {len(selected)} random obstacles")
            
            if self.point_a and self.point_b:
                self.find_path()
            self.draw_grid()
    
    def save_scenario(self, filename=None):
        """Save current scenario to JSON"""
        if filename is None:
            filename = f"scenario_{self.rows}x{self.cols}.json"
        
        scenario = {
            'rows': self.rows,
            'cols': self.cols,
            'point_a': list(self.point_a) if self.point_a else None,
            'point_b': list(self.point_b) if self.point_b else None,
            'waypoints': [list(wp) for wp in self.waypoints],
            'obstacles': [list(obs) for obs in self.obstacles],
            'terrain': self.terrain_grid.tolist(),
            'algorithm': self.current_algorithm
        }
        
        with open(filename, 'w') as f:
            json.dump(scenario, f, indent=2)
        
        print(f"\n💾 Scenario saved to {filename}")
    
    def load_scenario(self, filename=None):
        """Load scenario from JSON"""
        if filename is None:
            scenarios = [f for f in os.listdir('.') 
                        if f.startswith('scenario_') and f.endswith('.json')]
            if not scenarios:
                print("❌ No saved scenarios found!")
                return
            filename = scenarios[-1]
        
        try:
            with open(filename, 'r') as f:
                scenario = json.load(f)
            
            self.rows = scenario['rows']
            self.cols = scenario['cols']
            self.grid = np.zeros((self.rows, self.cols), dtype=int)
            self.obstacles = set()
            self.waypoints = []
            self.terrain_grid = np.full((self.rows, self.cols), 'normal', dtype=object)
            
            for obs in scenario['obstacles']:
                self.obstacles.add(tuple(obs))
                self.grid[obs[0], obs[1]] = 1
            
            if scenario['point_a']:
                self.point_a = tuple(scenario['point_a'])
            if scenario['point_b']:
                self.point_b = tuple(scenario['point_b'])
            
            if 'waypoints' in scenario:
                self.waypoints = [tuple(wp) for wp in scenario['waypoints']]
            
            if 'terrain' in scenario:
                self.terrain_grid = np.array(scenario['terrain'], dtype=object)
            
            if 'algorithm' in scenario:
                self.switch_algorithm(scenario['algorithm'])
            
            print(f"\n📂 Scenario loaded from {filename}")
            
            if self.point_a and self.point_b:
                self.find_path()
            self.draw_grid()
            
        except Exception as e:
            print(f"❌ Error loading scenario: {e}")
    
    def print_path_stats(self):
        """Print path statistics"""
        if self.path:
            print("\n" + "="*50)
            print(f"✅ PATH FOUND ({self.current_algorithm})")
            print("="*50)
            print(f"📍 A: {self.point_a} → B: {self.point_b}")
            if self.waypoints:
                print(f"📍 Waypoints: {self.waypoints}")
            print(f"📏 Length: {len(self.path)-1} steps")
            print(f"💰 Cost: {self.stats.get('path_cost', 0):.1f}")
            print(f"🔍 Explored: {self.stats.get('nodes_explored', 0)} nodes")
            print(f"⏱️  Time: {self.stats.get('computation_time', 0):.2f} ms")
            print(f"🔄 Turns: {self.stats.get('turns', 0)}")
            print("="*50 + "\n")
    
    def draw_grid(self):
        """Draw the complete grid with all panels"""
        # Clear grid axes
        self.ax_grid.clear()
        self.ax_grid.set_facecolor('#ECF0F1')
        
        # Draw grid cells
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) in self.obstacles:
                    color = '#2C3E50'
                    alpha = 0.9
                else:
                    terrain = self.terrain_grid[i, j]
                    color = self.terrain_types[terrain]['color']
                    alpha = 0.6
                    
                rect = Rectangle((j, self.rows - i - 1), 1, 1,
                               facecolor=color, edgecolor='#BDC3C7',
                               alpha=alpha, linewidth=0.5)
                self.ax_grid.add_patch(rect)
        
        # Draw explored nodes
        for node in self.explored_nodes:
            if node not in self.obstacles and node != self.point_a and node != self.point_b:
                rect = Rectangle((node[1] + 0.15, self.rows - node[0] - 0.85), 
                               0.7, 0.7, facecolor='#F1C40F', 
                               edgecolor='none', alpha=0.3)
                self.ax_grid.add_patch(rect)
        
        # Draw frontier nodes
        for node in self.frontier_nodes:
            if node not in self.obstacles and node != self.point_a and node != self.point_b:
                rect = Rectangle((node[1] + 0.15, self.rows - node[0] - 0.85),
                               0.7, 0.7, facecolor='#E67E22',
                               edgecolor='none', alpha=0.5)
                self.ax_grid.add_patch(rect)
        
        # Draw path
        if self.path:
            path_visual = [(p[1] + 0.5, self.rows - p[0] - 0.5) for p in self.path]
            xs, ys = zip(*path_visual)
            
            # Path shadow effect
            self.ax_grid.plot(xs, ys, 'k-', linewidth=4, alpha=0.3, zorder=3)
            # Main path
            self.ax_grid.plot(xs, ys, '#3498DB', linewidth=3, 
                            alpha=0.9, zorder=4, solid_capstyle='round')
            # Path nodes
            self.ax_grid.plot(xs, ys, 'o', color='#2980B9', 
                            markersize=5, zorder=4)
        
        # Draw points with glow effect
        if self.point_a:
            self.ax_grid.plot(self.point_a[1] + 0.5, self.rows - self.point_a[0] - 0.5,
                           'o', color='#27AE60', markersize=16, alpha=0.3, zorder=5)
            self.ax_grid.plot(self.point_a[1] + 0.5, self.rows - self.point_a[0] - 0.5,
                           'o', color='#2ECC71', markersize=12, 
                           markeredgecolor='#27AE60', markeredgewidth=2, zorder=6)
            self.ax_grid.text(self.point_a[1] + 0.5, self.rows - self.point_a[0] - 0.2,
                           'A', ha='center', va='bottom', fontweight='bold',
                           fontsize=12, color='#27AE60', zorder=7,
                           path_effects=[patheffects.withStroke(linewidth=2, foreground='white')])
        
        if self.point_b:
            self.ax_grid.plot(self.point_b[1] + 0.5, self.rows - self.point_b[0] - 0.5,
                           'o', color='#C0392B', markersize=16, alpha=0.3, zorder=5)
            self.ax_grid.plot(self.point_b[1] + 0.5, self.rows - self.point_b[0] - 0.5,
                           'o', color='#E74C3C', markersize=12,
                           markeredgecolor='#C0392B', markeredgewidth=2, zorder=6)
            self.ax_grid.text(self.point_b[1] + 0.5, self.rows - self.point_b[0] - 0.2,
                           'B', ha='center', va='bottom', fontweight='bold',
                           fontsize=12, color='#C0392B', zorder=7,
                           path_effects=[patheffects.withStroke(linewidth=2, foreground='white')])
        
        # Draw waypoints
        for idx, wp in enumerate(self.waypoints):
            self.ax_grid.plot(wp[1] + 0.5, self.rows - wp[0] - 0.5,
                           's', color='#F39C12', markersize=10,
                           markeredgecolor='#E67E22', markeredgewidth=2, zorder=6)
            self.ax_grid.text(wp[1] + 0.5, self.rows - wp[0] - 0.2,
                           f'W{idx+1}', ha='center', va='bottom', fontweight='bold',
                           fontsize=9, color='#E67E22', zorder=7,
                           path_effects=[patheffects.withStroke(linewidth=1.5, foreground='white')])
        
        # Grid settings
        self.ax_grid.set_xlim(0, self.cols)
        self.ax_grid.set_ylim(0, self.rows)
        self.ax_grid.set_xticks(range(self.cols))
        self.ax_grid.set_yticks(range(self.rows))
        self.ax_grid.grid(True, alpha=0.2, color='#7F8C8D')
        self.ax_grid.set_title(f'{self.rows}×{self.cols} Interactive Grid Pathfinder',
                              fontsize=13, fontweight='bold', color='#2C3E50', pad=10)
        
        # Info text on grid
        info = f'Obstacles: {len(self.obstacles)}'
        if self.waypoints:
            info += f' | Waypoints: {len(self.waypoints)}'
        if self.path:
            info += f' | Path: {len(self.path)-1} steps | Cost: {self.stats.get("path_cost", 0):.1f}'
        self.ax_grid.text(0.5, -0.06, info, transform=self.ax_grid.transAxes,
                        ha='center', fontsize=9, color='#7F8C8D', style='italic')
        
        # Update statistics panel (drawn as text, not interactive)
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        if self.path:
            stats_data = [
                ('Algorithm', self.current_algorithm),
                ('Path Length', f"{len(self.path)-1} steps"),
                ('Path Cost', f"{self.stats.get('path_cost', 0):.1f}"),
                ('Nodes Explored', str(self.stats.get('nodes_explored', 0))),
                ('Time', f"{self.stats.get('computation_time', 0):.2f} ms"),
                ('Turns', str(self.stats.get('turns', 0))),
                ('Max Frontier', str(self.stats.get('max_frontier_size', 0))),
            ]
        else:
            stats_data = [('Status', 'Waiting for path...')]
        
        for i, (label, value) in enumerate(stats_data):
            y = 0.85 - i * 0.11
            self.ax_stats.text(0.1, y, f"{label}:", transform=self.ax_stats.transAxes,
                           fontsize=9, color='#BDC3C7', fontweight='bold')
            self.ax_stats.text(0.95, y, value, transform=self.ax_stats.transAxes,
                           ha='right', fontsize=9, color='#ECF0F1')
        
        # Update terrain legend (drawn as text, not interactive)
        self.ax_terrain.clear()
        self.ax_terrain.axis('off')
        
        for i, (key, info) in enumerate(self.terrain_types.items()):
            y = 0.85 - i * 0.15
            rect = FancyBboxPatch((0.1, y - 0.04), 0.08, 0.08,
                                boxstyle="round,pad=0.01",
                                facecolor=info['color'], edgecolor='#BDC3C7',
                                linewidth=1)
            self.ax_terrain.add_patch(rect)
            self.ax_terrain.text(0.25, y, info['name'], transform=self.ax_terrain.transAxes,
                             fontsize=9, color='#ECF0F1', fontweight='bold')
            cost_text = f"{info['cost']}×"
            self.ax_terrain.text(0.9, y, cost_text, transform=self.ax_terrain.transAxes,
                             ha='right', fontsize=9, color='#BDC3C7')
        
        # Keyboard shortcuts hint
        self.ax_terrain.text(0.5, 0.08, "⌨️ Shortcuts: A B W O R F G C S L 1-5",
                          transform=self.ax_terrain.transAxes,
                          ha='center', fontsize=7, color='#7F8C8D')
        
        self.fig.canvas.draw_idle()
    
    def clear_all(self):
        """Clear everything"""
        self.point_a = None
        self.point_b = None
        self.waypoints.clear()
        self.obstacles.clear()
        self.path = []
        self.explored_nodes.clear()
        self.frontier_nodes.clear()
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.terrain_grid = np.full((self.rows, self.cols), 'normal', dtype=object)
        self.stats = {
            'nodes_explored': 0, 'path_length': 0,
            'computation_time': 0, 'path_cost': 0,
            'turns': 0, 'max_frontier_size': 0
        }
        print("\n🗑️  Everything cleared!")
        self.draw_grid()
    
    def show(self):
        """Display the interface"""
        print("\n" + "="*50)
        print(f"🚀 ULTIMATE {self.rows}×{self.cols} PATHFINDER")
        print("="*50)
        print("🖱️  Click grid to place points")
        print("🖱️  Click algorithm buttons on the right to switch")
        print("⌨️  Or use keyboard shortcuts")
        print("="*50 + "\n")
        plt.show()


def get_grid_size():
    """Ask user for grid dimensions"""
    print("\n" + "="*50)
    print("🎯 GRID PATHFINDER SETUP")
    print("="*50)
    
    while True:
        try:
            choice = input("\nUse preset? (y/n) [Enter=10×10]: ").strip().lower()
            
            if choice == "" or choice == "y":
                print("\n1. Small (5×5)")
                print("2. Standard (10×10)")
                print("3. Medium (15×15)")
                print("4. Large (20×20)")
                print("5. Huge (30×30)")
                
                size = input("\nChoice (1-5) [Enter=10×10]: ").strip()
                sizes = {'1': (5,5), '2': (10,10), '3': (15,15), 
                        '4': (20,20), '5': (30,30)}
                return sizes.get(size, (10, 10))
            else:
                rows = int(input("Rows (2-50): "))
                cols = int(input("Columns (2-50): "))
                if rows < 2 or cols < 2:
                    print("❌ Minimum 2×2. Using 10×10.")
                    return 10, 10
                return rows, cols
        except ValueError:
            print("❌ Invalid input. Using 10×10.")
            return 10, 10
        except KeyboardInterrupt:
            print("\nUsing 10×10 default.")
            return 10, 10


if __name__ == "__main__":
    try:
        rows, cols = get_grid_size()
        print(f"\n🚀 Creating {rows}×{cols} Pathfinder...")
        print("💡 Click ALGORITHM buttons on the right panel to switch algorithms!")
        pathfinder = UltimateGridPathfinder(rows, cols)
        pathfinder.show()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        pathfinder = UltimateGridPathfinder(10, 10)
        pathfinder.show()