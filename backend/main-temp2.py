from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3
from typing import Optional, List, Dict

from langchain_aws import ChatBedrock
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64
import math

app = FastAPI(title="Unified Drawing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global canvas state
canvas = None
draw = None

# ============================================================================
# CORE CANVAS TOOLS
# ============================================================================

@tool(description="Create a blank canvas. ALWAYS call this first.")
def create_canvas(width: int = 800, height: int = 600, background_color: str = "white") -> str:
    global canvas, draw
    canvas = Image.new("RGBA", (width, height), background_color)
    draw = ImageDraw.Draw(canvas)
    return f"Canvas created: {width}x{height}, background={background_color}"

# ============================================================================
# BASIC SHAPE TOOLS (for creative drawings)
# ============================================================================

@tool(description="Draw a filled rectangle. Specify position (x,y), size (width, height), fill color, and outline color.")
def add_rectangle(x: int = 0, y: int = 0, width: int = 100, height: int = 100, fill: str = "blue", outline: str = "black", outline_width: int = 1) -> str:
    if canvas is None or draw is None:
        return "ERROR: Canvas not created"
    draw.rectangle([x, y, x + width, y + height], fill=fill, outline=outline, width=outline_width)
    return f"Rectangle drawn at ({x},{y}) with size {width}x{height}, fill={fill}"

@tool(description="Draw a filled circle. Specify center position (x,y), radius, fill color, and outline color.")
def add_circle(x: int, y: int, radius: int, fill: str = "red", outline: str = "black", outline_width: int = 1) -> str:
    if canvas is None or draw is None:
        return "ERROR: Canvas not created"
    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bbox, fill=fill, outline=outline, width=outline_width)
    return f"Circle drawn at ({x},{y}) with radius {radius}, fill={fill}"

@tool(description="Draw an ellipse (oval). Specify center (x,y), horizontal radius (rx), vertical radius (ry).")
def add_ellipse(x: int, y: int, rx: int, ry: int, fill: str = "blue", outline: str = "black", outline_width: int = 1) -> str:
    if canvas is None or draw is None:
        return "ERROR: Canvas not created"
    bbox = [x - rx, y - ry, x + rx, y + ry]
    draw.ellipse(bbox, fill=fill, outline=outline, width=outline_width)
    return f"Ellipse drawn at ({x},{y}) with radii ({rx},{ry}), fill={fill}"

@tool(description="Draw a polygon from a list of (x,y) points.")
def add_polygon(points: list, fill: str = "green", outline: str = "black", outline_width: int = 1) -> str:
    if canvas is None or draw is None:
        return "ERROR: Canvas not created"
    
    normalized_points = []
    for p in points:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            normalized_points.append(tuple(p))
        elif isinstance(p, dict) and "x" in p and "y" in p:
            normalized_points.append((p["x"], p["y"]))
    
    if len(normalized_points) < 3:
        return f"Need at least 3 points for polygon, got {len(normalized_points)}"
    
    draw.polygon(normalized_points, fill=fill, outline=outline, width=outline_width)
    return f"Polygon drawn with {len(normalized_points)} points, fill={fill}"

@tool(description="Draw a line from point (x1,y1) to (x2,y2).")
def add_line(x1: int, y1: int, x2: int, y2: int, fill: str = "black", width: int = 2) -> str:
    if canvas is None or draw is None:
        return "ERROR: Canvas not created"
    draw.line([(x1, y1), (x2, y2)], fill=fill, width=width)
    return f"Line drawn from ({x1},{y1}) to ({x2},{y2}), color={fill}, width={width}"

@tool(description="Draw a star shape with N points.")
def add_star(x: int, y: int, outer_radius: int, inner_radius: int, points: int = 5, fill: str = "yellow", outline: str = "black") -> str:
    if canvas is None or draw is None:
        return "ERROR: Canvas not created"
    
    star_points = []
    angle_step = math.pi / points
    
    for i in range(points * 2):
        angle = i * angle_step - math.pi / 2
        r = outer_radius if i % 2 == 0 else inner_radius
        px = x + r * math.cos(angle)
        py = y + r * math.sin(angle)
        star_points.append((px, py))
    
    draw.polygon(star_points, fill=fill, outline=outline)
    return f"Star with {points} points drawn at ({x},{y}), fill={fill}"

@tool(description="Draw a regular polygon (equal sides) with N sides. N=3 is triangle, N=6 is hexagon.")
def add_regular_polygon(x: int, y: int, radius: int, sides: int, rotation: float = 0, fill: str = "blue", outline: str = "black") -> str:
    if canvas is None or draw is None:
        return "ERROR: Canvas not created"
    
    points = []
    angle_step = 2 * math.pi / sides
    start_angle = rotation * math.pi / 180
    
    for i in range(sides):
        angle = i * angle_step + start_angle - math.pi / 2
        px = x + radius * math.cos(angle)
        py = y + radius * math.sin(angle)
        points.append((px, py))
    
    draw.polygon(points, fill=fill, outline=outline)
    return f"Regular {sides}-sided polygon drawn at ({x},{y}), fill={fill}"

@tool(description="Create a gradient background (vertical or horizontal).")
def add_gradient(start_color: str, end_color: str, direction: str = "vertical") -> str:
    if canvas is None:
        return "ERROR: Canvas not created"
    
    width, height = canvas.size
    
    def parse_color(color_str):
        if color_str.startswith('#'):
            return tuple(int(color_str[i:i+2], 16) for i in (1, 3, 5))
        color_map = {
            'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255),
            'yellow': (255, 255, 0), 'orange': (255, 165, 0), 'purple': (128, 0, 128),
            'white': (255, 255, 255), 'black': (0, 0, 0), 'gray': (128, 128, 128),
            'cyan': (0, 255, 255), 'pink': (255, 192, 203), 'brown': (165, 42, 42)
        }
        return color_map.get(color_str.lower(), (255, 255, 255))
    
    start_rgb = parse_color(start_color)
    end_rgb = parse_color(end_color)
    
    if direction == "vertical":
        for i in range(height):
            ratio = i / height
            r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
            g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
            b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)
            draw.line([(0, i), (width, i)], fill=(r, g, b))
    else:
        for i in range(width):
            ratio = i / width
            r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
            g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
            b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)
            draw.line([(i, 0), (i, height)], fill=(r, g, b))
    
    return f"Gradient from {start_color} to {end_color} ({direction}) applied"

@tool(description="Add text at position (x,y).")
def add_text(x: int, y: int, text: str, font_size: int = 20, fill: str = "black") -> str:
    if canvas is None or draw is None:
        return "ERROR: Canvas not created"
    
    font = ImageFont.load_default()
    draw.text((x, y), text, fill=fill, font=font)
    return f'Text "{text}" drawn at ({x},{y})'

# ============================================================================
# ARCHITECTURE DIAGRAM TOOLS
# ============================================================================

@tool(description="Draw a component box with text. Returns connection points. Text wraps automatically.")
def draw_component(x: int, y: int, width: int, height: int, text: str, color: str = "lightblue", component_type: str = "service") -> str:
    """
    component_type options:
    - service: blue (API, microservice, backend)
    - database: green (SQL, NoSQL databases)
    - client: purple (frontend, mobile app, user)
    - external: orange (third-party services, external APIs)
    - queue: yellow (message queues, event streams)
    - cache: pink (Redis, Memcached)
    - storage: brown (file storage, S3)
    """
    if canvas is None or draw is None:
        return "ERROR: Canvas not created"
    
    color_map = {
        "service": ("#E3F2FD", "#1976D2"),
        "database": ("#E8F5E9", "#388E3C"),
        "client": ("#F3E5F5", "#7B1FA2"),
        "external": ("#FFF3E0", "#E65100"),
        "queue": ("#FFFDE7", "#F57F17"),
        "cache": ("#FCE4EC", "#C2185B"),
        "storage": ("#EFEBE9", "#5D4037")
    }
    
    fill_color, outline_color = color_map.get(component_type, ("#E3F2FD", "#1976D2"))
    
    draw.rectangle([x, y, x + width, y + height], fill=fill_color, outline=outline_color, width=3)
    
    # Text wrapping
    font = ImageFont.load_default()
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        test_width = bbox[2] - bbox[0]
        
        if test_width <= width - 20:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    max_lines = max(1, (height - 20) // 15)
    lines = lines[:max_lines]
    
    total_text_height = len(lines) * 15
    start_y = y + (height - total_text_height) // 2
    
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x + (width - text_width) // 2
        text_y = start_y + i * 15
        draw.text((text_x, text_y), line, fill="black", font=font)
    
    center_x = x + width // 2
    center_y = y + height // 2
    top = (center_x, y)
    bottom = (center_x, y + height)
    left = (x, center_y)
    right = (x + width, center_y)
    
    return f"Component '{text}' at ({x},{y}). Connections: top={top}, bottom={bottom}, left={left}, right={right}"

@tool(description="Draw an arrow between components. Use for data flow, API calls.")
def draw_connection(x1: int, y1: int, x2: int, y2: int, label: str = "", style: str = "solid", bidirectional: bool = False) -> str:
    """
    style options: solid, dashed
    bidirectional: True for two-way communication
    """
    if canvas is None or draw is None:
        return "ERROR: Canvas not created"
    
    if style == "dashed":
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx*dx + dy*dy)
        dash_length = 10
        gap_length = 5
        num_dashes = int(distance / (dash_length + gap_length))
        
        for i in range(num_dashes):
            start_ratio = i * (dash_length + gap_length) / distance
            end_ratio = (i * (dash_length + gap_length) + dash_length) / distance
            dash_x1 = x1 + dx * start_ratio
            dash_y1 = y1 + dy * start_ratio
            dash_x2 = x1 + dx * end_ratio
            dash_y2 = y1 + dy * end_ratio
            draw.line([(dash_x1, dash_y1), (dash_x2, dash_y2)], fill="black", width=2)
    else:
        draw.line([(x1, y1), (x2, y2)], fill="black", width=2)
    
    angle = math.atan2(y2 - y1, x2 - x1)
    arrow_len = 12
    arrow_angle = math.pi / 6
    
    p1 = (x2 - arrow_len * math.cos(angle - arrow_angle), y2 - arrow_len * math.sin(angle - arrow_angle))
    p2 = (x2 - arrow_len * math.cos(angle + arrow_angle), y2 - arrow_len * math.sin(angle + arrow_angle))
    draw.polygon([p1, (x2, y2), p2], fill="black")
    
    if bidirectional:
        angle_back = angle + math.pi
        p1_back = (x1 - arrow_len * math.cos(angle_back - arrow_angle), y1 - arrow_len * math.sin(angle_back - arrow_angle))
        p2_back = (x1 - arrow_len * math.cos(angle_back + arrow_angle), y1 - arrow_len * math.sin(angle_back + arrow_angle))
        draw.polygon([p1_back, (x1, y1), p2_back], fill="black")
    
    if label:
        font = ImageFont.load_default()
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        bbox = draw.textbbox((mid_x, mid_y), label, font=font)
        padding = 3
        draw.rectangle([bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding], fill="white", outline="white")
        draw.text((mid_x, mid_y), label, fill="black", font=font)
    
    return f"Connection drawn from ({x1},{y1}) to ({x2},{y2})" + (f" [{label}]" if label else "")

@tool(description="Draw a cylinder shape for databases.")
def draw_database(x: int, y: int, width: int, height: int, text: str) -> str:
    if canvas is None or draw is None:
        return "ERROR: Canvas not created"
    
    ellipse_height = height // 5
    
    draw.ellipse([x, y, x + width, y + ellipse_height], fill="#E8F5E9", outline="#388E3C", width=3)
    draw.rectangle([x, y + ellipse_height//2, x + width, y + height - ellipse_height//2], fill="#E8F5E9", outline="#388E3C", width=3)
    draw.ellipse([x, y + height - ellipse_height, x + width, y + height], fill="#E8F5E9", outline="#388E3C", width=3)
    
    font = ImageFont.load_default()
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        test_width = bbox[2] - bbox[0]
        
        if test_width <= width - 20:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    max_lines = max(1, (height - ellipse_height * 2 - 20) // 15)
    lines = lines[:max_lines]
    
    total_text_height = len(lines) * 15
    start_y = y + (height - total_text_height) // 2
    
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x + (width - text_width) // 2
        text_y = start_y + i * 15
        draw.text((text_x, text_y), line, fill="black", font=font)
    
    center_x = x + width // 2
    top = (center_x, y)
    bottom = (center_x, y + height)
    
    return f"Database '{text}' at ({x},{y}). Connections: top={top}, bottom={bottom}"

@tool(description="Draw a cloud shape for cloud services.")
def draw_cloud(x: int, y: int, width: int, height: int, text: str) -> str:
    if canvas is None or draw is None:
        return "ERROR: Canvas not created"
    
    r1 = height // 3
    r2 = height // 2
    r3 = height // 3
    
    draw.ellipse([x, y + r1, x + r1*2, y + r1 + r1*2], fill="#FFF3E0", outline="#E65100", width=3)
    draw.ellipse([x + width//3, y, x + width//3 + r2*2, y + r2*2], fill="#FFF3E0", outline="#E65100", width=3)
    draw.ellipse([x + width - r3*2, y + r3, x + width, y + r3 + r3*2], fill="#FFF3E0", outline="#E65100", width=3)
    draw.rectangle([x + r1, y + height - r1, x + width - r3, y + height], fill="#FFF3E0", outline="#E65100", width=3)
    
    font = ImageFont.load_default()
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        test_width = bbox[2] - bbox[0]
        
        if test_width <= width - 40:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    max_lines = max(1, (height - 30) // 15)
    lines = lines[:max_lines]
    
    total_text_height = len(lines) * 15
    start_y = y + (height - total_text_height) // 2
    
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x + (width - text_width) // 2
        text_y = start_y + i * 15
        draw.text((text_x, text_y), line, fill="black", font=font)
    
    center_x = x + width // 2
    center_y = y + height // 2
    top = (center_x, y)
    bottom = (center_x, y + height)
    left = (x, center_y)
    right = (x + width, center_y)
    
    return f"Cloud '{text}' at ({x},{y}). Connections: top={top}, bottom={bottom}, left={left}, right={right}"

@tool(description="Add a text label for architecture diagrams.")
def add_label(x: int, y: int, text: str, size: str = "normal") -> str:
    if canvas is None or draw is None:
        return "ERROR: Canvas not created"
    
    font = ImageFont.load_default()
    fill_color = "black" if size == "title" else "#424242" if size == "subtitle" else "black"
    
    draw.text((x, y), text, fill=fill_color, font=font)
    return f"Label '{text}' added at ({x},{y})"

# ============================================================================
# FLOWCHART TOOLS
# ============================================================================

@tool(description="Plan flowchart layout. For flowcharts only.")
def plan_flowchart(num_main_elements: int, has_decision: bool = False, has_branches: bool = False) -> str:
    """Plans flowchart layout - use only for flowcharts."""
    canvas_width = 700
    center_x = canvas_width // 2
    y_pos = 50
    spacing = 100
    
    result = "=== FLOWCHART LAYOUT GUIDE ===\n\n"
    result += f"Canvas: 700x800, Center X: {center_x}\n\n"
    result += "**CRITICAL: Follow this EXACT sequence and coordinates!**\n\n"
    
    current_y = y_pos
    step = 1
    
    # Start terminal
    result += f"STEP {step}: draw_terminal({center_x-70}, {current_y}, 140, 50, 'Start')\n"
    start_bottom_y = current_y + 50
    current_y += 50 + spacing
    step += 1
    
    # First element (if exists)
    first_top_y = current_y
    first_bottom_y = current_y + 50
    if num_main_elements >= 2:
        result += f"STEP {step}: draw_process({center_x-75}, {current_y}, 150, 50, 'Input data')\n"
        result += f"// This box: TOP at y={current_y}, BOTTOM at y={current_y + 50}\n\n"
        current_y += 50 + spacing
        step += 1
    else:
        first_bottom_y = start_bottom_y
    
    if has_decision:
        decision_y = current_y
        decision_top_y = decision_y
        decision_center_y = decision_y + 45  # Middle of 90px tall diamond
        decision_bottom_y = decision_y + 90
        decision_left_x = center_x - 80  # Left point of diamond
        decision_right_x = center_x + 80  # Right point of diamond
        
        result += f"STEP {step}: draw_decision({center_x-80}, {decision_y}, 160, 90, 'condition?')\n"
        result += f"// Diamond: TOP y={decision_y}, CENTER y={decision_center_y}, BOTTOM y={decision_bottom_y}\n"
        result += f"// Diamond: LEFT x={decision_left_x}, CENTER x={center_x}, RIGHT x={decision_right_x}\n\n"
        step += 1
        
        if has_branches:
            # Branch boxes at same Y level, to left and right
            branch_y = decision_bottom_y + 80
            yes_box_x = center_x + 100
            no_box_x = center_x - 250
            
            result += f"STEP {step}: draw_process({yes_box_x}, {branch_y}, 140, 50, 'True case')\n"
            result += f"// YES box: LEFT x={yes_box_x}, CENTER x={yes_box_x + 70}, RIGHT x={yes_box_x + 140}\n"
            result += f"// YES box: TOP y={branch_y}, CENTER y={branch_y + 25}, BOTTOM y={branch_y + 50}\n\n"
            yes_box_center_x = yes_box_x + 70
            yes_box_top_y = branch_y
            yes_box_bottom_y = branch_y + 50
            step += 1
            
            result += f"STEP {step}: draw_process({no_box_x}, {branch_y}, 140, 50, 'False case')\n"
            result += f"// NO box: LEFT x={no_box_x}, CENTER x={no_box_x + 70}, RIGHT x={no_box_x + 140}\n"
            result += f"// NO box: TOP y={branch_y}, CENTER y={branch_y + 25}, BOTTOM y={branch_y + 50}\n\n"
            no_box_center_x = no_box_x + 70
            no_box_top_y = branch_y
            no_box_bottom_y = branch_y + 50
            step += 1
            
            # Merge point and end
            merge_y = branch_y + 50 + 80
            end_y = merge_y + 60
            
            result += f"STEP {step}: draw_terminal({center_x-70}, {end_y}, 140, 50, 'End')\n"
            result += f"// END terminal: TOP y={end_y}\n\n"
            step += 1
            
            # Now arrows
            result += f"\n=== ARROWS (draw AFTER all shapes) ===\n\n"
            
            result += f"ARROW 1: Start to First\n"
            result += f"draw_arrow({center_x}, {start_bottom_y}, {center_x}, {first_top_y}, '')\n"
            result += f"// VERTICAL: from start BOTTOM to first TOP\n\n"
            
            result += f"ARROW 2: First to Decision\n"
            result += f"draw_arrow({center_x}, {first_bottom_y}, {center_x}, {decision_top_y}, '')\n"
            result += f"// VERTICAL: from first BOTTOM to decision TOP\n\n"
            
            result += f"ARROW 3: Decision to YES box (going RIGHT then DOWN)\n"
            result += f"draw_arrow({decision_right_x}, {decision_center_y}, {yes_box_center_x}, {decision_center_y}, '')\n"
            result += f"draw_arrow({yes_box_center_x}, {decision_center_y}, {yes_box_center_x}, {yes_box_top_y}, 'Yes')\n"
            result += f"// HORIZONTAL right from decision, then VERTICAL down to YES box\n\n"
            
            result += f"ARROW 4: Decision to NO box (going LEFT then DOWN)\n"
            result += f"draw_arrow({decision_left_x}, {decision_center_y}, {no_box_center_x}, {decision_center_y}, '')\n"
            result += f"draw_arrow({no_box_center_x}, {decision_center_y}, {no_box_center_x}, {no_box_top_y}, 'No')\n"
            result += f"// HORIZONTAL left from decision, then VERTICAL down to NO box\n\n"
            
            result += f"ARROW 5: YES box to merge point\n"
            result += f"draw_arrow({yes_box_center_x}, {yes_box_bottom_y}, {yes_box_center_x}, {merge_y}, '')\n"
            result += f"// VERTICAL: from YES box BOTTOM down to merge level\n\n"
            
            result += f"ARROW 6: NO box to merge point\n"
            result += f"draw_arrow({no_box_center_x}, {no_box_bottom_y}, {no_box_center_x}, {merge_y}, '')\n"
            result += f"// VERTICAL: from NO box BOTTOM down to merge level\n\n"
            
            result += f"ARROW 7: YES branch merge to center\n"
            result += f"draw_arrow({yes_box_center_x}, {merge_y}, {center_x}, {merge_y}, '')\n"
            result += f"// HORIZONTAL: from YES merge point to center\n\n"
            
            result += f"ARROW 8: NO branch merge to center\n"
            result += f"draw_arrow({no_box_center_x}, {merge_y}, {center_x}, {merge_y}, '')\n"
            result += f"// HORIZONTAL: from NO merge point to center\n\n"
            
            result += f"ARROW 9: Merge to End\n"
            result += f"draw_arrow({center_x}, {merge_y}, {center_x}, {end_y}, '')\n"
            result += f"// VERTICAL: from merge point down to END terminal TOP\n\n"
    else:
        # Simple flow without decision
        end_y = current_y
        result += f"STEP {step}: draw_terminal({center_x-70}, {end_y}, 140, 50, 'End')\n\n"
        step += 1
        
        result += f"\n=== ARROWS ===\n\n"
        result += f"ARROW 1: draw_arrow({center_x}, {start_bottom_y}, {center_x}, {first_top_y}, '')\n"
        if num_main_elements >= 2:
            result += f"ARROW 2: draw_arrow({center_x}, {first_bottom_y}, {center_x}, {end_y}, '')\n"
    
    result += "\n**CRITICAL RULES:**\n"
    result += "1. Use EXACT coordinates provided above\n"
    result += "2. For decision diamonds: arrows go to/from the SIDE POINTS (left/right), not corners\n"
    result += "3. Branch arrows: horizontal from decision, then vertical down to boxes\n"
    result += "4. All vertical arrows should have SAME X coordinate\n"
    result += "5. All horizontal arrows should have SAME Y coordinate\n"
    result += "6. Draw shapes FIRST, then ALL arrows\n"
    result += "7. Keep text SHORT (3-6 words max)\n"
    
    return result

@tool(description="Draw process rectangle for flowcharts.")
def draw_process(x: int, y: int, width: int, height: int, text: str) -> str:
    if canvas is None or draw is None:
        return "ERROR: Canvas not created"
    
    # Draw rectangle
    draw.rectangle([x, y, x + width, y + height], fill="#E3F2FD", outline="#1976D2", width=3)
    
    # Draw text - simplified, no wrapping for flowcharts
    font = ImageFont.load_default()
    
    # Calculate text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text
    text_x = x + (width - text_width) // 2
    text_y = y + (height - text_height) // 2
    
    # Draw text once
    draw.text((text_x, text_y), text, fill="black", font=font)
    
    return f"Process '{text}' drawn at ({x},{y})"

@tool(description="Draw decision diamond for flowcharts.")
def draw_decision(x: int, y: int, width: int, height: int, text: str) -> str:
    if canvas is None or draw is None:
        return "ERROR: Canvas not created"
    
    cx = x + width // 2
    cy = y + height // 2
    points = [(cx, y), (x + width, cy), (cx, y + height), (x, cy)]
    
    # Draw diamond
    draw.polygon(points, fill="#FFF9C4", outline="#F57C00", width=3)
    
    # Draw text - simplified, no wrapping
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text
    text_x = cx - text_width // 2
    text_y = cy - text_height // 2
    
    # Draw text once
    draw.text((text_x, text_y), text, fill="black", font=font)
    
    return f"Decision '{text}' drawn at ({x},{y})"

@tool(description="Draw terminal for flowcharts.")
def draw_terminal(x: int, y: int, width: int, height: int, text: str) -> str:
    if canvas is None or draw is None:
        return "ERROR: Canvas not created"
    
    radius = height // 2
    draw.rounded_rectangle([x, y, x + width, y + height], radius=radius, fill="#C8E6C9", outline="#388E3C", width=3)
    
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = x + (width - text_width) // 2
    text_y = y + (height - text_height) // 2
    draw.text((text_x, text_y), text, fill="black", font=font)
    
    return f"Terminal '{text}' drawn at ({x},{y})"

@tool(description="Draw arrow for flowcharts.")
def draw_arrow(x1: int, y1: int, x2: int, y2: int, label: str = "") -> str:
    if canvas is None or draw is None:
        return "ERROR: Canvas not created"
    
    # Draw line
    draw.line([(x1, y1), (x2, y2)], fill="black", width=2)
    
    # Calculate and draw arrowhead
    angle = math.atan2(y2 - y1, x2 - x1)
    arrow_len = 12
    arrow_angle = math.pi / 6
    
    p1 = (x2 - arrow_len * math.cos(angle - arrow_angle), y2 - arrow_len * math.sin(angle - arrow_angle))
    p2 = (x2 - arrow_len * math.cos(angle + arrow_angle), y2 - arrow_len * math.sin(angle + arrow_angle))
    draw.polygon([p1, (x2, y2), p2], fill="black")
    
    # Add label if provided - with white background to prevent overlap
    if label and label.strip():
        font = ImageFont.load_default()
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        
        # Offset label based on arrow direction
        if abs(x2 - x1) > abs(y2 - y1):  # Horizontal arrow
            label_y = mid_y - 15
            label_x = mid_x
        else:  # Vertical arrow
            label_y = mid_y
            label_x = mid_x + 10
        
        # Draw white background for label
        bbox = draw.textbbox((label_x, label_y), label, font=font)
        padding = 2
        draw.rectangle(
            [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding],
            fill="white"
        )
        
        # Draw label text
        draw.text((label_x, label_y), label, fill="black", font=font)
    
    return f"Arrow drawn from ({x1},{y1}) to ({x2},{y2})" + (f" [{label}]" if label else "")

@tool(description="Draw input/output parallelogram for flowcharts.")
def draw_input_output(x: int, y: int, width: int, height: int, text: str, is_input: bool = True) -> str:
    if canvas is None or draw is None:
        return "ERROR: Canvas not created"
    
    skew = 20
    points = [(x + skew, y), (x + width, y), (x + width - skew, y + height), (x, y + height)]
    
    color = "#E1BEE7" if is_input else "#FFCCBC"
    outline = "#7B1FA2" if is_input else "#E64A19"
    
    draw.polygon(points, fill=color, outline=outline, width=3)
    
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = x + (width - text_width) // 2
    text_y = y + (height - text_height) // 2
    draw.text((text_x, text_y), text, fill="black", font=font)
    
    return f"Input/Output '{text}' drawn at ({x},{y})"

# ============================================================================
# PYDANTIC MODELS & API SETUP
# ============================================================================

class AWSCredentials(BaseModel):
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_SESSION_TOKEN: Optional[str] = None
    AWS_REGION: str

class ChatRequest(BaseModel):
    question: str
    credentials: AWSCredentials
    session_id: Optional[str] = "default"

def create_bedrock_client(credentials: AWSCredentials):
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=credentials.AWS_REGION,
        aws_access_key_id=credentials.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=credentials.AWS_SECRET_ACCESS_KEY,
        aws_session_token=credentials.AWS_SESSION_TOKEN,
    )

def get_llm(bedrock_client, model=None, temperature: float = 0.2):
    return ChatBedrock(
        client=bedrock_client,
        model_id=model or "amazon.nova-lite-v1:0",
        inference_config={"temperature": float(temperature)}
    )

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def read_root():
    return {"message": "Unified Drawing API is running"}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        global canvas, draw
        canvas = None
        draw = None
        
        bedrock_client = create_bedrock_client(request.credentials)
        model = get_llm(bedrock_client=bedrock_client, model="amazon.nova-lite-v1:0", temperature=0.3)

        # All available tools
        tools = [
            # Core
            create_canvas,
            # Basic shapes (creative drawings)
            add_rectangle, add_circle, add_ellipse, add_polygon, add_line,
            add_star, add_regular_polygon, add_gradient, add_text,
            # Architecture diagrams
            draw_component, draw_connection, draw_database, draw_cloud, add_label,
            # Flowcharts
            plan_flowchart, draw_process, draw_decision, draw_terminal, 
            draw_arrow, draw_input_output
        ]

        system_prompt = """You are an expert drawing assistant that can create:
1. **ARCHITECTURE DIAGRAMS** - System designs, microservices, APIs, databases
2. **FLOWCHARTS** - Algorithms, logic flows, decision trees
3. **CREATIVE DRAWINGS** - Scenes, objects, illustrations using basic shapes

=== DIAGRAM TYPE DETECTION ===

**Architecture Diagram Keywords:** microservices, API, database, cloud, authentication, load balancer, cache, queue, system design, infrastructure, backend, frontend, service

**Flowchart Keywords:** algorithm, if/else, loop, decision, process flow, steps, logic, while, for, condition

**Creative Drawing Keywords:** draw, create, make, scene, landscape, tree, house, flag, person, animal, sun, sky

=== ARCHITECTURE DIAGRAMS ===

**Canvas Size:** 900x700

**Tools:**
- draw_component(x, y, width, height, text, color, component_type) - Main building block
- draw_database(x, y, width, height, text) - Cylinder for databases
- draw_cloud(x, y, width, height, text) - Cloud shape
- draw_connection(x1, y1, x2, y2, label, style, bidirectional) - Arrows
- add_label(x, y, text, size) - Titles

**Component Types:**
- "service": APIs, microservices (BLUE)
- "database": SQL, NoSQL (GREEN)
- "client": Frontend, mobile (PURPLE)
- "external": Third-party APIs (ORANGE)
- "queue": Message queues (YELLOW)
- "cache": Redis, Memcached (PINK)
- "storage": S3, file storage (BROWN)

**Layout Strategy (Layered Top-to-Bottom):**
- Layer 1 (y=50-80): Clients
- Layer 2 (y=200-230): API Gateway/Load Balancer
- Layer 3 (y=350-400): Services
- Layer 4 (y=500-560): Databases/Storage

**CRITICAL CONNECTION RULES:**
Each draw_component/draw_database/draw_cloud returns connection points like:
"Component 'API Gateway' at (150,230). Connections: top=(240,230), bottom=(240,330), left=(150,280), right=(330,280)"

**YOU MUST:**
1. READ the connection points from each component's return message
2. Use BOTTOM point of upper component as arrow start (x1, y1)
3. Use TOP point of lower component as arrow end (x2, y2)
4. For horizontal: use RIGHT of left component → LEFT of right component

**EXAMPLE:**
```
Component A drawn: bottom=(240, 330)
Component B drawn: top=(335, 400)
draw_connection(240, 330, 335, 400, "API call", "solid", False)
```

**DO NOT:**
- Connect from top to top (causes overlap)
- Connect from bottom to bottom (causes overlap)
- Guess coordinates - always use the returned connection points

**EXAMPLE ARCHITECTURE FLOW:**

Step 1: create_canvas(900, 700, "white")

Step 2: add_label(20, 20, "System Architecture", "title")

Step 3: Draw components layer by layer

```
# Layer 1 - Client
result1 = draw_component(50, 80, 150, 80, "Web Client", "lightblue", "client")
# Returns: "...Connections: top=(125, 80), bottom=(125, 160), left=(50, 120), right=(200, 120)"
# Extract: client_bottom = (125, 160)

# Layer 2 - API Gateway  
result2 = draw_component(150, 230, 180, 100, "API Gateway", "lightblue", "service")
# Returns: "...Connections: top=(240, 230), bottom=(240, 330), left=(150, 280), right=(330, 280)"
# Extract: gateway_top = (240, 230), gateway_bottom = (240, 330)

# Layer 3 - Service
result3 = draw_component(50, 400, 160, 90, "Auth Service", "lightblue", "service")
# Returns: "...Connections: top=(130, 400), bottom=(130, 490), left=(50, 445), right=(210, 445)"
# Extract: service_top = (130, 400), service_bottom = (130, 490)

# Layer 4 - Database
result4 = draw_database(180, 560, 140, 100, "PostgreSQL")
# Returns: "...Connections: top=(250, 560), bottom=(250, 660)"
# Extract: db_top = (250, 560)
```

Step 4: Connect using extracted points

```
# Client to Gateway: client BOTTOM → gateway TOP
draw_connection(125, 160, 240, 230, "HTTPS", "solid", False)

# Gateway to Service: gateway BOTTOM → service TOP
draw_connection(240, 330, 130, 400, "POST /login", "solid", False)

# Service to Database: service BOTTOM → database TOP
draw_connection(130, 490, 250, 560, "SQL Query", "solid", False)
```

**Key Points:**
- Vertical flow: upper.bottom → lower.top
- Horizontal flow: left.right → right.left
- Always extract and use the returned coordinates

=== FLOWCHARTS ===

**Canvas Size:** 700x800

**Workflow:**
1. create_canvas(700, 800)
2. plan_flowchart(num_elements, has_decision, has_branches) - Get layout guide
3. Follow exact coordinates from plan
4. Draw all shapes first, then all arrows

**Tools:**
- draw_terminal(x, y, width, height, text) - Start/End
- draw_process(x, y, width, height, text) - Process steps
- draw_decision(x, y, width, height, text) - Conditions
- draw_arrow(x1, y1, x2, y2, label) - Connections
- draw_input_output(x, y, width, height, text, is_input) - I/O

**IMPORTANT TEXT RULES:**
- Keep text SHORT and CONCISE (3-6 words max per box)
- Use abbreviations when needed
- Examples: "Input num", "n % 2 == 0?", "Print Even", "Print Odd"
- Avoid long sentences - they will overlap!

**Critical:** Use plan_flowchart first to get exact coordinates!

=== CREATIVE DRAWINGS ===

**Tools:**
- add_rectangle, add_circle, add_ellipse, add_polygon
- add_star, add_regular_polygon
- add_line, add_gradient
- add_text

**Techniques:**
1. **Layering:** Background → Midground → Foreground
2. **Gradients:** Use for skies, sunsets (add_gradient)
3. **Colors:** Use hex codes (#RRGGBB) for variety
4. **Composition:** Combine many shapes for complex objects

**Example - Simple Tree:**
```
create_canvas(400, 400)
add_gradient("#87CEEB", "#E0F6FF", "vertical")  # Sky
add_rectangle(180, 250, 40, 100, "#8B4513", "black", 0)  # Trunk
add_circle(200, 230, 50, "#228B22", "#228B22", 0)  # Foliage
add_circle(180, 220, 35, "#32CD32", "#32CD32", 0)  # Highlights
add_circle(220, 220, 35, "#006400", "#006400", 0)  # Shadows
```

=== EXECUTION STRATEGY ===

**Step 1: Analyze Request**
- What type of diagram? (architecture/flowchart/creative)
- What's the complexity? (simple/medium/complex)
- What elements are needed?

**Step 2: Choose Tools**
- Architecture → draw_component, draw_connection
- Flowchart → plan_flowchart, draw_process, draw_decision
- Creative → add_rectangle, add_circle, add_gradient

**Step 3: Draw in Order**
1. ALWAYS create_canvas first
2. Add background (gradient for creative)
3. Draw all shapes/components
4. Draw all connections/arrows last
5. Add labels/text

**Key Rules:**
- Canvas MUST be created first
- For architecture: READ connection points from draw_component output
- For flowcharts: Use plan_flowchart to get layout
- For creative: Layer from back to front
- Text wraps automatically - keep it concise
- Use appropriate canvas sizes (900x700 arch, 700x800 flow, variable creative)

**Decision Process:**
1. Read the user's request carefully
2. Identify keywords to determine type
3. Select appropriate tools and layout
4. Execute in proper order
5. Ensure all coordinates are correct"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        agent = create_tool_calling_agent(llm=model, tools=tools, prompt=prompt)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=50)

        result = executor.invoke({"input": request.question})
        output_text = result.get("output", "Drawing completed")
        
        # Export canvas
        img_base64 = None
        if canvas is not None:
            buff = BytesIO()
            canvas.save(buff, format="PNG")
            buff.seek(0)
            img_base64 = base64.b64encode(buff.read()).decode('utf-8')

        return JSONResponse({
            "image": img_base64,
            "message": output_text
        } if img_base64 else {"message": output_text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/{session_id}")
def clear_session(session_id: str):
    global canvas, draw
    canvas = None
    draw = None
    return {"message": f"Session {session_id} cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)