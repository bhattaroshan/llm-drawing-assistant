from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3
from typing import Optional, Dict

from langchain_aws import ChatBedrock
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from fastapi import Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64
import math

app = FastAPI(title="AI Chat API")

# Configure CORS
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

@tool(description="Create a blank canvas with given width and height. If no dimensions specified, use 800x600.")
def create_canvas(width: int = 800, height: int = 600, background_color: str = "white") -> str:
    global canvas, draw
    canvas = Image.new("RGBA", (width, height), background_color)
    draw = ImageDraw.Draw(canvas)
    return f"Canvas created with size {width}x{height}, background={background_color}"

@tool(description="Draw a filled rectangle. Specify position (x,y), size (width, height), fill color, and outline color. Use width parameter for outline thickness.")
def add_rectangle(x: int = 0, y: int = 0, width: int = 100, height: int = 100, fill: str = "blue", outline: str = "black", outline_width: int = 1) -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    x0, y0 = x, y
    x1, y1 = x + width, y + height
    draw.rectangle([x0, y0, x1, y1], fill=fill, outline=outline, width=outline_width)
    return f"Rectangle drawn at ({x},{y}) with size {width}x{height}, fill={fill}"

@tool(description="Draw a filled circle/ellipse. Specify center position (x,y), radius, fill color, and outline color.")
def add_circle(x: int, y: int, radius: int, fill: str = "red", outline: str = "black", outline_width: int = 1) -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bbox, fill=fill, outline=outline, width=outline_width)
    return f"Circle drawn at ({x},{y}) with radius {radius}, fill={fill}"

@tool(description="Draw an ellipse (oval). Specify center (x,y), horizontal radius (rx), vertical radius (ry). Useful for eggs, eyes, leaves.")
def add_ellipse(x: int, y: int, rx: int, ry: int, fill: str = "blue", outline: str = "black", outline_width: int = 1) -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    bbox = [x - rx, y - ry, x + rx, y + ry]
    draw.ellipse(bbox, fill=fill, outline=outline, width=outline_width)
    return f"Ellipse drawn at ({x},{y}) with radii ({rx},{ry}), fill={fill}"

@tool(description="Draw a polygon from a list of (x,y) points. Points should be a list of tuples like [(x1,y1), (x2,y2), (x3,y3)].")
def add_polygon(points: list, fill: str = "green", outline: str = "black", outline_width: int = 1) -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    
    # Normalize points to tuples
    normalized_points = []
    for p in points:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            normalized_points.append(tuple(p))
        elif isinstance(p, dict) and "x" in p and "y" in p:
            normalized_points.append((p["x"], p["y"]))
    
    if len(normalized_points) < 3:
        return f"Need at least 3 points for polygon, got {len(normalized_points)}"
    
    draw.polygon(normalized_points, fill=fill, outline=outline)
    return f"Polygon drawn with {len(normalized_points)} points, fill={fill}"

@tool(description="Draw a line from point (x1,y1) to (x2,y2). Useful for stems, branches, borders, connections.")
def add_line(x1: int, y1: int, x2: int, y2: int, fill: str = "black", width: int = 2) -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    draw.line([(x1, y1), (x2, y2)], fill=fill, width=width)
    return f"Line drawn from ({x1},{y1}) to ({x2},{y2}), color={fill}, width={width}"

@tool(description="Draw an arc (curved line) within a bounding box. Angles in degrees. Start=0 is 3 o'clock, goes counter-clockwise.")
def add_arc(x: int, y: int, width: int, height: int, start_angle: int, end_angle: int, fill: str = "black", width_px: int = 2) -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    bbox = [x, y, x + width, y + height]
    draw.arc(bbox, start=start_angle, end=end_angle, fill=fill, width=width_px)
    return f"Arc drawn in box ({x},{y},{width},{height}), angles={start_angle}-{end_angle}"

@tool(description="Draw a chord (filled arc wedge). Like a pie slice. Useful for pie charts, wedges, partial circles.")
def add_chord(x: int, y: int, width: int, height: int, start_angle: int, end_angle: int, fill: str = "yellow", outline: str = "black") -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    bbox = [x, y, x + width, y + height]
    draw.chord(bbox, start=start_angle, end=end_angle, fill=fill, outline=outline)
    return f"Chord drawn, angles={start_angle}-{end_angle}, fill={fill}"

@tool(description="Draw a pieslice (filled triangular wedge from center). Like pizza slice. Useful for pie charts.")
def add_pieslice(x: int, y: int, width: int, height: int, start_angle: int, end_angle: int, fill: str = "orange", outline: str = "black") -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    bbox = [x, y, x + width, y + height]
    draw.pieslice(bbox, start=start_angle, end=end_angle, fill=fill, outline=outline)
    return f"Pieslice drawn, angles={start_angle}-{end_angle}, fill={fill}"

@tool(description="Draw text at position (x,y). Use this to add labels or writing to the canvas.")
def add_text(x: int, y: int, text: str, font_size: int = 20, fill: str = "black") -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    
    try:
        font = ImageFont.load_default()
        draw.text((x, y), text, fill=fill, font=font)
        return f'Text "{text}" drawn at ({x},{y})'
    except Exception as e:
        return f"Error drawing text: {e}"

@tool(description="Draw a star shape with N points at position (x,y). Outer radius and inner radius control star shape. N=5 is typical star.")
def add_star(x: int, y: int, outer_radius: int, inner_radius: int, points: int = 5, fill: str = "yellow", outline: str = "black") -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    
    star_points = []
    angle_step = math.pi / points
    
    for i in range(points * 2):
        angle = i * angle_step - math.pi / 2
        if i % 2 == 0:
            r = outer_radius
        else:
            r = inner_radius
        px = x + r * math.cos(angle)
        py = y + r * math.sin(angle)
        star_points.append((px, py))
    
    draw.polygon(star_points, fill=fill, outline=outline)
    return f"Star with {points} points drawn at ({x},{y}), fill={fill}"

@tool(description="Draw a regular polygon (equal sides) with N sides. N=3 is triangle, N=6 is hexagon, etc.")
def add_regular_polygon(x: int, y: int, radius: int, sides: int, rotation: float = 0, fill: str = "blue", outline: str = "black") -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    
    points = []
    angle_step = 2 * math.pi / sides
    start_angle = rotation * math.pi / 180  # Convert to radians
    
    for i in range(sides):
        angle = i * angle_step + start_angle - math.pi / 2
        px = x + radius * math.cos(angle)
        py = y + radius * math.sin(angle)
        points.append((px, py))
    
    draw.polygon(points, fill=fill, outline=outline)
    return f"Regular {sides}-sided polygon drawn at ({x},{y}), fill={fill}"

@tool(description="Create a gradient background (vertical or horizontal). Use for sky, sunsets, realistic backgrounds.")
def add_gradient(start_color: str, end_color: str, direction: str = "vertical") -> str:
    if canvas is None:
        return "Canvas not created yet"
    
    width, height = canvas.size
    
    # Parse colors
    def parse_color(color_str):
        if color_str.startswith('#'):
            return tuple(int(color_str[i:i+2], 16) for i in (1, 3, 5))
        # Basic color mapping
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
    else:  # horizontal
        for i in range(width):
            ratio = i / width
            r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
            g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
            b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)
            draw.line([(i, 0), (i, height)], fill=(r, g, b))
    
    return f"Gradient from {start_color} to {end_color} ({direction}) applied"

@tool(description="Draw a flowchart process box (rectangle) with centered text inside.")
def add_flowchart_process(x: int, y: int, width: int, height: int, text: str, fill: str = "lightblue", outline: str = "black") -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    
    draw.rectangle([x, y, x + width, y + height], fill=fill, outline=outline, width=2)
    
    try:
        font = ImageFont.load_default()
        # Calculate text position (approximate centering)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x + (width - text_width) // 2
        text_y = y + (height - text_height) // 2
        draw.text((text_x, text_y), text, fill="black", font=font)
    except Exception as e:
        return f"Error drawing text: {e}"
    
    return f'Process box "{text}" drawn at ({x},{y})'

@tool(description="Draw a flowchart decision diamond with centered text inside.")
def add_flowchart_decision(x: int, y: int, width: int, height: int, text: str, fill: str = "lightyellow", outline: str = "black") -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    
    cx = x + width // 2
    cy = y + height // 2
    points = [
        (cx, y),              # top
        (x + width, cy),      # right
        (cx, y + height),     # bottom
        (x, cy)               # left
    ]
    
    draw.polygon(points, fill=fill, outline=outline)
    
    try:
        font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = cx - text_width // 2
        text_y = cy - text_height // 2
        draw.text((text_x, text_y), text, fill="black", font=font)
    except Exception as e:
        return f"Error drawing text: {e}"
    
    return f'Decision diamond "{text}" drawn at ({x},{y})'

@tool(description="Draw a flowchart arrow from (x1,y1) to (x2,y2) with arrowhead. Use for connecting flowchart elements.")
def add_flowchart_arrow(x1: int, y1: int, x2: int, y2: int, label: str = "", fill: str = "black") -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    
    # Draw main line
    draw.line([(x1, y1), (x2, y2)], fill=fill, width=2)
    
    # Calculate arrowhead
    import math
    angle = math.atan2(y2 - y1, x2 - x1)
    arrow_length = 10
    arrow_angle = math.pi / 6
    
    # Arrowhead points
    p1 = (
        x2 - arrow_length * math.cos(angle - arrow_angle),
        y2 - arrow_length * math.sin(angle - arrow_angle)
    )
    p2 = (
        x2 - arrow_length * math.cos(angle + arrow_angle),
        y2 - arrow_length * math.sin(angle + arrow_angle)
    )
    
    draw.polygon([p1, (x2, y2), p2], fill=fill)
    
    # Add label if provided
    if label:
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        font = ImageFont.load_default()
        draw.text((mid_x + 5, mid_y - 10), label, fill=fill, font=font)
    
    return f'Arrow drawn from ({x1},{y1}) to ({x2},{y2})' + (f' with label "{label}"' if label else '')

@tool(description="Draw a flowchart start/end terminal (rounded rectangle) with text.")
def add_flowchart_terminal(x: int, y: int, width: int, height: int, text: str, fill: str = "lightgreen", outline: str = "black") -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    
    radius = min(width, height) // 4
    
    # Draw rounded rectangle (simplified with arcs)
    draw.rounded_rectangle([x, y, x + width, y + height], radius=radius, fill=fill, outline=outline, width=2)
    
    # Draw centered text
    try:
        font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x + (width - text_width) // 2
        text_y = y + (height - text_height) // 2
        draw.text((text_x, text_y), text, fill="black", font=font)
    except Exception as e:
        return f"Error drawing text: {e}"
    
    return f'Terminal "{text}" drawn at ({x},{y})'

@tool(description="Draw a bezier curve (smooth curved line) through control points. Great for organic shapes, paths.")
def add_curved_line(points: list, fill: str = "black", width: int = 2) -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    
    # Normalize points
    normalized_points = []
    for p in points:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            normalized_points.append(tuple(p))
        elif isinstance(p, dict) and "x" in p and "y" in p:
            normalized_points.append((p["x"], p["y"]))
    
    if len(normalized_points) < 2:
        return "Need at least 2 points for curved line"
    
    # Draw smooth curve through points
    for i in range(len(normalized_points) - 1):
        draw.line([normalized_points[i], normalized_points[i + 1]], fill=fill, width=width)
    
    return f"Curved line drawn through {len(normalized_points)} points"


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


def get_llm(bedrock_client, model=None, temperature: float = 0.3):
    return ChatBedrock(
        client=bedrock_client,
        model_id=model or "amazon.nova-lite-v1:0",
        inference_config={
            "temperature": float(temperature)
        }
    )


@app.get("/")
def read_root():
    return {"message": "AI Chat API is running"}


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        global canvas, draw
        canvas = None
        draw = None
        
        bedrock_client = create_bedrock_client(request.credentials)

        model = get_llm(
            bedrock_client=bedrock_client,
            model="amazon.nova-lite-v1:0",
            temperature=0.4  # Slightly higher for creativity
        )

        tools = [
            create_canvas, add_rectangle, add_circle, add_ellipse, add_polygon, 
            add_line, add_arc, add_chord, add_pieslice, add_text, add_star,
            add_regular_polygon, add_gradient, add_curved_line,
            add_flowchart_process, add_flowchart_decision, add_flowchart_arrow, 
            add_flowchart_terminal
        ]

        system_prompt = """You are a creative drawing assistant that can compose complex, realistic drawings using basic shapes and advanced techniques.

You have these tools available:

BASIC SHAPES:
- create_canvas: Create canvas (ALWAYS call this first!) - can set background color
- add_rectangle: Draw rectangles with outline width control
- add_circle: Draw perfect circles
- add_ellipse: Draw ovals (different horizontal/vertical radii)
- add_polygon: Draw custom polygons from points
- add_regular_polygon: Draw regular polygons (triangle, hexagon, octagon, etc.) with rotation

ADVANCED SHAPES:
- add_star: Draw star shapes (specify points, inner/outer radius)
- add_line: Draw straight lines (for stems, branches, connections)
- add_curved_line: Draw smooth curves through multiple points
- add_arc: Draw curved arcs (part of circle/ellipse)
- add_chord: Draw filled arc segments
- add_pieslice: Draw pie/pizza slices

EFFECTS:
- add_gradient: Create smooth color gradients (vertical/horizontal) for realistic skies, sunsets
- add_text: Add text labels

IMPORTANT TECHNIQUES FOR REALISM:
1. **Layering**: Draw background elements first (sky, ground), then midground, then foreground
2. **Gradients**: Use add_gradient for skies (blue to cyan top to bottom), sunsets (orange to purple top to bottom), ground (green to dark green top to bottom)
3. **Color Variation**: Use different shades of same color (#228B22, #32CD32, #006400 for different greens)
4. **Overlapping**: Layer shapes to create depth and complexity
5. **Outlines**: Use outline_width=0 for smooth shapes, outline_width=2-3 for defined edges
6. **Rotation**: Use regular_polygon with rotation for varied orientations
7. **Multiple Shapes**: Combine many small shapes (e.g., many circles for tree foliage)

VALID COLOR NAMES:
- Basic: red, green, blue, yellow, orange, purple, pink, brown, gray, black, white, cyan
- Hex codes: #RRGGBB format (e.g., #DC143C for crimson, #8B4513 for saddle brown, #87CEEB for sky blue)
- Shades: Use hex codes for variations (#006400 dark green, #228B22 forest green, #90EE90 light green)

REALISTIC EXAMPLES:

**Tree with depth:**
1. add_rectangle (x=300, y=400, width=40, height=100, fill="#8B4513") - trunk
2. add_ellipse (x=320, y=350, rx=60, ry=70, fill="#228B22", outline_width=0) - main foliage
3. add_circle (x=280, y=340, radius=35, fill="#32CD32", outline_width=0) - left foliage cluster
4. add_circle (x=360, y=340, radius=35, fill="#006400", outline_width=0) - right foliage cluster
5. add_circle (x=320, y=310, radius=30, fill="#90EE90", outline_width=0) - top highlight

**Mountain range with sky:**
1. create_canvas(800, 600)
2. add_gradient("#87CEEB", "#E0F6FF", "vertical") - sky gradient
3. add_polygon([(0,400), (200,200), (400,400)], fill="#696969") - left mountain
4. add_polygon([(300,400), (500,150), (700,400)], fill="#808080") - center mountain
5. add_polygon([(600,400), (750,250), (800,400)], fill="#A9A9A9") - right mountain
6. add_polygon([(450,150), (500,120), (550,150)], fill="white") - snow cap

**Nepal Flag:**
1. create_canvas(400, 500)
2. add_polygon([(50,50), (50,250), (300,150)], fill="#DC143C", outline="blue", outline_width=8) - upper pennant
3. add_polygon([(50,250), (50,450), (300,350)], fill="#DC143C", outline="blue", outline_width=8) - lower pennant
4. add_star(x=150, y=150, outer_radius=30, inner_radius=15, points=12, fill="white", outline="white") - sun symbol
5. add_circle(x=150, y=340, radius=25, fill="white", outline="white") - moon (can add crescent with another shape)

**House with details:**
1. add_gradient("#87CEEB", "#E0F6FF", "vertical") - sky
2. add_rectangle(x=200, y=350, width=200, height=150, fill="#D2691E", outline="black") - walls
3. add_polygon([(180,350), (300,250), (420,350)], fill="#8B0000", outline="black") - roof
4. add_rectangle(x=250, y=400, width=40, height=60, fill="#8B4513", outline="black") - door
5. add_rectangle(x=320, y=380, width=35, height=35, fill="#87CEEB", outline="black") - window
6. add_line(x1=337, y1=380, x2=337, y2=415, fill="black", width=2) - window cross
7. add_line(x1=320, y1=397, x2=355, y2=397, fill="black", width=2) - window cross

Think step by step:
1. What's the background? (sky, ground) - use gradients
2. What are the main objects? Break into shapes
3. What details add realism? (shadows, highlights, textures)
4. What's the drawing order? (back to front)

Then execute tools in proper order for layering and depth.

**STANDARD FLOWCHART LAYOUT (FOLLOW EXACTLY):**

For vertical flowcharts (most common):
- Canvas: 600x800 or 800x600 depending on complexity
- Center X: 300 (for 600 width) or 400 (for 800 width)
- Start Y: 50
- Vertical spacing: 120px between elements
- Box dimensions: 140px wide, 50px tall
- Diamond dimensions: 140px wide, 80px tall

**ELEMENT POSITIONING FORMULA:**
1. Start/End Terminal: center_x=300, y=50, width=120, height=40
2. After each element, add: current_y + element_height + 120
3. For decision branches:
   - Left branch (No): x = center_x - 200
   - Right branch (Yes): x = center_x + 200
   - Both at same Y as decision + 120

**ARROW CONNECTION POINTS:**
- Rectangle bottom center: (x + width/2, y + height)
- Rectangle top center: (x + width/2, y)
- Diamond bottom: (x + width/2, y + height)
- Diamond sides: left (x, y + height/2), right (x + width, y + height/2)

**CONCRETE EXAMPLE - ODD/EVEN FLOWCHART:**
```
Canvas: 600 wide, so center_x = 300

1. create_canvas(600, 700, "white")

2. add_flowchart_terminal(x=240, y=50, width=120, height=40, text="Start")
   // Center: 240 + 60 = 300

3. add_flowchart_arrow(x1=300, y1=90, x2=300, y2=170)
   // From: terminal bottom (300, 90), To: next element top (300, 170)

4. add_flowchart_process(x=230, y=170, width=140, height=50, text="Input Number")
   // Center: 230 + 70 = 300, Bottom: 220

5. add_flowchart_arrow(x1=300, y1=220, x2=300, y2=290)
   // From: process bottom (300, 220), To: decision top (300, 290)

6. add_flowchart_decision(x=230, y=290, width=140, height=80, text="n % 2 == 0?")
   // Center: 230 + 70 = 300, Right edge: 370, Left edge: 230, Bottom: 370

7. add_flowchart_arrow(x1=370, y1=330, x2=450, y2=330, label="Yes")
   // From: decision right (370, 330), To: right box (450, 330)

8. add_flowchart_process(x=450, y=310, width=100, height=40, text="Even")
   // Right branch: x=450 (= 300 + 150), y=310 (decision center Y)

9. add_flowchart_arrow(x1=230, y1=330, x2=150, y2=330, label="No")
   // From: decision left (230, 330), To: left box (150, 330)

10. add_flowchart_process(x=100, y=310, width=100, height=40, text="Odd")
    // Left branch: x=100 (= 300 - 200), y=310 (decision center Y)

11. add_flowchart_arrow(x1=500, y1=350, x2=500, y2=450, label="")
    // From Even box down

12. add_flowchart_arrow(x1=150, y1=350, x2=150, y2=450, label="")
    // From Odd box down

13. add_flowchart_arrow(x1=500, y1=450, x2=300, y2=450, label="")
    // Right merge to center

14. add_flowchart_arrow(x1=150, y1=450, x2=300, y2=450, label="")
    // Left merge to center

15. add_flowchart_arrow(x1=300, y1=450, x2=300, y2=490, label="")
    // Down to end

16. add_flowchart_terminal(x=240, y=490, width=120, height=40, text="End")
    // Center: 300
```

**CALCULATION HELPER:**
When drawing flowcharts:
1. Write down your canvas width → calculate center_x = width / 2
2. List all elements vertically with Y positions:
   - Element 1: y = 50
   - Element 2: y = 50 + 40 + 120 = 210
   - Element 3: y = 210 + 50 + 120 = 380
3. For decisions, calculate branch positions:
   - Left: center_x - 180
   - Right: center_x + 180
4. Calculate exact arrow coordinates using element centers

**DEBUGGING TIP:**
If positions look wrong, recalculate:
- Box center X = x + (width / 2)
- Box bottom Y = y + height
- Diamond center = (x + width/2, y + height/2)

"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        agent = create_tool_calling_agent(
            llm=model,
            tools=tools,
            prompt=prompt
        )

        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=30  # Allow more iterations for complex drawings
        )

        result = executor.invoke({"input": request.question})
        output_text = result.get("output", "Drawing completed")
        
        # Export canvas to base64
        img_base64 = None
        if canvas is not None:
            try:
                buff = BytesIO()
                canvas.save(buff, format="PNG")
                buff.seek(0)
                img_base64 = base64.b64encode(buff.read()).decode('utf-8')
            except Exception as e:
                print(f"Error encoding image: {e}")

        res = {}
        if img_base64:
            res["image"] = img_base64
            res["message"] = output_text
        else:
            res["message"] = output_text

        return JSONResponse(res)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/{session_id}")
def clear_session(session_id: str):
    """Clear conversation history for a session"""
    global canvas, draw
    canvas = None
    draw = None
    return {"message": f"Session {session_id} cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)