# ===============================
# imports
# ===============================
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import boto3

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_aws import ChatBedrock
import os
import math
from langchain.tools import tool
from PIL import Image,ImageDraw, ImageFont
from io import BytesIO
import base64
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Global canvas state
canvas = None
draw = None

def safe_color(color_str):
    if color_str is None:
        return None
    if isinstance(color_str, str) and color_str.lower() == "transparent":
        return None  # PIL treats None as no fill/outline
    return color_str

def create_canvas(width: int = 800, height: int = 600, background_color: str = "white") -> str:
    global canvas, draw
    canvas = Image.new("RGBA", (width, height), background_color)
    draw = ImageDraw.Draw(canvas)
    return f"Canvas created with size {width}x{height}, background={background_color}"

@tool(description="Draw a filled rectangle. Specify position (x,y), size (width, height), fill color, and outline color. Use width parameter for outline thickness.")
def add_rectangle(x: int = 0, y: int = 0, width: int = 100, height: int = 100, fill: str = "blue", outline: str = "black", outline_width: int = 1) -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    fill = safe_color(fill)
    outline = safe_color(outline)
    x0, y0 = x, y
    x1, y1 = x + width, y + height
    draw.rectangle([x0, y0, x1, y1], fill=fill, outline=outline, width=outline_width)
    return f"Rectangle drawn at ({x},{y}) with size {width}x{height}, fill={fill}"

@tool(description="Draw a filled circle/ellipse. Specify center position (x,y), radius, fill color, and outline color.")
def add_circle(x: int, y: int, radius: int, fill: str = "red", outline: str = "black", outline_width: int = 1) -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    fill = safe_color(fill)
    outline = safe_color(outline)
    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bbox, fill=fill, outline=outline, width=outline_width)
    return f"Circle drawn at ({x},{y}) with radius {radius}, fill={fill}"

@tool(description="Draw an ellipse (oval). Specify center (x,y), horizontal radius (rx), vertical radius (ry). Useful for eggs, eyes, leaves.")
def add_ellipse(x: int, y: int, rx: int, ry: int, fill: str = "blue", outline: str = "black", outline_width: int = 1) -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    fill = safe_color(fill)
    outline = safe_color(outline)
    bbox = [x - rx, y - ry, x + rx, y + ry]
    draw.ellipse(bbox, fill=fill, outline=outline, width=outline_width)
    return f"Ellipse drawn at ({x},{y}) with radii ({rx},{ry}), fill={fill}"

@tool(description="Draw a polygon from a list of (x,y) points. Points should be a list of tuples like [(x1,y1), (x2,y2), (x3,y3)].")
def add_polygon(points: list, fill: str = "green", outline: str = "black", outline_width: int = 1) -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    
    fill = safe_color(fill)
    outline = safe_color(outline)

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
    fill = safe_color(fill)
    draw.line([(x1, y1), (x2, y2)], fill=fill, width=width)
    return f"Line drawn from ({x1},{y1}) to ({x2},{y2}), color={fill}, width={width}"

@tool(description="Draw an arc (curved line) within a bounding box. Angles in degrees. Start=0 is 3 o'clock, goes counter-clockwise.")
def add_arc(x: int, y: int, width: int, height: int, start_angle: int, end_angle: int, fill: str = "black", width_px: int = 2) -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    fill = safe_color(fill)
    bbox = [x, y, x + width, y + height]
    draw.arc(bbox, start=start_angle, end=end_angle, fill=fill, width=width_px)
    return f"Arc drawn in box ({x},{y},{width},{height}), angles={start_angle}-{end_angle}"

@tool(description="Draw a chord (filled arc wedge). Like a pie slice. Useful for pie charts, wedges, partial circles.")
def add_chord(x: int, y: int, width: int, height: int, start_angle: int, end_angle: int, fill: str = "yellow", outline: str = "black") -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    fill = safe_color(fill)
    outline = safe_color(outline)
    bbox = [x, y, x + width, y + height]
    draw.chord(bbox, start=start_angle, end=end_angle, fill=fill, outline=outline)
    return f"Chord drawn, angles={start_angle}-{end_angle}, fill={fill}"

@tool(description="Draw a pieslice (filled triangular wedge from center). Like pizza slice. Useful for pie charts.")
def add_pieslice(x: int, y: int, width: int, height: int, start_angle: int, end_angle: int, fill: str = "orange", outline: str = "black") -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    fill = safe_color(fill)
    outline = safe_color(outline)
    bbox = [x, y, x + width, y + height]
    draw.pieslice(bbox, start=start_angle, end=end_angle, fill=fill, outline=outline)
    return f"Pieslice drawn, angles={start_angle}-{end_angle}, fill={fill}"

@tool(description="Draw text at position (x,y). Use this to add labels or writing to the canvas.")
def add_text(x: int, y: int, text: str, font_size: int = 20, fill: str = "black") -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    fill = safe_color(fill)
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
    fill = safe_color(fill)
    outline = safe_color(outline)
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

@tool(description="Draw a regular polygon (equal sides) with N sides. N=3 is triangle, N=6 is hexagon, etc.")
def add_regular_polygon(x: int, y: int, radius: int, sides: int, rotation: float = 0, fill: str = "blue", outline: str = "black") -> str:
    if canvas is None or draw is None:
        return "Canvas not created yet"
    fill = safe_color(fill)
    outline = safe_color(outline)
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

# ===============================
# pydantic models
# ===============================

class DrawObject(BaseModel):
    name: str = Field(..., description="Drawable object like tree, house, sun")
    count: int = Field(1, ge=1, le=20)

class DrawIntent(BaseModel):
    objects: List[DrawObject]

class AWSCredentials(BaseModel):
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_SESSION_TOKEN: Optional[str] = None
    AWS_REGION: str

class ChatRequest(BaseModel):
    question: str
    credentials: AWSCredentials


# ===============================
# prompt
# ===============================

DRAW_PARSER_PROMPT = """
You extract drawable objects from user instructions.

Rules:
- Return ONLY valid JSON matching the schema.
- Objects must be simple nouns (tree, house, mountain, sun, river).
- If user says "few", use count = 3 to 5.
- If user says "many", use count = 6 to 10.
- If count is not specified, use 1.
- Merge duplicates.

Example:
User: draw a few trees and a house
Output:
{{
  "objects": [
    {{ "name": "tree", "count": 4 }},
    {{ "name": "house", "count": 1 }}
  ]
}}
"""


# ===============================
# bedrock helpers
# ===============================

def create_bedrock_client(credentials: AWSCredentials):
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=credentials.AWS_REGION,
        aws_access_key_id=credentials.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=credentials.AWS_SECRET_ACCESS_KEY,
        aws_session_token=credentials.AWS_SESSION_TOKEN,
    )

def get_llm(bedrock_client, model="amazon.nova-lite-v1:0", temperature=0.3):
    return ChatBedrock(
        client=bedrock_client,
        model_id=model,
        inference_config={"temperature": float(temperature)}
    )


# ===============================
# core logic
# ===============================

def parse_draw_intent(llm, user_input: str) -> DrawIntent:
    parser = PydanticOutputParser(pydantic_object=DrawIntent)

    prompt = ChatPromptTemplate.from_messages([
        ("system", DRAW_PARSER_PROMPT),
        ("human", "{input}\n\n{format_instructions}")
    ])

    chain = prompt | llm | parser

    return chain.invoke({
        "input": user_input,
        "format_instructions": parser.get_format_instructions()
    })


# ===============================
# fastapi app
# ===============================

app = FastAPI(title="Draw Intent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_kb_context(object_name: str) -> Optional[str]:
    """
    Returns KB content if present, else None
    """
    filename = f"{object_name.lower()}.txt"
    filepath = os.path.join("kb/", filename)

    if not os.path.exists(filepath):
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()
    
def enrich_with_kb(draw_intent: DrawIntent):
    enriched_objects = []

    for obj in draw_intent.objects:
        kb_context = load_kb_context(obj.name)

        enriched_objects.append({
            "name": obj.name,
            "count": obj.count,
            "kb_context": kb_context  # None if not found
        })

    return enriched_objects

def build_input_with_kb(user_question: str, enriched_objects: list) -> str:
    kb_texts = []
    for obj in enriched_objects:
        if obj['kb_context']:
            kb_texts.append(f"Knowledge about {obj['name']}:\n{obj['kb_context']}")
    kb_combined = "\n\n".join(kb_texts)

    full_input = f"{user_question}\n\n{kb_combined}"
    return full_input

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        bedrock_client = create_bedrock_client(request.credentials)
        llm = get_llm(bedrock_client)

        draw_intent = parse_draw_intent(llm, request.question)
        enriched_objects = enrich_with_kb(draw_intent)
        enhanced_input = build_input_with_kb(request.question, enriched_objects)

        # Create canvas for this request
        create_canvas()

        # Define prompt for agent
        system_prompt = """You are an AI drawing assistant. Use the drawing tools to create images based on user instructions."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        tools = [
            add_rectangle, add_circle, add_ellipse, add_polygon, 
            add_line, add_arc, add_chord, add_pieslice, add_text, add_star,
            add_regular_polygon, add_gradient
        ]

        agent = create_tool_calling_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )

        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=30
        )

        print("********************")
        print(enhanced_input)
        enhanced_input = "Make sure not to overlap any drawing object over each other. "+ enhanced_input
        print("********************")
        result = executor.invoke({"input": enhanced_input})
        output_text = result.get("output", "Drawing completed")

        # Export canvas image as base64
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
