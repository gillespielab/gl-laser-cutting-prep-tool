# -*- coding: utf-8 -*-
"""
This tool streamlines the workflow for preparing files for laser cutters.

Instructions:
    1. Export the desired cuts as a pdf
    2. Run this tool on the pdf
    3. Check the results in Illustrator/Inkscape

USAGE:
    >>> LaserCuttingPrep.py <filename> <options>

RECOMMENDED:
    >>> LaserCuttingPrep.py <filename> --output <output path> --margin 0.3 --uls --centerv --debug
    OR
    >>> main([r"<filename>", '--output', r"<output path>", '--margin', '0.3', '--uls', '--centerv', '--debug'])
    OR (if filepaths do not contain spaces)
    >>> main(r"<filename> --output <output path> --margin 0.3 --uls --centerv --debug".split(' '))

All 3 of these calls are equivalent, but the first should be used in a terminal, while the second and third 
should be used in a python console (like Spyder or Jupyter) after running the file. Also, note that the third
option only works if there are no spaces in either the filename or the output path; its main advantage over 
the second option is that it's more convenient to type, so it isn't really worth making a structure to 
get around this (you could use a custom separator if you really want to though).


INSTALLING DEPENDENCIES
    Run the following commands in an Anaconda prompt, run as an administrator:
        pip install reportlab
        pip install svglib
        pip install pypdf

Alternatively, you can skip installing poppler and pdf2image, and convert the downloaded pdfs 
to svgs using the tool of your choice (Illustrator, Inkscape, svgconverter.com/pdf-to-svg, etc.)

Author: Violet Saathoff
"""

# Import Libraries
import os
import sys
from tqdm import trange
import argparse
from collections import defaultdict
import reportlab.graphics as rl
from reportlab.graphics import renderPDF
from reportlab.graphics.renderbase import colors
from svglib.svglib import svg2rlg
import numpy as np
from scipy.optimize import curve_fit
#import drawsvg as draw
#from pypdf import PdfMerger


"""Helper Functions"""


# reportlab path operators {operation code : number of parameters}
# this mapping can be found at https://github.com/MrBitBucket/reportlab-mirror/blob/master/src/reportlab/graphics/shapes.py#L932
class operators:
    MoveTo = 0
    LineTo = 1
    CurveTo = 2
    ClosePath = 3

    _Parameters = {
        0: 2,
        1: 2,
        2: 6,
        3: 0
    }

    _Names = ['Move To', 'Line To', 'Curve To', 'Close Path']

    def parse(codes: list, points: list) -> (int, list):
        """yields each operator code with its associated control points"""
        i = 0
        for o in codes:
            j = i + operators._Parameters[o]
            yield o, points[i:j]
            i = j
    
    # Estimate the Length of a Single Path Element given the starting point of the element (x, y), the op-code (op), the control points (points), and the start of the path (start = [x0, y0])
    def length(x: float, y: float, op: int, points: list, start: list) -> (float, float, float, bool):
        # initialize the variable for the path length
        l = 0

        # data validation (currently disabled)
        """
        if op not in operators._Parameters:
            raise KeyError(f"unregocnized opcode {op}, known values are: [" + ', '.join(f"{o}: {n}" for o, n in zip(operators._Parameters.keys(), operators._Names)) + ']')
        elif op != operators.CurveTo and len(points) != operators._Parameters[op]:
            raise ValueError(f"wrong number of control values ({len(points)}) for '{operators._Names[op]}'. expected {operators._Parameters[op]}")
        #"""
        
        # select the appropriate formula to use
        if op in (operators.MoveTo, operators.LineTo):
            l = dist.l2(x, y, *points)
            x, y = points
        elif op == operators.ClosePath:
            l = dist.l2(x, y, *start)
            x, y = start
        elif op == operators.CurveTo:
            # Estimate the Length of the Bezier Curve From the Straight-Line Distance and the Distance Following the Bounding Polygon
            lc = dist.l2(x, y, *points[-2:])
            lp = dist.l2(x, y, *points[:2]) + sum(dist.l2(*points[i:i + 4]) for i in range(0, len(points) - 2, 2))
            n = len(points) // 2                #  the degree of the bezier curve
            l = (2*lc + (n - 1)*lp) / (n + 1)   #  the estimated distance (from https://doi.org/10.1016/0925-7721(95)00054-2)
            x, y = points[-2:]
        
        # Return the ending position (x, y), the estimated path length (l), and whether or not the segment is a cut (as opposed to a move)
        return x, y, l, op != operators.MoveTo


# Standard Distance Formula
class dist:
    def l1(x1: float, y1: float, x2: float, y2: float):
        """compute the L1 (manhattan/taxicab) distance between 2 points"""
        return abs(x2 - x1) + abs(y2 - y1)

    def l2s(x1: float, y1: float, x2: float, y2: float):
        """compute the squared L2 (euclidean) distance between 2 points"""
        return (x2 - x1) ** 2 + (y2 - y1) ** 2

    def l2(x1: float, y1: float, x2: float, y2: float):
        """compute the L2 (euclidean) distance between 2 points"""
        return dist.l2s(x1, y1, x2, y2) ** 0.5

def format_time(seconds: float):
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    hours = int(round(hours))
    minutes = int(round(minutes))
    seconds = int(round(seconds))
    if minutes < 10: minutes = f"0{minutes}"
    if seconds < 10: seconds = f"0{seconds}"
    if hours:
        return f"{hours}:{minutes}:{seconds}"
    elif minutes:
        return f"{minutes}:{seconds}"
    else:
        return f"{seconds} seconds"

def ctrange(start: int, end: int = None, step: int = None, N:int = 100001, desc: str = ''):
    if end is None:
        end = start
        start = 0
    if step is None:
        step = 1

    if False and N > 100000: # TODO: fix trange (it says 'step' isn't valid)
        return trange(start, end, step=step, desc=desc)
    else:
        return range(start, end, step)


def add_segment_to_path(path: rl.shapes.Path, o: int, controls: list):
    """helper function for building paths"""

    # TODO: update this section as reportLab paths get updated

    if o == operators.MoveTo and (not path.points or tuple(path.points[-2:]) != tuple(controls)):
        path.moveTo(*controls)
    elif o == operators.LineTo:
        path.lineTo(*controls)
    elif o == operators.CurveTo:
        path.curveTo(*controls)
    elif o == operators.ClosePath:
        path.closePath()


def Line(x1: float, y1: float, x2: float, y2: float, **kwargs):
    """create a line between the 2 end points"""
    return Path([x1, y1, x2, y2], [operators.MoveTo, operators.LineTo], **kwargs)
    #return rl.shapes.Line(x1, y1, x2, y2, **kwargs)

metrics = defaultdict(int)

def reverse_path(pts:list, ops:list):
    ops = [operators.MoveTo] + ops[1:][::-1]
    grouped = [pts[i:i + 2] for i in range(0, len(pts), 2)][::-1]
    points = []
    for P in grouped:
        points.extend(P)
    return points, ops

def Path(pts: list = None, ops: list = None, path: rl.shapes.Path = None, isClipPath: bool = 0, **kwargs):
    """create a new path from a set of points/operators, or add to an existing path (assuming the 2 paths meet)"""
    
    """
    the cases are organized by which endpoints match up, where 'path' is a path which goes from A->B, 
    and 'pts'/'ops' is another path which goes from C->D:
        B=C: 
            A->B, C->D
            A->B, B->D
            A->B->B->D
            A->B->D
        A=C:
            A->B, C->D
            A->B, A->D
            B->A, A->D
            B->A->A->D
            B->A->D
        B=D:
            A->B, C->D
            A->B, C->B
            A->B, B->C
            A->B->B->C
            A->B->C
        A=D:
            A->B, C->D
            A->B, C->A
            B->A, A->C
            B->A->A->C
            B->A->C
    in the event none of these cases are matched (which really shouldn't be possible), we do still need to 
    preserve both paths as A->B->C->D where the segment from B->C is left as a MoveTo operation
    
    (note that all paths must begin with a MoveTo operation)
    """
    
    if path:
        if points_are_equal(path.points[-2:], pts[:2]):
            metrics['B=C'] += 1
        elif points_are_equal(path.points[:2], pts[:2]):
            metrics['A=C'] += 1
            path.points, path.operators = reverse_path(path.points, path.operators)
        elif points_are_equal(path.points[-2:], pts[-2:]): # TODO: test this
            metrics['B=D'] += 1
            pts, ops = reverse_path(pts, ops)
        elif points_are_equal(path.points[:2], pts[-2:]): # TODO: test this
            metrics['A=D'] += 1
            pts, ops = reverse_path(pts, ops)
            path.points, path.operators = reverse_path(path.points, path.operators)
        else:
            metrics['BAD'] += 1
            ops = [None] + ops
            pts = [None]*2 + pts
        path.operators.extend(ops[1:])
        path.points.extend(pts[2:])
        return path
    else:
        return rl.shapes.Path(pts, ops, isClipPath, **kwargs)



"""Main Scripts"""


#  Make the Command Line Args be a Global Variable
args = None


def main(argv:list = None):
    """
    for usage, run:
        >>> main(["-h"])
    
    when passing argv in the console, all arguments/values must be separate strings in a list, and 
    you must include the double-hyphens for flags (i.e. use [..., "--margin", "0.3", ...], not 
    [..., "margin", 0.3, ...] - note that in the second case both the parameter name and the value 
    are invalid). Also, it is often helpful to remember that r-strings ignore backslash as an 
    escape character, which is useful when copy/pasting the path to the file(s)/output.
    
    """
    
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Prepare file(s) for Laser Cutting")
    parser.add_argument('filepath', type=str,
                        help='the path to the pdf/svg file(s) you want to prep for laser cutting')
    parser.add_argument('--width', type=float, default=31.75,
                        help='the width of the cutter bed in inches (default: 31.75in)')
    parser.add_argument('--height', type=float, default=19.75,
                        help='the height of the cutter bed in inches (default: 19.75in)')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='the margin as measured from the top left (default: 0.5in)')
    parser.add_argument('--output', type=str, default=None,
                        help='a path to the folder where prepped files should be stored '
                             + '(default: same directory as input file)')
    parser.add_argument('--passes', type=int, default=1,
                        help='the number of times to cut the entire document (default: 1)')
    parser.add_argument('--epsilon', type=float, default=1,
                        help='if the distance between 2 points (in pt) is less than epsilon, they are considered the same point (default: 1e-10)')
    parser.add_argument('--inkscape', type=str, default=r"C:\Program Files\Inkscape",
                        help=r'the directory to inkscape.exe (default: C:\Program Files\Inkscape)')
    parser.add_argument('--center', action='store_true', default=False,
                        help='indicates that cut should have equal margins on all sides (overrides --margin)')
    parser.add_argument('--centerv', action='store_true', default=False,
                        help='indicates that cut should have equal margins on the top/bottom (overrides --margin)')
    parser.add_argument('--centerh', action='store_true', default=False,
                        help='indicates that cut should have equal margins on the left/right (overrides --margin)')
    parser.add_argument('--preserve-overlap', action='store_true', default=False,
                        help='indicates that overlapping lines should not be unified')
    parser.add_argument('--uls', action='store_true', default=False,
                        help='indicates that the cutter is a ULS machine')
    parser.add_argument('--epilog', action='store_true', default=False,
                        help='indicates that the cutter is an Epilog machine')
    parser.add_argument('--preserve-rectangles', action='store_true', default=False,
                        help="use this if none of your parts are true rectangles")
    parser.add_argument('--autoscale', action='store_true', default=False,
                        help="set the height/width based on the size of the drawing")
    parser.add_argument('--save-svg', action='store_true', default=False,
                        help="save the pathed svg file (useful for debugging the pathing algorithm)")
    parser.add_argument('--debug', action='store_true', default=False,
                        help="enable debug mode")
    
    global args
    args = parser.parse_args(argv) if argv else parser.parse_args()

    # Check how Many Printers are Selected
    printers = sum([args.uls, args.epilog])
    if printers == 0:
        args.uls = True
        print("warning: no printer selected, defaulting to ULS")
    elif printers > 1:
        print("warning: multiple printers selected, this may result in formatting errors")

    # Get the Absolute Filepath
    args.filepath = os.path.abspath(args.filepath)

    # Set Format Variables by Printer
    if args.uls:
        args.strokeWidth = 0.01 # hairline (0.25? 0.5?)
        args.strokeColor = colors.red
        args.s_cut = 3.204843
        args.s_move = 1615.673
        args.t_offset = -25.67509
        # etch: blue
        # raster: black
    elif args.epilog:
        args.strokeWidth = 0.01
        args.strokeColor = colors.black
        args.s_cut = 1 # TODO: measure these speeds (inches/second)
        args.s_move = 1000
        args.t_offset = 0

    # Convert the Height/Width from inches to points
    args.height *= 72
    args.width *= 72
    args.margin *= 72

    # Make the Output Directory
    if args.output and not os.path.isdir(args.output):
        os.mkdir(args.output)
    
    # Check if the Filepath is a File or a Folder
    if is_valid_file(args.filepath):
        prep_file(args.filepath)
    elif os.path.isdir(args.filepath):
        for file in find_drawings(args.filepath):
            prep_file(file)
            print('')
    elif os.path.isfile(args.filepath):
        raise ValueError(f"'{args.filepath}' is not a valid pdf/svg file or directory containing pdf/svg files")
    else:
        raise FileNotFoundError(f"'{args.filepath}' not found")


"""functions for identifying svg files and working with filenames"""


# check if a given filename corresponds to a valid file for processing
def is_valid_file(filename: str) -> bool:
    return (os.path.isfile(filename)
            and (filename.endswith('.svg') or filename.endswith('.pdf'))
            and ' - pathed' not in filename
            and ' - ready' not in filename)


# find svgs within the given directory
def find_drawings(root: str = os.getcwd(), recursive: bool = True):
    # Search the Top-Level Folder
    folders = []
    for file in os.listdir(root):
        path = root + '\\' + file
        if is_valid_file(path):
            yield path
        elif recursive and os.path.isdir(path):
            folders.append(path)

    # Add the Lower-Level Matches ('folders' is empty when 'recursive' is False)
    for path in folders:
        yield from find_drawings(path)


# parse a filepath
def parse_filepath(path: str) -> (str, str, str):
    path = os.path.abspath(path)
    pieces = os.path.basename(path).split('.')
    name = '.'.join(pieces[:-1])
    suffix = pieces[-1]
    return os.path.dirname(path), name, suffix



"""functions for loading/editing existing svg files"""


def prep_file(filename: str):
    """Prepare an SVG File for Laser Cutting"""
    # Load the File
    svg = load(filename)

    # Get the Height/Width
    height, width = args.height, args.width

    # Get All the Lines
    elements = get_lines(svg)
    n = len(elements)

    # TODO: figure out if this actually saves cutting time
    # Remove Overlapping Lines
    if not args.preserve_overlap:
        elements = remove_overlap(elements)
        if args.debug: print(f"{n - len(elements)} overlapping elements removed")

    # Join the Line Segments into Paths
    paths = make_paths(elements)
    if args.debug: print(f"{n} elements reduced to {len(paths)} paths")

    # Sort the Paths
    paths = sort_paths(paths)
    
    # Estimate the Cutting Time
    t = None
    try: t = estimate_cutting_time(paths, args.s_move, args.s_cut)
    except: pass
    
    # Draw the SVG
    d = draw_svg(paths, height, width)
    
    # Save the SVG
    folder, name, suffix = parse_filepath(filename)
    if args.save_svg:
        svgname = name + ' - pathed'
        print(f"saving pathed svg in '{folder}'")
        d.save(formats = ['svg'], outDir = folder, fnRoot = svgname)
    
    # Save the SVG as a PDF
    pdfname = folder + '\\' + name + ' - pathed.pdf'
    renderPDF.drawToFile(d, pdfname)
    print(f"pdf saved at '{pdfname}'")
    
    # Clear the Endpoint Map (to avoid collisions when prepping multiple files)
    endpoint_list.clear()
    
    # Print the Estimated Cutting Time
    if t: print(f"estimated cutting time: {format_time(t)}")
    else: print("cutting time estimation failed")


def load(filename: str) -> rl.shapes.Drawing:
    """Load an SVG file as a reportlab graphics drawing"""

    #  Check the Suffix
    path, name, suffix = parse_filepath(filename)
    suffix = suffix.lower()
    if suffix == 'svg':
        print(f"loading '{filename}'")
        return svg2rlg(filename)
    elif suffix == 'pdf':
        out = path + '\\' + name + ' - temp.svg'
        print(f"converting '{filename}' to svg: '{out}'")
        cwd = os.getcwd()
        os.chdir(args.inkscape)
        os.system(f"inkscape \"{filename}\" --export-plain-svg \"{out}\"")
        os.chdir(cwd)
        print(f"loading '{out}'")
        svg = svg2rlg(out)
        print(f"removing temp file '{out}'")
        os.remove(out)
        return svg
    else:
        raise ValueError("only svgs are currently supported")


def draw_svg(elements: list, h0, w0, **svg_args) -> rl.shapes.Drawing:
    """Draw an SVG Using the Given Elements"""

    # Set the Formatting
    for e in elements:
        e.setProperties({'strokeWidth': args.strokeWidth, 'strokeColor': args.strokeColor})

    # Rescale the Elements
    transform(elements, 9 / 75, 0, 0)  # 9/25, 1/8

    # Initialize a Drawing
    temp = rl.shapes.Drawing()
    for e in elements:
        temp.add(e)

    # Get the Bounds to Set the Height/Width/Margins
    x0, y0, x1, y1 = temp.getBounds()
    w = x1 - x0
    h = y1 - y0
    
    # Reduce the Size of the PDF
    args.width = min(w + 2 * args.margin, args.width)
    args.height = min(h + 2 * args.margin, args.height)
    
    # Calculate the Margins
    mx = my = args.margin
    if args.center:
        mx = (args.width - w) / 2
        my = (args.height - h) / 2
    if args.centerv:
        my = (args.height - h) / 2
    if args.centerh:
        mx = (args.width - w) / 2

    # Set the Drawing Height/Width
    W, H = (w + 2 * mx, h + 2 * my) if args.autoscale else (args.width, args.height)

    # Set the Margins
    transform(elements, 1, mx - x0, my - y0)

    # Draw the SVG
    d = rl.shapes.Drawing(W, H, **svg_args)
    #d.setProperties({'stroke-width' : args.strokeWidth, 'stroke' : args.strokeColor})
    for _ in range(args.passes):
        for e in elements:
            d.add(e)

    # Return the Results
    return d


def points_are_equal(P: tuple, Q: tuple, measure = dist.l2) -> bool:
    return measure(*P, *Q) < args.epsilon


def path_is_closed(path: rl.shapes.Path) -> bool:
    return points_are_equal(path.points[:2], path.points[-2:]) or operators.ClosePath in path.operators


def get_lines(svg: rl.shapes.Drawing) -> (list, tuple):
    """extract the individual line segments (or arcs) in the svg into a single flattened list"""

    # Look for Elements in the SVG Contents List
    elements = []
    for x in svg.contents:
        if type(x) == rl.shapes.Group:
            # Recursively Extract the Items Inside the Group
            elements.extend(get_lines(x))
        elif type(x) == rl.shapes.PolyLine:
            # Break the Polyline into Individual Line Segments
            A = x.points[:2]
            for i in range(2, len(x.points), 2):
                B = x.points[i:i + 2]
                elements.append(Line(*A, *B))
                A = B
        elif type(x) in (rl.shapes.Path, rl.shapes.ArcPath):# and not path_is_closed(x):
            # TODO: update this section as reportLab paths get updated

            # break up the path into individual segments
            start = None
            prev = [0, 0]
            for o, controls in operators.parse(x.operators, x.points):
                # Check the Type of Line Segment
                if o == operators.MoveTo:
                    prev = controls
                    start = controls.copy() if start is None else start
                elif o == operators.LineTo:
                    elements.append(Line(*prev, *controls))
                    prev = controls
                elif o == operators.CurveTo:
                    elements.append(Path(prev + controls, [operators.MoveTo, operators.CurveTo]))
                    prev = controls[-2:]
                elif o == operators.ClosePath:
                    elements.append(Line(*prev, *start))
                    prev = start
        elif (type(x) != rl.shapes.Rect or args.preserve_rectangles) and 'reportlab' in str(type(x)):
            # Keep the Curve or Rectangle as-is
            elements.append(x)

    # Return the List of Elements
    return elements


def transform(elements: list, scale: float = 1, dx: float = 0, dy: float = 0):
    """rescale, then translate all the points for elements in a list (in place)"""

    # define helper functions for the transformation
    if scale == 1:
        if dx == 0:
            def tx(x):
                return x
        else:
            def tx(x):
                return x + dx

        if dy == 0:
            def ty(y):
                return y
        else:
            def ty(y):
                return y + dy
    else:
        if dx == 0:
            def tx(x):
                return x * scale
        else:
            def tx(x):
                return x * scale + dx

        if dy == 0:
            def ty(y):
                return y * scale
        else:
            def ty(y):
                return y * scale + dy

    def t(x, y):
        return tx(x), ty(y)

    # transform each element in-place
    for e in elements:
        if type(e) == rl.shapes.Line:
            e.x1, e.y1 = t(e.x1, e.y1)
            e.x2, e.y2 = t(e.x2, e.y2)
        elif type(e) in (rl.shapes.Path, rl.shapes.PolyLine, rl.shapes.Polygon):
            for i in range(0, len(e.points), 2):
                e.points[i] = tx(e.points[i])
                e.points[i + 1] = ty(e.points[i + 1])
        elif type(e) == rl.shapes.ArcPath:
            raise NotImplementedError("ArcPaths not yet supported")  # TODO
        elif type(e) == rl.shapes.Circle:
            e.cx, e.cy = t(e.cx, e.cy)
            e.r *= scale
        elif type(e) == rl.shapes.Ellipse:
            e.cx, e.cy = t(e.cx, e.cy)
            e.rx *= scale
            e.ry *= scale
        elif type(e) == rl.shapes.Rect:
            e.x, e.y = t(e.x, e.y)
            e.width *= scale
            e.height *= scale
            e.rx *= scale
            e.ry *= scale



def point_is_on_line(P: tuple, Q: tuple, X: tuple) -> bool:
    """Determine if the Point X is Within eps of the Line PQ"""
    
    """
    Formula Derivation (capital letters are points, lowercase letters are scalars)
    
    U = Q - P = (a, b, 0)
    V = X - P = (c, d, 0)
    
    d = |V|sin(θ)
    sin(θ) = |UxV|/|U||V|
    ∴ d = |UxV|/|U|
    
    UxV = (0, 0, ad - bc)
    ∴ |UxV| = |ad - bc|
    ∴ d^2 = (ad - bc)^2 / (a^2 + b^2)
    
    
    Possible Division by 0: a^2 + b^2 = 0 ∴ X = P
    Implicit Possible Division by 0: c^2 = d^2 = 0 ∴ X = Q
    
    These cases can easilly be tested for separately. Note that in the second 
    case, the division by 0 has been hidden since assumed |V|/|V| = 1 in the 
    derivation. This likely won't hurt the numerical stability, but it's still 
    worth avoiding by putting that in a separate check.
    """
    
    # Check the Division by 0 Cases
    eps2 = args.epsilon**2
    if dist.l2s(*P, *X) < eps2 or dist.l2s(*Q, *X) < eps2:
        return True
    
    # Unpack the Points
    x1, y1 = P
    x2, y2 = Q
    x3, y3 = X
    
    # Compute the Relavent Vector Components
    a = x2 - x1
    b = y2 - y1
    c = x3 - x1
    d = y3 - y1
    
    # Extra Safety Check
    D = a**2 + b**2
    if D == 0:
        # this means that a = b = 0, and thus a*d - b*c = 0 as well
        return True
    
    # Compute the Distance Squared From the Point to the Line
    d2 = (a*d - b*c)**2 / D
    
    # Compare that Distance Squared to the Tolerance Squared
    return d2 < eps2


def point_is_on_line_segment(P: tuple, Q: tuple, X: tuple):
    """check if a point X lies on the line segment PQ"""
    
    """
    I wish I remembered how I did this, but I remember thinking really hard 
    while drawing it out on the white board. The points_are_equal() statements 
    are testing edge cases though, so the heart of it is the last piece, where 
    we are only testing if the point X is on the segment PQ; nothing about 
    if the segment X is a part of is colinear/overlapping the segment PQ
    """
    
    # Get the Points
    (x1, y1), (x2, y2), (x3, y3) = P, Q, X

    # Check if the Point is on the Line Segment
    return (points_are_equal(P, X) or points_are_equal(Q, X) or
        (y3 < y1) != (y3 < y2) and (x3 < x1) != (x3 < x2) and point_is_on_line(P, Q, X))



def unify_lines(l1: rl.shapes.Line, l2: rl.shapes.Line):
    """test if 2 lines overlap, creating a new line which encompases both lines"""

    """
    I don't fully remember how I derived this, but just looking at it I think it's 
    pretty intuitive, no? Basically, if the lines aren't colinear, then at most 
    one of the tests can return True, and if they are colinear than there are a 
    few cases:
        Case 1: count = 4, full overlap
            A ----- B
            C ----- D
        Case 2: count = 0, no overlap
            A ----- B
                       C ----- D
        Case 3: count = 2, no overlap (but unification is possible, just not desired)
            A ----- B
                    C ----- D
        Case 4: count = 2, partial overlap (C on AB, B on CD)
            A ----- B
                C ----- D
        Case 5: count = 2, full overlap (C and D on AB)
            A ----- B
              C - D
        Case 6: count = 3, full overlap (1 endpoint is shared)
            A ----- B
              C --- D
        (other cases are symmetrical to cases 2-6)
        Case 7: count = 3, full overlap (1 endpoint is shared)
            A ----- B
            C --- D
        Case 7: count = 2, full overlap (A and B on CD)
              A - B
            C ----- D
        Case 8: count = 2, partial overlap (A on CD and D on AB)
                A ----- B
            C ----- D
        Case 9, count = 2, no overlap (but unification is possible, just not desired)
                    A ----- B
            C ----- D
        Case 10, count = 0, no overlap
                       A ----- B
            C ----- D
        (remaining cases are symmetrical by flipping the order of C and D, and are not shown)
    
    No matter the case, if unification is required, the optimal line segment is 
    the one which maximizes the length of the resulting segment. But in practice 
    we can speed up computation by using different approaches for different cases.
    
    Also, note that if count == 1 then the lines can't be parallel (and thus not 
    colinear), and if count == 0 then the lines can't overlap
    """
    
    # Get the Sanitized Endpoints
    A, B = get_endpoints(l1)
    C, D = get_endpoints(l2)
    points = [A, B, C, D]
    
    # Check for Lines which Share Endpoints but Aren't Parallel
    if not all([
                point_is_on_line(C, D, A),
                point_is_on_line(C, D, B),
                point_is_on_line(A, B, C),
                point_is_on_line(A, B, D)
            ]):
        return False
    
    # Record Which Points Exist Within the Other Line Segment (these points are not endpoints of the new line)
    endpoints = [
        point_is_on_line_segment(C, D, A),
        point_is_on_line_segment(C, D, B),
        point_is_on_line_segment(A, B, C),
        point_is_on_line_segment(A, B, D)
    ]
    
    # Count the Number of Tests which Returned True
    count = sum(endpoints)
    
    # Check if the Lines Overlap
    if count == 2:
        # Find the New Endpoints
        P, Q = [P for P, e in zip(points, endpoints) if not e]
        
        # Check if Unification is Desired
        if abs(dist.l2(*P, *Q) - (dist.l2(*A, *B) + dist.l2(*C, *D))) < args.epsilon:
            # The Lines Share 1 Endpoint but don't Overlap
            return False
        else:
            # Return the New Line Segment
            return Line(*P, *Q)
    elif count == 3:
        # The 2 Lines Share 1 Endpoint and Overlap
        # Select the Points which Maximize the Length of the Line Segment
        P = [P for P, e in zip(points, endpoints) if not e][0]
        Q = max(points, key = lambda Q : dist.l2s(*P, *Q))
        return Line(*P, *Q)
    elif count == 4:
        # The 2 Lines are the Same Line (i.e. A == C and B == D)
        return Line(*A, *B)
    else:
        # The Lines Do Not Overlap
        return False


def is_line(element):
    return (
        type(element) == rl.shapes.Line or 
        (
            type(element) == rl.shapes.Path and 
            len(element.operators) == 2 and 
            element.operators[0] == operators.MoveTo and 
            element.operators[1] == operators.LineTo
        )
    )

def remove_overlap(elements: list):
    """Simplify Overlapping Lines"""

    # Look at All Distinct Pairs of Lines
    simplified = []
    used = [0]*len(elements)
    for i in ctrange(len(elements), N=len(elements)*(len(elements) - 1)//2, desc='merging overlapping lines'):
        # Check if the line was already absorbed into a previous line
        if not used[i]:
            # Mark the Element as Used (this step is unnecessary)
            used[i] = 1
            
            # create a reference to store the merged lines
            line = elements[i]

            # Don't bother looking for overlap unless it's a straight line segment
            if is_line(elements[i]):
                # look at all the other lines which haven't been used yet
                for j in range(i + 1, len(elements)):
                    if is_line(elements[j]) and not used[j]:
                        # Try to Combine the Lines
                        union = unify_lines(line, elements[j])

                        # check if the jth line was sucessfully merged with the unified line
                        if union:
                            # save the new unified line, and mark that the jth line has been used
                            line = union
                            used[j] = 1

            # Add the line (or element) to the new list
            simplified.append(line)
    
    # Return the New Element List
    return simplified


endpoint_list = {}
def encode_point(P: tuple) -> tuple:
    if P not in endpoint_list:
        for Q in endpoint_list:
            if dist.l2s(*P, *Q) < args.epsilon:
                endpoint_list[P] = Q
                return Q
        endpoint_list[P] = P
    return endpoint_list[P]


def get_endpoints(element, encode:bool = True) -> tuple:
    """get the endpoints of a line segment (returns None for segments without endpoints, e.g. circles)"""

    P = Q = None
    endpoints = True
    if type(element) == rl.shapes.Line:
        P, Q = (element.x1, element.y1), (element.x2, element.y2)
    elif type(element) in (rl.shapes.Path, rl.shapes.ArcPath):
        if element.operators[-1] == operators.ClosePath:
            P = Q = tuple(element.points[:2])
        else:
            P, Q = tuple(element.points[:2]), tuple(element.points[-2:])
    elif type(element) in (rl.shapes.Circle, rl.shapes.Ellipse, rl.shapes.Rect):
        endpoints = False
    elif type(element) in (rl.shapes.PolyLine, rl.shapes.Polygon):
        raise NotImplementedError("polylines/polygons shouldn't exist at this point in the process")
    elif type(element) == rl.shapes.Group:
        raise NotImplementedError("groups shouldn't exist at this point in the process")
    else:
        raise ValueError(f"'{type(element)}' is not a recognized type of reportLab.shapes element")
    
    if endpoints:
        return (encode_point(P), encode_point(Q)) if encode else (P, Q)
    else:
        return None


def get_endpoint_map(elements: list) -> tuple:
    """
    create a dictionary of where the elements start/end, also return a list of elements
    which don't have endpoints (e.g. circles)
    
    (x_start, y_start) -> (index in elements list, ref to the element, (x_end, y_end))
    """

    # Check for an Empty Elements List
    if not elements:
        print("warning: no elements in list")
        return {}, []

    # Reverse an Arbitrary Elements
    def reverse(e, sx, sy, ex, ey):
        if type(e) == rl.shapes.Line:
            return Line(ex, ey, sx, sy)
        elif type(e) in (rl.shapes.Path, rl.shapes.PolyLine):
            # Reverse the Points
            grouped_points = [[e.points[i], e.points[i + 1]] for i in range(0, len(e.points), 2)][::-1]
            points = []
            for P in grouped_points:
                points.extend(P)

            # Reverse the Operators
            if e.operators[0] == operators.MoveTo:
                if e.operators[-1] == operators.ClosePath:
                    codes = [operators.MoveTo] + e.operators[1:-1][::-1] + [operators.ClosePath]
                else:
                    codes = [operators.MoveTo] + e.operators[1:][::-1]
            else:
                if e.operators[-1] == operators.ClosePath:
                    codes = [operators.MoveTo] + e.operators[:-1][::-1] + [operators.ClosePath]
                else:
                    codes = [operators.MoveTo] + e.operators[::-1]
                points = [ex, ey] + points

            # Build the Reversed Path
            return Path(points, codes)
        else:
            return e

    # Map the Endpoints
    endpoints = defaultdict(list)
    closed = []
    for i, e in enumerate(elements):
        endpts = get_endpoints(e)
        if endpts:
            P, Q = endpts
            endpoints[P].append((i, e, Q))
            endpoints[Q].append((i, reverse(e, *P, *Q), P))
            endpoints[i] = (P, Q)
        else:
            closed.append((i, e, None))

    # Return the Map and the List of Closed-Loop Elements
    return endpoints, closed


def make_paths(elements: list) -> list:
    """join line segments into paths (greedy), returns a list of path elements"""
    
    # Add an Individual Line to a Path
    def add_line(path, e, P, Q):
        if type(e) == rl.shapes.Line:
            if path.points:
                if points_are_equal(path.points[-2:], P):
                    path.lineTo(*Q)
                elif points_are_equal(path.points[-2:], P):
                    path.lineTo(*P)
                else:
                    path.moveTo(*P)
                    path.lineTo(*Q)
            else:
                path.moveTo(*P)
                path.lineTo(*Q)
        elif type(e) in (rl.shapes.Path, rl.shapes.ArcPath):
            Path(e.points, e.operators, path)
        else:
            raise TypeError("the element being added to the path must have endpoints")

    # Generate the Endpoint Map
    endpoints, closed = get_endpoint_map(elements)

    # Initialize an Array of Which Elements Already Exist in Paths
    paths = []
    used = [0] * len(elements)
    for i, e in closed:
        paths.append(e)
        used[i] = 1

    # Make Paths
    for i, e in enumerate(elements):
        # Check if the Current Element was Already Used
        if not used[i]:
            # Start a New Path
            P, Q = endpoints[i]
            path = rl.shapes.Path()
            paths.append(path)
            path.moveTo(*P)

            # Continue the Line as Long as Possible
            j = i
            while not used[j]:
                # Add the Current Line to the Path and Mark it as Used
                add_line(path, e, P, Q)
                used[j] = 1

                # Find a Line to Add to the Path (if one exists)
                # (otherwise j won't update, so "not used[j]" will evaluate false, breaking out of the while loop)
                A = P
                P = Q
                for j, e, Q in endpoints[A]:
                    if not used[j]:
                        break

    # Return the List of Paths
    return paths


def sort_paths(paths: list) -> list:
    """Greedily Arrange to Paths to try to Minimize Cutter Head Travel"""

    # Find the Path Which Starts Closest to a Point
    used = [0] * len(paths)

    def find_next(p: tuple) -> int:
        return min((j for j in range(len(paths)) if not used[j]), key=lambda j: dist.l2s(*p, *paths[j].points[-2:]))

    # Start at the Origin
    P = (0, 0)
    ordered = []
    for _ in ctrange(len(paths), N=len(paths)**2, desc="optimizing path order"):
        # Find the Next Path, and then Swap the Paths
        i = find_next(P)

        # Add the Path to the Ordered List
        ordered.append(paths[i])
        used[i] = 1

        # Get the End Point of the jth Path (which is the starting point of the travel to the next path)
        P = get_endpoints(paths[i])[1]

    # Return the Ordered Path List
    return ordered

# Estimate the Total Distance the Cutter Head Will Travel (speeds are in inches/second)
# TODO: measure the cutting speed
def estimate_cutting_time(paths: list, s_move: float = 7851.39, s_cut: float = 3.14376) -> float:
    
    # Estimate the Cutting Time for a Single Path (excluding the initial MoveTo)
    def estimate_path_time(path: rl.shapes.Path, s_move: float, s_cut: float) -> (float, int, int):
        t = 0
        start = path.points[:2]
        x, y = start
        for op, points in operators.parse(path.operators[1:], path.points[2:]):
            x, y, d, cut = operators.length(x, y, op, points, start)
            t += d / (s_cut if cut else s_move)
        
        # Return the Time in Seconds (converting from points to inches in the process: (points) * (seconds/inch) * (inches/point) = seconds)
        return t / 72
    
    # Start at the Origin
    t = 0
    x = 0
    y = 0
    for path in paths:
        # Get the Distance to the Start of the Path from the End of the Previous Path
        t += dist.l2(x, y, *path.points[:2]) / s_move
        
        # Get the Distnace of the Path
        t += estimate_path_time(path, s_move, s_cut)
        
        # Get the End Point of the Path
        _, (x, y) = get_endpoints(path, False)
    
    # Return the Cutting Time Estimate in Seconds
    return max(t + args.t_offset, 0)


# Utility Functions for Debugging
class utils:
    def getTypesAndOperators(elements: list) -> (set, set):
        types = set()
        operators = set()
        for e in elements:
            types.add(type(e))
            if hasattr(e, 'operators'):
                for o in e.operators:
                    operators.add(o)
        return types, operators

    def practiceRun(filepath_target: str) -> (rl.shapes.Drawing, list, list, list):
        svg = load(next((f for f in find_drawings(os.getcwd()) if filepath_target in f)))
        elements = get_lines(svg)
        simplified = remove_overlap(elements)
        paths = make_paths(simplified, *get_endpoint_map(simplified))
        return svg, elements, simplified, paths
    
    def sec(hours, minutes, seconds):
        return 3600 * hours + 60 * minutes + seconds

class timingData:
    # Cutting/Move Distances (pts) vs Total Cut Time (sec) for the ULS Dual CO2 Laser Cutter
    X = np.array([[1819, 105101], [2169, 78932], [2533, 70499], [5051, 168980], [1488, 47535], [3796, 57129], [1534, 36618], [1148, 47550]]).transpose()
    Y = np.array([592, 700, 826, 1661, 469, 1178, 487, 357])
    
    # Fit Function for the Time Data
    def time(X, a, b, c):
        return a * X[0] + b * X[1] + c
    
    def fit_dual_uls(invert_speeds:bool = True):
        params, var = curve_fit(timingData.time, timingData.X, timingData.Y)
        var = np.sqrt(np.diag(var))
        if invert_speeds:
            for i in range(2):
                var[i] /= params[i] ** 2 # dy/y = |-1| dx/x with y = x^-1 -> dy = dx / x^2
                params[i] = 1 / params[i]
        return params, var
    
    def estimate_error_dual_uls(params = None):
        # Compute the RMS %Error on the Data for the Dual CO2 ULS Laser
        if not params: params , _ = timingData.fit_dual_uls(False)
        return 100*np.sqrt(sum(((timingData.time(x, *params) - y)/y)**2 for x, y in zip(timingData.X.transpose(), timingData.Y))/len(timingData.Y))

# Run the Program
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
    except:
        print(sys.exc_info()[0])
        import traceback

        print(traceback.format_exc())
    finally:
        #input("\npress enter to continue...") TODO?: un-comment this for running in a terminal
        pass
