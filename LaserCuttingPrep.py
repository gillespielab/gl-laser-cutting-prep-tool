# -*- coding: utf-8 -*-
"""
This tool streamlines the workflow for preparing files for laser cutters.

Instructions:
    1. Export the desired cuts as a pdf (DO NOT export svg files directly, that's currently broken)
    2. Convert the pdf to an svg using Inkscape, Illustrator, or svgconverter.com/pdf-to-svg
    3. Run this tool on the svg
    4. Check the results in Illustrator/Inkscape

USAGE:
    >>> LaserCuttingPrep.py <filename> <options>

RECOMMENDED:
    >>> LaserCuttingPrep.py <filename> --output <output path> --margin 0.3 --uls --centerv --debug

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

    def parse(codes: list, points: list) -> (int, list):
        """yields each operator code with its associated control points"""
        i = 0
        for o in codes:
            j = i + operators._Parameters[o]
            yield o, points[i:j]
            i = j


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


def main():
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
    args = parser.parse_args()

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
        # etch: blue
        # raster: black
    elif args.epilog:
        args.strokeWidth = 0.01
        args.strokeColor = colors.black

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

    # TODO: get this to work
    # Remove Overlapping Lines
    # elements = remove_overlap(elements)
    # if args.debug: print(f"{n - len(elements)} overlapping elements removed")

    # Join the Line Segments into Paths
    paths = make_paths(elements)
    if args.debug: print(f"{n} elements reduced to {len(paths)} paths")

    # Sort the Paths
    paths = sort_paths(paths)

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
        elif type(x) == rl.shapes.Path and not path_is_closed(x):
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


"""
def are_parallel(l1:draw.shapes.Line, l2:draw.shapes.Line) -> bool:
    " ""Test if 2 Lines are Parallel" ""
    
    # Check the Types of the Lines
    #if type(l1) != draw.elements.Line or type(l2) != draw.elements.Line:
    #    raise TypeError(f"both l1 and l2 must be type '{draw.elements.line}'")
    
    # Get the Points
    (a1, a2), (b1, b2) = get_endpoints(l1)
    (c1, c2), (d1, d2) = get_endpoints(l2)
    
    # Test if The Lines are Parallel
    return ((a1 - b1) * (c2 - d2) == (c1 - d1) * (a2 - b2) and
            (a1 - c1) * (b2 - d2) == (b1 - d1) * (a2 - c2) and 
            (a1 - d1) * (c2 - b2) == (c1 - b1) * (a2 - d2))
"""


def point_is_on_line_segment(P: tuple, A: tuple, B: tuple):
    """check if a point (x0, y0) lies on the line segment AB"""
    # Get the Points
    (x0, y0), (x1, y1), (x2, y2) = P, A, B

    # Check if the Point is on the Line Segment
    return ((y0 < y1) != (y0 < y2) and
            (x0 < x1) != (x0 < x2) and
            abs((x2 - x1) * (y0 - y1) - (y2 - y1) * (x0 - x1)) < args.epsilon)


def unify_lines(l1: rl.shapes.Line, l2: rl.shapes.Line):
    """test if 2 lines overlap, creating a new line which encompases both lines"""

    # Get the Points
    A, B = get_endpoints(l1)
    C, D = get_endpoints(l2)

    # Record Which Points Exist Within the Other Line Segment (these points are not endpoints of the new line)
    endpoints = [
        point_is_on_line_segment(A, C, D),
        point_is_on_line_segment(B, C, D),
        point_is_on_line_segment(C, A, B),
        point_is_on_line_segment(D, A, B)
    ]

    # Check if the Lines Overlap
    if sum(endpoints) == 2:
        # Find the New Endpoints
        points = [[A, B, C, D][i] for i in range(4) if not endpoints[i]]

        # Return the New Line Segment
        return Line(points[0][0], points[0][1], points[1][0], points[1][1])
    else:
        # The Lines Do Not Overlap
        return False


def remove_overlap(elements: list):
    """Simplify Overlapping Lines"""

    # Look at All Distinct Pairs of Lines
    simplified = []
    used = set()
    for i in ctrange(len(elements), N=len(elements)*(len(elements) - 1)//2, desc='merging overlapping lines'):
        # Check if the line was already absorbed into a previous line
        if i not in used:
            # create a reference to store the merged lines
            line = elements[i]

            # Don't bother looking for overlap unless it's a straight line segment
            if type(elements[i]) == rl.shapes.Line:
                # look at all the lines which haven't been visited yet
                for j in range(i + 1, len(elements)):
                    if type(elements[j]) == rl.shapes.Line and j not in used:
                        # Try to Combine the Lines
                        union = unify_lines(line, elements[j])

                        # check if the jth line was sucessfully merged with the unified line
                        if union:
                            # save the new unified line, and mark that the jth line has been used
                            line = union
                            used.add(j)

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


def get_endpoints(element) -> tuple:
    """get the endpoints of a line segment (returns None for segments without endpoints, e.g. circles)"""

    P = Q = None
    endpoints = False
    if type(element) == rl.shapes.Line:
        endpoints, P, Q = True, (element.x1, element.y1), (element.x2, element.y2)
    elif type(element) in (rl.shapes.Path, rl.shapes.ArcPath):
        endpoints, P, Q = True, tuple(element.points[:2]), tuple(element.points[-2:])
    elif type(element) in (rl.shapes.Circle, rl.shapes.Ellipse, rl.shapes.Rect):
        pass
    elif type(element) in (rl.shapes.PolyLine, rl.shapes.Polygon):
        raise NotImplementedError("polylines/polygons shouldn't exist at this point in the process")
    elif type(element) == rl.shapes.Group:
        raise NotImplementedError("groups shouldn't exist at this point in the process")
    else:
        raise ValueError(f"'{type(element)}' is not a recognized type of reportLab.shapes element")

    return encode_point(P), encode_point(Q) if endpoints else None


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
