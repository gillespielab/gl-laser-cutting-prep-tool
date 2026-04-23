## Preping a File for Laser Cutting
(primarily in OnShape)

1. Design your part(s) in a part studio
2. Create a new part studio, create a derived object with the parts you want to cut, and then add material to account for the kerf of the laser where necessary
3. In a new part studio, create "parts" to represent the sheets of material you will be cutting from, it is recommended that these sheets are 1/4"-1/2" smaller on all sides than the actual material you will purchase to leave space for a margin.
4. Create an assembly of all the parts you want to cut, laid out on the sheets, flat-packed for laser cutting (fix all parts in place using mates). It is recommended that you use the space as efficiently as possible, sharing sides when possible (making one laser cut do twice the work!).
5. Suppress the sheets in the assembly
6. Right click the flat-pack assembly tab to create a new drawing. Use a custom template (ISO A0, do not include border or title block)
7. Un-suppress the sheets in the flat pack assembly
8. Add relief cuts to the drawing using the line tool as necessary
9. Export each sheet as a pdf (right click the tab, and everything on the current sheet will be included in the pdf)
10. Run the laser cutting prep tool in a python terminal (usage instructions are available using flags such as -?, -h, or --help)

## Adding Support for a New Machine
Currently, formatting settings are only supported for ULS and Epilog laser cutters. To add support for a new printer which has different formatting requirements, you need to edit LaserCuttingPrep.py. First, add a flag where the "# TODO : add a flag for new printers here" comment is, then set the parameters where the "TODO : add more printer setting here as needed" comment is. Lastly, update the list of printer options found where the " # TODO: add new printer flags here" comment is. Unless a new category of formatting is required, it should be fairly straightforward to use the ULS/Epilog settings code as an example.

## Notes
This tool is highly dependent on the reportlab python package, and may need to be updated as that library is updated.

## Dependencies
reportlab, svglib, scipy, pypdf, numpy

## Prepping Files by Hand
To prepare a file by hand, first follow steps 1-9 to create the initial pdf, then follow the steps below:
1. Adjust line formatting as specified by the laser cutter manufacturer (e.g. at time of writing, ULS lasers cut on red lines with a thickness of 0.01pt, etch blue lines with a thickness of 0.01pt, and raster anything else unless otherwise programmed in the print settings on the machine).
2. While not strictly necessary (especially for simple components with fewer/longer edges), it is highly recommened you join line segments together in as efficient a way as possible (combining any overlapping lines into a single line segment as you go). This sets the paths the machine will actually follow while cutting, and if you skip this step the machine will likely cut line segments in a seemingly random order, drastically increasing the cut time, and often decreasing the cut quality (possibly requiring extra passes to complete compared to a well-pathed file).
