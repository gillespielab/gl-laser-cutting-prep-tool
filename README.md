## Preping a File for Laser Cutting
(in OnShape)

1. Design your part(s) in a part studio
2. Create a new part studio, create a derived object with the parts you want to cut, and then add material to account for the kerf of the laser where necessary
3. In a new part studio, create "parts" to represent the sheets of material you will be cutting from
4. Create an assembly of all the parts you want to cut, laid out on the sheets, flat-packed for laser cutting (fix all parts in place using mates)
5. Suppress the sheets in the assembly
6. Right click the flat-pack assembly tab to create a new drawing. Use a custom template (ISO A0, do not include border or title block)
7. Un-suppress the sheets in the flat pack assembly
8. Add relief cuts to the drawing using the line tool as necessary
9. Export each sheet as a pdf (right click the tab, and everything on the current sheet will be included in the pdf)
10. Run the laser cutting prep tool in a python terminal

## Adding Support for a New Machine
Currently, formatting settings are only supported for ULS and Epilog laser cutters. To add support for a new printer which has different formatting requirements, you need to edit LaserCuttingPrep.py. First, add a flag where the "# TODO : add a flag for new printers here" comment is, then set the parameters where the "TODO : add more printer setting here as needed" comment is. Lastly, update the list of printer options found where the " # TODO: add new printer flags here" comment is. For all of these steps it should be fairly straightforward use the ULS/Epilog settings code as an example.

## Notes
This tool is highly dependent on the reportlab python package, and may need to be updated as that library is updated.

## Dependencies
reportlab, svglib, scipy, pypdf, numpy
